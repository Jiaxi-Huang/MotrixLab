# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Staged WM/BPTT learner consuming the derived modeling program.

Stage separation per the roadmap gradient contract:

- world model: supervised multi-step window loss on detached real targets
  (the assembly's flat vector; floating base scores Bryson-weighted channel
  groups of the integrated physical block);
- policy update: BPTT through the frozen dynamics and critic — their
  parameters carry no gradient while derivatives still flow through their
  inputs back to the actor. Imagined transitions run through the SAME term
  call records as the real env: ``evaluate`` for reward/termination/
  truncation, ``advance_exogenous`` for the command channels, ``observe``
  for both observation groups. The survival gate is upstream's soft
  termination ``1 - p`` (a task-declared value-group term) — truncations
  never gate imagination (clip-end is clamp-and-continue via the exogenous
  clamp rule);
- critic: GAE(lambda) regression over the same imagined rollout, returns
  detached (no target network, as upstream), clamped at the task's maximum
  discounted return.

The recipe constants upstream hard-codes live here too: actor Adam betas
(0.9, 0.95) and the actor gradient-value clip of 0.1. Optimizers follow
upstream ``build_optimizers``: Muon on the hidden Linear weights of each
module, fused capturable Adam on everything else, both at the module's lr —
and the actor steps unconditionally (upstream has no non-finite-grad skip;
such a check would force a host sync). Policy observation statistics come
from real collection and stay frozen inside the updates that consume them;
the critic consumes raw value observations (as upstream, whose critic input
carries no normalizer).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from motrix_env_core.manager.modeling import ModelingProgram
from motrix_rl.normalize import EmpiricalNormalization
from motrix_rl.replay import SequenceReplay
from motrix_rl_builtin.apg.assemble import Assembly
from motrix_rl_builtin.apg.capture import CapturedUpdate
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.networks import Dynamics, MlpActor, MlpCritic

_ACTOR_BETAS = (0.9, 0.95)
_ACTOR_GRAD_VALUE_CLIP = 0.1
_MAX_START_ROUNDS = 16


def _build_optimizers(
    module: nn.Module, lr: float, betas: tuple[float, float], device: torch.device
) -> list[torch.optim.Optimizer]:
    """Upstream ``build_optimizers``: Muon on the hidden Linear weights — every
    Linear between the first and the last of each ``nn.Sequential`` — and fused
    capturable Adam on the rest, both at ``lr``. Muon keeps its defaults
    (momentum 0.95, weight decay 0.1) with ``adjust_lr_fn="match_rms_adamw"``
    as upstream; its lr is a float fixed for the run (this trainer never
    schedules lr, matching upstream's frozen-at-capture semantics)."""
    hidden = [
        linear.weight
        for seq in module.modules()
        if isinstance(seq, nn.Sequential)
        for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
    hidden_ids = {id(parameter) for parameter in hidden}
    rest = [p for p in module.parameters() if id(p) not in hidden_ids]
    adam = torch.optim.Adam(rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True)
    if not hidden:
        return [adam]
    return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


class ApgLearner:
    """Owns the optimization stages; holds no environment and no task branch."""

    def __init__(
        self,
        *,
        program: ModelingProgram,
        assembly: Assembly,
        dynamics: Dynamics,
        actor: MlpActor,
        critic: MlpCritic,
        replay: SequenceReplay,
        cfg: ApgCfg,
        device: torch.device,
        obs_dim: int,
        value_dim: int,
        action_limit: float,
        value_max: float,
        goal_columns: tuple[torch.Tensor, ...] = (),
        soft_term_column: int | None = None,
    ):
        self.program = program
        self.assembly = assembly
        self.dynamics = dynamics.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.replay = replay
        self.cfg = cfg
        self.device = device
        self.action_limit = action_limit
        # Upstream clamps the critic's bootstrap and regression targets at the
        # task's maximum discounted return, suppressing value overestimation —
        # always on, as upstream.
        self.value_max = value_max
        # Per-encoder goal columns within the value group: the actor consumes
        # each encoder's block as a single current frame appended to the
        # stacked proprio history (upstream stacks sensors only, one encoder
        # per goal modality). Empty disables the goal input.
        self.goal_columns = tuple(columns.to(device) for columns in goal_columns)
        self.goal_slices = tuple(int(columns.shape[0]) for columns in self.goal_columns)
        self.goal_norms = tuple(EmpiricalNormalization(shape=width, device=device) for width in self.goal_slices)
        # Column of the one-dim soft-termination term in the value group;
        # None falls back to hard ``terminated`` gates.
        self.soft_term_column = soft_term_column
        self._optimizers = {
            "dynamics": _build_optimizers(dynamics, cfg.world_model.learning_rate, (0.9, 0.999), device),
            "actor": _build_optimizers(actor, cfg.actor.learning_rate, _ACTOR_BETAS, device),
            "critic": _build_optimizers(critic, cfg.critic.learning_rate, (0.9, 0.999), device),
        }
        amp = cfg.trainer.amp
        if amp not in (None, "bf16"):
            raise ValueError(f"trainer.amp must be null or 'bf16', got {amp!r}")
        amp_dtype = torch.bfloat16 if amp == "bf16" and device.type == "cuda" else None
        for module in (self.dynamics, self.actor, self.critic):
            module.amp_dtype = amp_dtype
        self.obs_norm = EmpiricalNormalization(shape=obs_dim, device=device)
        if device.type == "cuda":
            # Upstream compiles the hot modules on CUDA (eager on CPU tests).
            # Compiling the bound forwards leaves state_dict and optimizer
            # parameter identity untouched — checkpoints and Muon are blind
            # to it. The assembly's flat<->fields transforms (level-batched
            # FK + quaternion chains) compile as one graph each.
            for module in (self.dynamics, self.actor, self.critic):
                module.forward = torch.compile(module.forward)
            self._fields_to_flat = torch.compile(self.assembly.fields_to_flat)
            self._flat_to_fields = torch.compile(self.assembly.flat_to_fields)
        else:
            self._fields_to_flat = self.assembly.fields_to_flat
            self._flat_to_fields = self.assembly.flat_to_fields
        # Whole-update CUDA-graph capture (upstream CapturedUpdate): the
        # dispatch-heavy unroll records once and replays as one graph. The
        # engines fall back to eager calls on CPU. Replay order is the
        # iteration's own order (WM batches, then the policy update), which
        # is also the record order.
        self._wm_engine = CapturedUpdate(
            self._wm_update_core, state=lambda: self._graph_state(optimizer_keys=("dynamics",))
        )
        self._policy_engine = CapturedUpdate(
            self._policy_update_core, state=lambda: self._graph_state(optimizer_keys=("actor", "critic"))
        )
        # Imagination runs at the actor batch width, not the env's lane count:
        # one resized shadow host, its exogenous channels re-seeded from every
        # start batch.
        self._imagination: ModelingProgram | None = None
        self._channels = tuple(program_exogenous_channels(program))

    # -- imagination host -------------------------------------------------------

    def _imagination_program(self) -> ModelingProgram:
        if self._imagination is None:
            self._imagination = self.program.resized(self.cfg.actor.batch_size)
        return self._imagination

    def _normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        """Normalize each frame of a ``(B, H, D)`` stack, then flatten."""
        batch, length, dim = history.shape
        flat = self.obs_norm(history.reshape(batch * length, dim), update=False)
        return flat.reshape(batch, length * dim)

    def _actor_features(self, history: torch.Tensor, goal: torch.Tensor | None) -> torch.Tensor:
        """Concatenated actor input: normalized stacked proprio history, then
        each encoder's normalized flat goal block, in encoder order — the
        GoalConditionedActor's contract (each block passes its own encoder
        inside the actor)."""
        stacked = self._normalize_history(history)
        if not self.goal_columns or goal is None:
            return stacked
        blocks = (
            norm(block, update=False)
            for norm, block in zip(self.goal_norms, goal.split(self.goal_slices, dim=-1), strict=True)
        )
        return torch.cat((stacked, *blocks), dim=-1)

    def goal_of(self, value_obs: torch.Tensor) -> torch.Tensor | None:
        """Concatenated raw goal blocks of value-group observations (None
        without goal encoders); ``_actor_features`` splits by encoder width."""
        if not self.goal_columns:
            return None
        return torch.cat([value_obs[..., columns] for columns in self.goal_columns], dim=-1)

    # -- world model -------------------------------------------------------------

    def update_world_model(self) -> dict[str, torch.Tensor]:
        """Multi-step window MSE: unroll the dynamics over stored actions.

        The loss stays on device and the caller must never read it directly:
        the round's single metric sync happens after the policy update.
        """
        world = self.cfg.world_model
        batch = self.replay.sample_windows(world.batch_size, world.window)
        return dict(self._wm_engine(batch))

    def _wm_update_core(self, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """One supervised window batch: 8-step unroll, MSE, dynamics step.

        Capture-safe: fixed shapes, no host syncs; the metric tensor is read
        by the caller after the round's replay.
        """
        world = self.cfg.world_model
        # (B, K, flat) supervision; every step's target is its own stored
        # reset-free successor (the done step's is the pre-reset terminal the
        # assembler wrote). Fields the assembly derives (FK rows, duplicate
        # joint views) are exact, so the residual scheme supervises them too
        # while floating base supervises the physical block they derive from.
        states = self._fields_to_flat(batch["states"]).to(self.device)
        targets = self._fields_to_flat(batch["next_states"]).to(self.device)
        actions = batch["actions"].to(self.device)
        valid = batch["valid"].to(self.device).float()
        loss = states.new_zeros(())
        prediction = states[:, 0]
        for step in range(world.window):
            prediction = self.dynamics(prediction, actions[:, step])
            step_error = self.dynamics.prediction_error(prediction, targets[:, step])
            loss = loss + (step_error * valid[:, step]).sum()
        loss = self.dynamics.reduce_window_loss(loss, valid)
        for optimizer in self._optimizers["dynamics"]:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics.parameters(), world.max_grad_norm)
        for optimizer in self._optimizers["dynamics"]:
            optimizer.step()
        return {"train/wm_loss": self.dynamics.loss_metric(loss.detach(), world.window)}

    def update_dynamics_statistics(self, states: dict, actions: torch.Tensor, next_states: dict) -> None:
        """Update structured-model scales from one real full-batch step."""
        with torch.no_grad():
            self.dynamics.update_statistics(
                self._fields_to_flat(states).to(self.device),
                actions.to(self.device),
                self._fields_to_flat(next_states).to(self.device),
            )

    # -- imagination -------------------------------------------------------------

    def _sample_valid_starts(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Imagination starts with intact observation history, filled to batch."""
        batch_size = self.cfg.actor.batch_size
        history_len = self.cfg.actor.history
        filled = 0
        states: dict[str, list[torch.Tensor]] = {}
        histories: list[torch.Tensor] = []
        rounds = 0
        while filled < batch_size:
            rounds += 1
            if rounds > _MAX_START_ROUNDS:
                raise ValueError("Imagination start sampling could not fill the actor batch; check the replay fill")
            remaining = batch_size - filled
            candidate = self.replay.sample_starts(2 * remaining)
            mask = candidate["intact"].to(self.device)
            accepted = int(mask.sum())
            if accepted == 0:
                continue
            picked = torch.argsort(~mask, stable=True)[: min(remaining, accepted)]
            for name, value in candidate["states"].items():
                states.setdefault(name, []).append(value[picked].to(self.device))
            histories.append(
                self.replay.history(candidate["envs"][picked], candidate["positions"][picked], "obs", history_len)
            )
            filled += int(picked.shape[0])
        return {name: torch.cat(chunks) for name, chunks in states.items()}, torch.cat(histories).to(self.device)

    def _unroll(
        self, start_states: dict[str, torch.Tensor], history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential policy/dynamics unroll through the modeling program.

        Per step (matching the real env's step order): the actor consumes the
        history ending at ``obs(s_t)``; the dynamics predicts ``s_{t+1}``;
        ``evaluate(s_{t+1}, a_t)`` applies the action to the shadow state and
        computes reward/termination/truncation; ``advance_exogenous``
        advances the command channels (the frame index clamps at the clip
        end — clamp-and-continue, so truncations never fire in imagination);
        ``observe(s_{t+1})`` produces both observation groups and pushes the
        policy history.

        The survival gate is upstream's soft termination ``1 - p(s_{t+1})``,
        read from the configured one-dim value-group term; without one it
        falls back to the hard ``terminated`` union (truncations still do
        not gate — time-limit ends bootstrap rather than kill).

        Returns the ``(H, B)`` rewards and survival gates plus the
        ``(H+1, B, D)`` value-group observations of every rollout state.
        """
        cfg = self.cfg
        horizon = cfg.imagination.horizon
        program = self._imagination_program()
        # Batch boundary: cut the persistent shadow state from the previous
        # unroll's (already freed) graph before seeding this batch's channels.
        program.rebind_shadow_state(detach=True)
        program.seed_exogenous({name: start_states[name] for name in self._channels if name in start_states})
        flat = self._fields_to_flat(start_states)
        rewards: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        first = program.observe(self._flat_to_fields(flat))
        value_obs = [first[cfg.value_group]]
        for _ in range(horizon):
            # The actor consumes the history ending at obs(s_t) plus the
            # current goal window; the dynamics predicts s_{t+1}; then
            # evaluate/observe run on s_{t+1} exactly as the real env's
            # post-physics evaluate does (reward and observations live on the
            # successor state, the applied action visible in both).
            action = self.action_limit * self.actor(self._actor_features(history, self.goal_of(value_obs[-1])))
            flat = self.dynamics(flat, action)
            fields = self._flat_to_fields(flat)
            # Fresh shadow state: under autograd every mutable leaf must
            # be written exactly once (BPTT version safety).
            program.rebind_shadow_state()
            reward, terminated, _truncated = program.evaluate(fields, action)
            program.advance_exogenous(action)
            observed = program.observe(fields)
            value_obs.append(observed[cfg.value_group])
            rewards.append(reward)
            if self.soft_term_column is not None:
                gates.append(1.0 - value_obs[-1][..., self.soft_term_column])
            else:
                gates.append((~terminated).float())
            history = torch.cat((history[:, 1:], observed[cfg.policy_group].unsqueeze(1)), dim=1)
        return torch.stack(rewards), torch.stack(gates), torch.stack(value_obs)

    def _horizon_return(self, rewards: torch.Tensor, gates: torch.Tensor, terminal_value: torch.Tensor) -> torch.Tensor:
        """Discounted survival-weighted return plus bootstrapped terminal value."""
        gamma = self.cfg.gamma
        batch_return = torch.zeros(rewards.shape[1], device=self.device)
        survival = torch.ones(rewards.shape[1], device=self.device)
        for step in range(rewards.shape[0]):
            batch_return = batch_return + (gamma**step) * survival * rewards[step]
            survival = survival * gates[step]
        return batch_return + (gamma ** rewards.shape[0]) * survival * terminal_value

    # -- policy update ------------------------------------------------------------

    def policy_update(self) -> dict[str, float]:
        """One actor BPTT step plus GAE critic regression on the rollout."""
        start_states, history = self._sample_valid_starts()
        outputs = self._policy_engine({"states": start_states, "history": history})
        return {key: float(value) for key, value in outputs.items()}

    def _policy_update_core(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        """The whole policy update on staged static inputs: unroll, actor
        backward with clips, critic regression, actor optimizer steps.

        Capture-safe: fixed shapes, no host syncs, no non-finite-grad skip
        (upstream steps unconditionally — such a check would force a sync).
        Returns metric tensors read by the caller after the replay."""
        cfg = self.cfg
        start_states, history = inputs["states"], inputs["history"]

        # Freeze the dynamics and critic for the actor stage: gradients must
        # not touch their parameters while still flowing through their inputs.
        _set_trainable(self.dynamics, False)
        _set_trainable(self.critic, False)
        try:
            rewards, gates, value_obs = self._unroll(start_states, history)
            terminal_value = self.critic(value_obs[-1]).squeeze(-1).clamp(max=self.value_max)
            batch_return = self._horizon_return(rewards, gates, terminal_value)
            actor_loss = -batch_return.mean()
            for optimizer in self._optimizers["actor"]:
                optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.imagination.max_grad_norm)
            nn.utils.clip_grad_value_(self.actor.parameters(), _ACTOR_GRAD_VALUE_CLIP)
        finally:
            _set_trainable(self.dynamics, True)
            _set_trainable(self.critic, True)

        # The critic regresses on the detached rollout while the actor's
        # gradients sit untouched; the parameters are disjoint, so stepping
        # the actor afterwards is equivalent to stepping before.
        critic_loss = self._update_critic(rewards.detach(), gates.detach(), value_obs.detach())
        for optimizer in self._optimizers["actor"]:
            optimizer.step()
        return {
            "train/policy_loss": actor_loss.detach().mean(),
            "train/imag_reward": -actor_loss.detach().mean(),
            "train/policy_grad_norm": grad_norm.detach(),
            "train/value_loss": critic_loss,
        }

    def _graph_state(self, *, optimizer_keys: tuple[str, ...]) -> list[torch.Tensor]:
        """What one captured update mutates or reads by tensor identity:
        every module's parameters and buffers (all three nets feed both
        updates), plus the named optimizers' state and lr tensors. Per-engine
        optimizer sets — the WM graph records before the policy optimizers'
        state exists, so a shared view would misreport lazy init as a move.
        """
        state: list[torch.Tensor] = []
        for module in (self.dynamics, self.actor, self.critic):
            state += [*module.parameters(), *module.buffers()]
        for key in optimizer_keys:
            for optimizer in self._optimizers[key]:
                state += [t for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t)]
                state += [group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])]
        return state

    def _update_critic(self, rewards: torch.Tensor, gates: torch.Tensor, value_obs: torch.Tensor) -> torch.Tensor:
        """GAE(lambda) regression over the imagined rollout, targets detached.

        Upstream ``gae_returns`` verbatim with soft-done gates: TD deltas
        accumulated into advantages, returns ``A_t + V(s_t)`` clamped at the
        task's maximum discounted return. The critic reads raw value-group
        observations (upstream's critic input carries no normalizer).

        Returns the mean minibatch loss as a device tensor; the caller folds
        it into the round's single metric row.
        """
        cfg = self.cfg
        horizon = cfg.imagination.horizon
        with torch.no_grad():
            values = self.critic(value_obs).squeeze(-1)  # (H+1, B)
            next_value = values[-1]
            advantage = torch.zeros_like(next_value)
            targets = []
            for step in range(horizon - 1, -1, -1):
                not_done = gates[step]
                delta = rewards[step] + not_done * cfg.gamma * next_value - values[step]
                advantage = delta + not_done * cfg.gamma * cfg.critic.lam * advantage
                targets.append(advantage + values[step])
                next_value = values[step]
            targets = torch.stack(targets[::-1]).clamp(max=self.value_max)  # (H, B)

        inputs = value_obs[:-1].flatten(0, 1)  # (H*B, D)
        flat_targets = targets.flatten(0, 1)
        total = inputs.shape[0]
        minibatch_size = total // cfg.critic.minibatches
        if minibatch_size == 0:
            raise ValueError("critic.minibatches exceeds the imagined transitions")
        loss_sum = inputs.new_zeros(())
        updates = cfg.critic.epochs * cfg.critic.minibatches
        for _ in range(cfg.critic.epochs):
            permutation = torch.randperm(total, device=self.device)
            for index in range(cfg.critic.minibatches):
                picked = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                loss = nn.functional.mse_loss(self.critic(inputs[picked]).squeeze(-1), flat_targets[picked])
                for optimizer in self._optimizers["critic"]:
                    optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic.max_grad_norm)
                for optimizer in self._optimizers["critic"]:
                    optimizer.step()
                loss_sum = loss_sum + loss.detach()
        return loss_sum / updates

    # -- schedule and checkpoint ---------------------------------------------------

    def reset_learning_rates(self) -> None:
        """Reset every optimizer to its configured base learning rate.

        Upstream keeps a fixed lr throughout training; restoring an optimizer
        state_dict would otherwise import whatever lr the checkpoint carried.
        Fused capturable Adam carries lr as a device tensor, reset in place.
        """
        bases = {
            "dynamics": self.cfg.world_model.learning_rate,
            "actor": self.cfg.actor.learning_rate,
            "critic": self.cfg.critic.learning_rate,
        }
        for name, optimizers in self._optimizers.items():
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    if torch.is_tensor(group["lr"]):
                        group["lr"].fill_(bases[name])
                    else:
                        group["lr"] = bases[name]

    def learning_state(self) -> dict[str, Any]:
        """Serializable learning state: modules, normalizers, and optimizers."""
        return {
            "obs_norm": self.obs_norm.state_dict(),
            **{f"goal_norm_{index}": norm.state_dict() for index, norm in enumerate(self.goal_norms)},
            "dynamics": self.dynamics.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {
                name: [optimizer.state_dict() for optimizer in optimizers]
                for name, optimizers in self._optimizers.items()
            },
        }

    def load_learning_state(self, state: dict[str, Any]) -> None:
        self.obs_norm.load_state_dict(state["obs_norm"])
        for index, norm in enumerate(self.goal_norms):
            key = f"goal_norm_{index}"
            if state.get(key) is not None:
                norm.load_state_dict(state[key])
        self.dynamics.load_state_dict(state["dynamics"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        for name, optimizers in self._optimizers.items():
            # Optimizer lists are format-locked (Muon split since the sixth
            # alignment round): pre-split checkpoints fail loudly here.
            for optimizer, optimizer_state in zip(optimizers, state["optimizers"][name], strict=True):
                optimizer.load_state_dict(optimizer_state)
        self.reset_learning_rates()


def program_exogenous_channels(program: ModelingProgram) -> list[str]:
    """Channel names of a program's exogenous entries (its private host's)."""
    return [entry.channel for entry in program._exogenous_entries]  # noqa: SLF001

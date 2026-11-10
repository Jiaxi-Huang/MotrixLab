# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Staged WM/BPTT learner consuming the derived modeling program.

Stage separation per the roadmap gradient contract:

- world model: supervised multi-step window MSE on detached real targets
  (the assembly's flat vector; floating base supervises the physical block);
- policy update: BPTT through the frozen dynamics and critic — their
  parameters carry no gradient while derivatives still flow through their
  inputs back to the actor. Imagined transitions run through the SAME term
  call records as the real env: ``evaluate`` for reward/termination/
  truncation, ``advance_exogenous`` for the command channels, ``observe``
  for both observation groups;
- critic: lambda-return regression over the same imagined rollout, targets
  detached (no target network, as upstream).

The recipe constants upstream hard-codes live here too: actor Adam betas
(0.9, 0.95) and the actor gradient-value clip of 0.1. Observation statistics
come from real collection (policy input) and imagined rollouts (value input,
as upstream's critic norm role); both stay frozen inside the updates that
consume them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from motrix_env_core.manager.modeling import ModelingProgram
from motrix_rl.normalize import EmpiricalNormalization
from motrix_rl.replay import SequenceReplay
from motrix_rl_builtin.apg.assemble import Assembly
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.networks import Dynamics, MlpActor, MlpCritic

_ACTOR_BETAS = (0.9, 0.95)
_ACTOR_GRAD_VALUE_CLIP = 0.1
_MAX_START_ROUNDS = 16


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
        self._dynamics_opt = torch.optim.Adam(dynamics.parameters(), cfg.world_model.learning_rate)
        self._actor_opt = torch.optim.Adam(actor.parameters(), cfg.actor.learning_rate, betas=_ACTOR_BETAS)
        self._critic_opt = torch.optim.Adam(critic.parameters(), cfg.critic.learning_rate)
        amp = cfg.trainer.amp
        if amp not in (None, "bf16"):
            raise ValueError(f"trainer.amp must be null or 'bf16', got {amp!r}")
        amp_dtype = torch.bfloat16 if amp == "bf16" and device.type == "cuda" else None
        for module in (self.dynamics, self.actor, self.critic):
            module.amp_dtype = amp_dtype
        self.obs_norm = EmpiricalNormalization(shape=obs_dim, device=device)
        self.value_norm = EmpiricalNormalization(shape=value_dim, device=device)
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

    # -- world model -------------------------------------------------------------

    def update_world_model(self) -> dict[str, torch.Tensor]:
        """Multi-step window MSE: unroll the dynamics over stored actions.

        The loss stays on device and the caller must never read it directly:
        the round's single metric sync happens after the policy update.
        """
        world = self.cfg.world_model
        batch = self.replay.sample_windows(world.batch_size, world.window)
        # (B, K, flat) supervision; every step's target is its own stored
        # reset-free successor (the done step's is the pre-reset terminal the
        # assembler wrote). Fields the assembly derives (FK rows, duplicate
        # joint views) are exact, so the residual scheme supervises them too
        # while floating base supervises the physical block they derive from.
        states = self.assembly.fields_to_flat(batch["states"]).to(self.device)
        targets = self.assembly.fields_to_flat(batch["next_states"]).to(self.device)
        actions = batch["actions"].to(self.device)
        valid = batch["valid"].to(self.device).float()
        loss = states.new_zeros(())
        prediction = states[:, 0]
        for step in range(world.window):
            prediction = self.dynamics(prediction, actions[:, step])
            step_error = self.dynamics.prediction_error(prediction, targets[:, step])
            loss = loss + (step_error * valid[:, step]).sum()
        loss = self.dynamics.reduce_window_loss(loss, valid)
        self._dynamics_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics.parameters(), world.max_grad_norm)
        self._dynamics_opt.step()
        return {"apg.wm_loss": self.dynamics.loss_metric(loss.detach(), world.window)}

    def update_dynamics_statistics(self, states: dict, actions: torch.Tensor, next_states: dict) -> None:
        """Update structured-model scales from one real full-batch step."""
        with torch.no_grad():
            self.dynamics.update_statistics(
                self.assembly.fields_to_flat(states).to(self.device),
                actions.to(self.device),
                self.assembly.fields_to_flat(next_states).to(self.device),
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
            histories.append(self.replay.history(candidate["envs"][picked], candidate["positions"][picked], "obs", history_len))
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
        advances the command channels; ``observe(s_{t+1})`` produces both
        observation groups and pushes the policy history.

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
        flat = self.assembly.fields_to_flat(start_states)
        rewards: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        value_obs = [program.observe(self.assembly.flat_to_fields(flat))[cfg.value_group]]
        for _ in range(horizon):
            fields = self.assembly.flat_to_fields(flat)
            action = self.action_limit * self.actor(self._normalize_history(history))
            # Fresh shadow state: under autograd every mutable leaf must
            # be written exactly once (BPTT version safety).
            program.rebind_shadow_state()
            reward, terminated, truncated = program.evaluate(fields, action)
            rewards.append(reward)
            gates.append(~(terminated | truncated))
            program.advance_exogenous(action)
            observed = program.observe(fields)
            value_obs.append(observed[cfg.value_group])
            history = torch.cat((history[:, 1:], observed[cfg.policy_group].unsqueeze(1)), dim=1)
            flat = self.dynamics(flat, action)
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
        """One actor BPTT step plus lambda-return critic regression on the rollout."""
        cfg = self.cfg
        start_states, history = self._sample_valid_starts()

        # Freeze the dynamics and critic for the actor stage: gradients must
        # not touch their parameters while still flowing through their inputs.
        _set_trainable(self.dynamics, False)
        _set_trainable(self.critic, False)
        try:
            rewards, gates, value_obs = self._unroll(start_states, history)
            terminal_value = self.critic(self.value_norm(value_obs[-1], update=False)).squeeze(-1)
            batch_return = self._horizon_return(rewards, gates, terminal_value)
            actor_loss = -batch_return.mean()
            self._actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.imagination.max_grad_norm)
            nn.utils.clip_grad_value_(self.actor.parameters(), _ACTOR_GRAD_VALUE_CLIP)
            grad_finite = torch.isfinite(grad_norm)
        finally:
            _set_trainable(self.dynamics, True)
            _set_trainable(self.critic, True)

        # The critic regresses on the detached rollout while the actor's
        # gradients sit untouched; the parameters are disjoint, so stepping
        # the actor afterwards is equivalent to stepping before.
        critic_loss = self._update_critic(rewards.detach(), gates.detach(), value_obs.detach())
        row = torch.stack(
            (
                actor_loss.detach().mean(),
                -actor_loss.detach().mean(),
                grad_norm.detach(),
                grad_finite.float(),
                critic_loss,
            )
        ).cpu()
        if bool(row[3]):
            self._actor_opt.step()
        return {
            "apg.policy_loss": float(row[0]),
            "apg.imag_reward": float(row[1]),
            "apg.policy_grad_norm": float(row[2]),
            "apg.value_loss": float(row[4]),
        }

    def _update_critic(self, rewards: torch.Tensor, gates: torch.Tensor, value_obs: torch.Tensor) -> torch.Tensor:
        """Lambda-return regression over the imagined rollout, targets detached.

        Returns the mean minibatch loss as a device tensor; the caller folds
        it into the round's single metric row.
        """
        cfg = self.cfg
        horizon = cfg.imagination.horizon
        # Normalize the whole rollout once; the frozen statistics make the
        # normalized inputs valid for both the value pass and every
        # regression minibatch. Statistics fold the rollout observations
        # (upstream's critic norm role).
        with torch.no_grad():
            features = value_obs.flatten(0, 1)
            self.value_norm.update(features)
            inputs_all = self.value_norm(value_obs.flatten(0, 1), update=False).reshape(value_obs.shape)
            values = self.critic(inputs_all).squeeze(-1)  # (H+1, B)
            target = values[-1]
            targets = []
            for step in range(horizon - 1, -1, -1):
                target = rewards[step] + cfg.gamma * gates[step] * (
                    (1.0 - cfg.critic.lam) * values[step + 1] + cfg.critic.lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])  # (H, B)

        inputs = inputs_all[:-1].flatten(0, 1)  # (H*B, D)
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
                self._critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic.max_grad_norm)
                self._critic_opt.step()
                loss_sum = loss_sum + loss.detach()
        return loss_sum / updates

    # -- schedule and checkpoint ---------------------------------------------------

    def set_learning_rate_scale(self, scale: float) -> None:
        """Apply a shared decay factor to each configured base learning rate."""
        for optimizer, base in (
            (self._dynamics_opt, self.cfg.world_model.learning_rate),
            (self._actor_opt, self.cfg.actor.learning_rate),
            (self._critic_opt, self.cfg.critic.learning_rate),
        ):
            for group in optimizer.param_groups:
                group["lr"] = base * scale

    def learning_state(self) -> dict[str, Any]:
        """Serializable learning state: modules, normalizers, and optimizers."""
        return {
            "obs_norm": self.obs_norm.state_dict(),
            "value_norm": self.value_norm.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {
                "dynamics": self._dynamics_opt.state_dict(),
                "actor": self._actor_opt.state_dict(),
                "critic": self._critic_opt.state_dict(),
            },
        }

    def load_learning_state(self, state: dict[str, Any]) -> None:
        self.obs_norm.load_state_dict(state["obs_norm"])
        self.value_norm.load_state_dict(state["value_norm"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self._dynamics_opt.load_state_dict(state["optimizers"]["dynamics"])
        self._actor_opt.load_state_dict(state["optimizers"]["actor"])
        self._critic_opt.load_state_dict(state["optimizers"]["critic"])


def program_exogenous_channels(program: ModelingProgram) -> list[str]:
    """Channel names of a program's exogenous entries (its private host's)."""
    return [entry.channel for entry in program._exogenous_entries]  # noqa: SLF001

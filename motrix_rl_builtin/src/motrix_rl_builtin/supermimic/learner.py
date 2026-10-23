# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic world-model learner over the MotrixLab environment contract.

Ported from UniSuperMimic ``rl/algos/supermimic/learner.py``: a learned
latent dynamics model predicts frame-local state increments, the analytic
task heads (reward, soft termination, observations, goals) stay
differentiable, the goal-conditioned policy trains on imagined rollouts with
a λ-returns critic, and collection alternates with updates at a fixed
update-to-data ratio.

Environment coupling: the task env is a NumPy manager environment whose
motion command publishes the per-environment state, frame, and
simulation-context buffers that form each transition; the torch MDP
(analytic heads over the same corpus) is built by the trainer from those
surfaces and passed in.
"""

from __future__ import annotations

import math
import time
from typing import Any
from typing import cast as type_cast

import numpy as np
import torch
import torch.nn as nn

from . import networks
from .config import SuperMimicAgentCfg
from .contracts import State, require_supermimic_env
from .geometry import unflatten_state
from .networks import Critic, Policy, WorldModel
from .replay import ReplayBuffer

_ANNOTATE = type_cast

# Per-row squared error cap for the world-model loss. Transitions beyond this
# normalized error (contact impacts, solver pops) are chaotic: their gradient
# is excluded so the fit on the predictable mass keeps improving.
_WM_ERROR_CAP = 4.0


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SuperMimicLearner:
    """Owns the networks, replay, and the collect/update training phases."""

    def __init__(
        self,
        env,
        cfg: SuperMimicAgentCfg,
        mdp,
        *,
        device: torch.device,
        action_low: np.ndarray,
        action_high: np.ndarray,
        goal_frames: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = require_supermimic_env(env)
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        self.cfg = cfg
        self.goal_frames = int(cfg.goal_frames if goal_frames is None else goal_frames)
        self.device = device
        self.mdp = mdp.to(device)

        # TF32 speeds up Ampere+ NVIDIA matmuls, but the ROCm TF32 path is
        # pathologically slow for the small 3x3 bmms this MDP issues (tens of
        # milliseconds of host-side time per call on ROCm 7.x), so keep it off
        # there. bf16 autocast (amp) covers the large-matmul speedup instead.
        if torch.version.hip is None:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            torch.manual_seed(seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" and device.type == "cuda" else None

        self.motion = self.env.command_terms["motion"]
        num_joints = self.mdp.num_joints
        self.wm = WorldModel(cfg.wm_hidden, num_joints, self.mdp.context_dim).to(device)
        goal_dim = self.mdp.goal_frame_dim * self.goal_frames
        self.pi = Policy(self.mdp.obs_dim, goal_dim, cfg.hist, cfg.z_dim, num_joints, cfg.policy_hidden).to(device)
        self.critic = Critic(self._critic_input_dim(), cfg.critic_hidden).to(device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            self.env.num_envs,
            device,
            num_joints=num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.obs_hist = torch.zeros(self.env.num_envs, cfg.hist, self.mdp.obs_dim, device=device)
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((num_joints,), 3e-3),
                torch.full((num_joints,), 5e-2),
            )
        ).to(device)
        self._action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self._action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        self._a_prev = torch.zeros((self.env.num_envs, num_joints), device=device)
        self._ep_step = torch.zeros((self.env.num_envs,), dtype=torch.int64, device=device)

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats: dict[str, list[float]] = {"r": [], "ep_len": [], "term": [], "done": []}
        compile_kernels = cfg.compile and device.type == "cuda"
        self._wm_rollout_fn = torch.compile(self._wm_rollout) if compile_kernels else self._wm_rollout
        self._imag_step_fn = torch.compile(self._imag_step) if compile_kernels else self._imag_step

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def _critic_input_dim(self) -> int:
        frames = torch.zeros((1,), dtype=torch.long, device=self.device)
        context = torch.zeros((1, self.mdp.context_dim), device=self.device)
        state = self.mdp.ref_state(frames)
        with torch.no_grad():
            features = self.mdp.critic_features(state, frames, self.mdp.normalize_context(context))
        return int(features.shape[-1] + self.mdp.goal_frame_dim * self.goal_frames)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _as_tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            # The command's buffers hold s_t and f_t as of the last evaluate
            # pass; the policy consumes the analytic observation of s_t.
            s = self._as_tensor(self.motion.state)
            frames = self._as_tensor(self.motion.steps[:, 0], torch.long)
            context = self._as_tensor(self.motion.context)
            a_prev = self._a_prev.clone()
            ep_step = self._ep_step.clone()
            state = self.mdp.unflatten_state(s)
            obs = self.mdp.add_noise(self.mdp.obs(state, a_prev, frames))
            fresh = ep_step == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted)
            goal = self.mdp.goal(frames, self.goal_frames)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = self.pi(self.obs_hist, goal)
            action = action + self.cfg.action_noise * torch.randn_like(action)
            action = torch.clamp(action, self._action_low, self._action_high)  # ±20, effectively unbounded

            result = self.env.step(action.detach().cpu().numpy().astype(np.float32))
            next_s = self._as_tensor(self.motion.state)
            done = self._as_tensor(result.terminated, torch.bool) | self._as_tensor(result.truncated, torch.bool)
            transition = {
                "s": s,
                "s1": next_s,
                "a": action,
                "a_prev": a_prev,
                "xi": context,
                "r": self._as_tensor(result.reward),
                "f": frames,
                "ep_step": ep_step.to(torch.int32),
                "done": done,
            }
            self.buffer.add(transition)
            self._a_prev = action.clone()
            self._ep_step = torch.where(done, torch.zeros_like(self._ep_step), self._ep_step + 1)

            initial = self.mdp.unflatten_state(s)
            self.wm.in_norm.update(self.wm.inputs(initial, action, self.mdp.normalize_context(context), self.mdp))
            self.wm.out_norm.update(self.wm.targets(initial, self.mdp.unflatten_state(next_s)))
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(float(result.reward.mean()) / (self.mdp.max_step_reward * self.mdp.ctrl_dt))
            self.stats["ep_len"].append(float(ep_step.float().mean()))
            self.stats["term"].append(float(np.asarray(result.terminated).sum()))
            self.stats["done"].append(float(done.sum()))

    # ------------------------------------------------------------------
    # World-model update
    # ------------------------------------------------------------------

    def _wm_rollout(self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
                self.mdp,
            )
            predictions.append(state)
        return _ANNOTATE(
            State,
            tuple(torch.stack([prediction[i] for prediction in predictions], 1) for i in range(6)),
        )

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(window["s1"])) / std) ** 2
        error = error.mean(-1)
        # Chaotic-but-real transitions (contact impacts and solver pops) are
        # unpredictable in principle: without a cap their squared error
        # dominates the window sum and the fit on the predictable mass
        # degrades over training. Rows beyond the cap are excluded from the
        # gradient instead of steering it.
        error = error.clamp(max=_WM_ERROR_CAP)
        mask = window["mask"]
        loss = (error * mask).sum() / window["s1"].shape[0]
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        last_error = error[:, -1]
        mean_last_error = (last_error.detach() * mask[:, -1]).sum() / mask[:, -1].sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    # ------------------------------------------------------------------
    # Policy / critic update in imagination
    # ------------------------------------------------------------------

    def _sample_valid_starts(self, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts: list[dict[str, torch.Tensor]] = []
        histories: list[dict[str, torch.Tensor]] = []
        count = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def _imag_step(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = self.wm.step(state, action, normalized_context, self.mdp)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_frames, self.goal_frames)
        return next_state, action, next_hist, next_frames, next_goal

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = self.mdp.unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                self.mdp.unflatten_state(history["s"]),
                history["a_prev"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        initial_goal = self.mdp.goal(frames, self.goal_frames)
        goal = initial_goal
        initial_state = state
        states, actions, previous_actions, all_frames, goals = [], [], [], [], []
        loss = torch.zeros((), device=self.device)
        grad_norm = torch.zeros((), device=self.device)
        survival = torch.zeros((), device=self.device)
        reward_sum = torch.zeros((), device=self.device)
        try:
            for _ in range(self.cfg.horizon):
                previous_actions.append(previous_action)
                state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                states.append(state)
                actions.append(previous_action)
                all_frames.append(frames)
                goals.append(goal)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = _ANNOTATE(State, tuple(flatten([item[index] for item in states]) for index in range(6)))
            flat_frames = flatten(all_frames)
            flat_actions = flatten(actions)
            flat_previous = flatten(previous_actions)
            rewards = self.mdp.reward(all_states, flat_frames, flat_actions, flat_previous)
            dones = self.mdp.soft_termination(all_states, flat_frames)
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * self.critic(
                torch.cat((self.mdp.critic_features(state, frames, normalized_context), goal), -1)
            )
            loss = -objective.mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if torch.isfinite(grad_norm):
                self.opt_pi.step()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        with torch.no_grad():
            critic_states = _ANNOTATE(
                State,
                tuple(
                    torch.cat(
                        (initial_state[index].unsqueeze(0), torch.stack([item[index] for item in states])),
                        0,
                    ).flatten(0, 1)
                    for index in range(6)
                ),
            )
            critic_frames = torch.cat((start["f"].unsqueeze(0), torch.stack(all_frames)), 0).flatten()
            critic_goals = torch.cat((initial_goal.unsqueeze(0), torch.stack(goals)), 0).flatten(0, 1)
            critic_context = normalized_context.unsqueeze(0).expand(self.cfg.horizon + 1, -1, -1).flatten(0, 1)
            critic_input = torch.cat(
                (
                    self.mdp.critic_features(critic_states, critic_frames, critic_context),
                    critic_goals,
                ),
                -1,
            ).view(self.cfg.horizon + 1, batch_size, -1)
        self.critic.norm.update(critic_input)
        with torch.no_grad():
            values = self.critic(critic_input)
            target = values[-1]
            targets = []
            for index in range(self.cfg.horizon - 1, -1, -1):
                target = detached_rewards[index] + self.cfg.gamma * (1 - detached_dones[index]) * (
                    (1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1])
        flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
        flat_targets = targets.reshape(-1)
        num_samples = flat_input.shape[0]
        minibatch_size = num_samples // self.cfg.critic_minibatches
        if minibatch_size == 0:
            raise ValueError("critic_minibatches exceeds the number of imagined transitions")
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for index in range(self.cfg.critic_minibatches):
                indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                value_loss = ((self.critic(flat_input[indices]) - flat_targets[indices]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    # ------------------------------------------------------------------
    # Schedule / checkpoints
    # ------------------------------------------------------------------

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters)))

    def state_dict(self, include_optimizers: bool = True) -> dict[str, Any]:
        checkpoint: dict[str, Any] = {
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "critic": self.critic.state_dict(),
            # Hyperparameters the evaluator needs to rebuild the policy.
            "args": {
                "goal_frames": self.goal_frames,
                "hist": self.cfg.hist,
                "z_dim": self.cfg.z_dim,
                "policy_hidden": list(self.cfg.policy_hidden),
                "amp": self.cfg.amp,
            },
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if include_optimizers:
            checkpoint.update(
                {
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def load_state_dict(self, checkpoint: dict[str, Any], *, strict: bool = True) -> None:
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "opt_wm" in checkpoint:
            self.opt_wm.load_state_dict(checkpoint["opt_wm"])
            self.opt_pi.load_state_dict(checkpoint["opt_pi"])
            self.opt_v.load_state_dict(checkpoint["opt_v"])
        elif strict:
            raise ValueError("SuperMimic resume requires a full checkpoint with optimizer states")
        self.iteration = int(checkpoint["iteration"])
        self.wall_time = float(checkpoint["wall_time"])
        self.sim_steps = int(checkpoint["sim_steps"])


__all__ = ["SuperMimicLearner"]

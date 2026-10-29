# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-agnostic APG world-model learner over a task-declared parsing head.

Ported from the upstream reference learner: a learned latent dynamics model
predicts frame-local state increments, the task's differentiable parsing head
(reward, soft termination, observations, goals, critic features) is supplied
by the task side, the goal-conditioned policy trains on imagined rollouts
with a λ-returns critic, and collection alternates with updates at a fixed
update-to-data ratio.

Environment coupling: every task fact — dimensions, the analytic heads, the
env-buffer reads of each transition, termination classification, and the
curriculum sampling state — is owned by the ``ApgTask`` head bound to the
environment (see ``contracts.py``); the learner below is generic over it.
"""

from __future__ import annotations

import math
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from motrix_env_core.torch.geometry import unflatten_state

from . import networks
from .config import ApgAgentCfg
from .contracts import State, head_transition_arrays, require_apg_env
from .imagined_env import ImaginedEnv, ImaginedTrajectory, imagine_rollout, imagine_step
from .networks import Critic, Policy, WorldModel
from .replay import ReplayBuffer
from .rollout import select_action, shift_history


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def world_model_rollout(wm, head, window: int, states, actions, contexts) -> State:
    state = unflatten_state(states[:, 0])
    predictions = []
    for index in range(window):
        state = wm.step(state, actions[:, index], head.normalize_context(contexts[:, index]))
        predictions.append(state)
    return cast(State, tuple(torch.stack([prediction[i] for prediction in predictions], 1) for i in range(6)))


def update_world_model(
    wm,
    optimizer,
    head,
    buffer,
    cfg,
    err_floor,
    rollout_fn,
    *,
    return_tensor: bool = False,
):
    """One WM optimizer batch shared by the sync learner and the WM worker."""
    window = buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
    mask = window["mask"]
    predicted = rollout_fn(window["s"], window["a"], window["xi"])
    std = torch.maximum(wm.out_norm.std(), err_floor)
    error = (wm.state_error(predicted, unflatten_state(window["s1"])) / std) ** 2
    error = error.mean(-1)
    loss = (error * mask).sum() / window["s1"].shape[0]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    optimizer.step()
    mean_last_error = (error[:, -1].detach() * mask[:, -1]).sum() / mask[:, -1].sum().clamp(min=1)
    value = loss / cfg.wm_window if return_tensor else loss.detach() / cfg.wm_window
    return value, mean_last_error


class ApgLearner:
    """Owns the networks, replay, and the collect/update training phases."""

    def __init__(
        self,
        env,
        cfg: ApgAgentCfg,
        head,
        *,
        device: torch.device,
        num_envs: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.env = require_apg_env(env)
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        self.cfg = cfg
        self.num_envs = env.num_envs if num_envs is None else num_envs
        self.device = device
        self.head = head.to(device)

        # TF32 speeds up Ampere+ NVIDIA matmuls, but the ROCm TF32 path is
        # pathologically slow for the small 3x3 bmms the parsing head issues
        # (tens of milliseconds of host-side time per call on ROCm 7.x), so
        # keep it off there. bf16 autocast (amp) covers the large-matmul
        # speedup instead.
        if torch.version.hip is None:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if seed is not None:
            torch.manual_seed(seed)

        num_joints = self.head.num_joints
        self.wm = WorldModel(cfg.wm_hidden, num_joints, self.head.context_dim).to(device)
        goal_dim = self.head.goal_dim
        self.pi = Policy(self.head.obs_dim, goal_dim, cfg.hist, cfg.z_dim, num_joints, cfg.policy_hidden).to(device)
        self.critic = Critic(self._critic_input_dim(), cfg.critic_hidden).to(device)
        amp_dtype = torch.bfloat16 if cfg.amp == "bf16" and device.type == "cuda" else None
        for net in (self.wm, self.pi, self.critic):
            net.amp_dtype = amp_dtype
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            self.num_envs,
            device,
            num_joints=num_joints,
            context_dim=self.head.context_dim,
        )
        self.obs_hist = torch.zeros(self.num_envs, cfg.hist, self.head.obs_dim, device=device)
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
        self._action_low = torch.tensor(np.asarray(self.head.action_low), dtype=torch.float32, device=device)
        self._action_high = torch.tensor(np.asarray(self.head.action_high), dtype=torch.float32, device=device)
        self._a_prev = torch.zeros((self.num_envs, num_joints), device=device)
        self._ep_step = torch.zeros((self.num_envs,), dtype=torch.int64, device=device)
        # Current env-produced observation split into the proprioceptive and
        # goal tensors; adopted through :meth:`observe` after reset and every
        # step, using the head's layout.
        self._obs: torch.Tensor | None = None
        self._goal: torch.Tensor | None = None

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats: dict[str, list[float]] = {
            "r": [],
            "ep_len_sum": [],
            "ep_ret_sum": [],
            "term": [],
            "done": [],
        }
        self._ep_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        compile_kernels = cfg.compile and device.type == "cuda"
        self._wm_rollout_fn = torch.compile(self._wm_rollout) if compile_kernels else self._wm_rollout
        self._imag_step_fn = torch.compile(imagine_step) if compile_kernels else imagine_step

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def _critic_input_dim(self) -> int:
        frames = torch.zeros((1,), dtype=torch.long, device=self.device)
        context = torch.zeros((1, self.head.context_dim), device=self.device)
        state = self.head.ref_state(frames)
        with torch.no_grad():
            features = self.head.critic_features(state, frames, self.head.normalize_context(context))
        return int(features.shape[-1] + self.head.goal_dim)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def _as_tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    def observe(self, obs_policy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Adopt the environment's current policy observation.

        Collection consumes the env-produced observation (its kernels already
        apply the task's observation noise) instead of re-deriving it
        analytically; the head splits it into the proprioceptive and goal
        tensors the policy consumes.
        """
        full = self._as_tensor(obs_policy)
        obs, goal = self.head.split_observation(full)
        goal_dim = self.head.goal_dim
        if obs.shape[-1] != self.head.obs_dim or goal.shape[-1] != goal_dim:
            raise ValueError(
                f"The head must split the policy observation into {self.head.obs_dim} proprioceptive "
                f"and {goal_dim} goal features, got {obs.shape[-1]} and {goal.shape[-1]}."
            )
        self._obs = obs
        self._goal = goal
        return self._obs, self._goal

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        if self._obs is None:
            raise RuntimeError("call observe() with the env's post-reset observation before collect()")
        for _ in range(n_steps):
            # The env's own observation drives collection (noise included);
            # the analytic torch heads serve imagination only. The head's
            # buffers hold s_t, f_t and xi_t as of the last evaluate pass.
            state, frames_raw, context = head_transition_arrays(self.head)
            s = self._as_tensor(state)
            frames = self._as_tensor(frames_raw, torch.long)
            context = self._as_tensor(context)
            a_prev = self._a_prev.clone()
            ep_step = self._ep_step.clone()
            obs, goal = self._obs, self._goal
            self.obs_hist = shift_history(self.obs_hist, obs, ep_step == 0)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = select_action(
                self.pi, self.obs_hist, goal, self._action_low, self._action_high, self.cfg.action_noise
            )

            result = self.env.step(action.detach().cpu().numpy().astype(np.float32))
            self.observe(result.obs.policy)
            next_s = self._as_tensor(np.asarray(self.head.endpoint_state()))
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

            initial = unflatten_state(s)
            self.wm.in_norm.update(self.wm.inputs(initial, action, self.head.normalize_context(context)))
            self.wm.out_norm.update(self.wm.targets(initial, unflatten_state(next_s)))
            self.sim_steps += self.num_envs
            self.stats["r"].append(float(result.reward.mean()) / (self.head.max_step_reward * self.head.ctrl_dt))
            self.stats["ep_len_sum"].append(float(((ep_step + 1) * done).sum()))
            rewards = self._as_tensor(result.reward)
            self._ep_returns += rewards
            finished = float(done.sum())
            if finished:
                self.stats["ep_ret_sum"].append(float(self._ep_returns[done].sum()))
                self._ep_returns[done] = 0.0
            # The head classifies its own failure metrics into task terminations.
            self.stats["term"].append(float(self.head.termination_counts(result.metrics)))
            self.stats["done"].append(float(done.sum()))

    def episode_metrics(self) -> dict[str, float]:
        """Canonical live-metrics rollout means over completed episodes.

        Empty when no episode finished since the last reset of ``stats`` —
        the live panel then keeps its previous values instead of a NaN.
        """
        done = sum(self.stats["done"])
        if not done:
            return {}
        return {
            "rollout/mean_return": sum(self.stats["ep_ret_sum"]) / done,
            "rollout/mean_ep_len": sum(self.stats["ep_len_sum"]) / done,
        }

    # ------------------------------------------------------------------
    # World-model update
    # ------------------------------------------------------------------

    def _wm_rollout(self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor) -> State:
        return world_model_rollout(self.wm, self.head, self.cfg.wm_window, states, actions, contexts)

    def wm_update(self, *, return_tensor: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit one WM batch, optionally retaining the final loss tensor."""
        loss, error = update_world_model(
            self.wm,
            self.opt_wm,
            self.head,
            self.buffer,
            self.cfg,
            self.err_floor,
            self._wm_rollout_fn,
            return_tensor=return_tensor,
        )
        if return_tensor:
            return loss, error
        return loss.detach(), error.detach()

    # ------------------------------------------------------------------
    # Policy / critic update in imagination
    # ------------------------------------------------------------------

    def _sample_valid_starts(self, batch_size: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        starts: list[dict[str, torch.Tensor]] = []
        histories: list[dict[str, torch.Tensor]] = []
        count = 0
        empty_rounds = 0
        while count < batch_size:
            start, history = self.buffer.sample_starts(2 * batch_size, self.cfg.hist)
            valid = self.head.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[: batch_size - count]
            if len(indices) == 0:
                empty_rounds += 1
                if empty_rounds >= 8:
                    raise RuntimeError(
                        f"The replay buffer holds no valid imagination start (buffer size {self.buffer.size}, "
                        f"horizon {self.cfg.horizon}); shorten the horizon or collect longer episodes."
                    )
            else:
                empty_rounds = 0
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)
        try:
            env = ImaginedEnv(self.wm, self.pi, self.head, step_fn=self._imag_step_fn)
            env.reset(start, history)
            traj = imagine_rollout(env, self.cfg.horizon)
            loss, grad_norm, survival, reward_sum = self._policy_objective(traj)
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)
        # The critic regression runs with its parameters trainable again, so
        # it must stay outside the freeze above.
        value_loss = self._critic_update(traj)
        imagined_reward = (reward_sum / self.cfg.horizon / (self.head.max_step_reward * self.head.ctrl_dt)).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def _policy_objective(
        self, traj: ImaginedTrajectory
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maximize the discounted continuation-weighted imagined reward."""
        batch_size = self.cfg.pi_batch
        alive = torch.cumprod(1 - traj.dones, 0)
        continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
        discount = self.cfg.gamma ** torch.arange(self.cfg.horizon, device=self.device, dtype=torch.float32).unsqueeze(
            -1
        )
        objective = (discount * continuation * traj.rewards).sum(0)
        survival = alive[-1]
        reward_sum = (continuation * traj.rewards).detach().sum(0)
        # The terminal bootstrap differentiates through the final imagined
        # state's critic input; the frozen critic parameters receive no grads.
        objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * self.critic(traj.critic_input[-1])
        loss = -objective.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(grad_norm):
            self.opt_pi.step()
        return loss, grad_norm, survival, reward_sum

    def _critic_update(self, traj: ImaginedTrajectory) -> torch.Tensor:
        """Regress the critic on λ-returns over the same cached trajectory."""
        critic_input = traj.critic_input.detach()
        detached_rewards = traj.rewards.detach()
        detached_dones = traj.dones.detach()
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
        cached_features = self.critic.norm(flat_input) if self.cfg.cache_critic_features else None
        num_samples = flat_input.shape[0]
        minibatch_size = num_samples // self.cfg.critic_minibatches
        if minibatch_size == 0:
            raise ValueError("critic_minibatches exceeds the number of imagined transitions")
        value_loss = torch.zeros((), device=self.device)
        for _ in range(self.cfg.critic_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for index in range(self.cfg.critic_minibatches):
                indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                prediction = (
                    networks.run_mlp(self.critic.net, cached_features[indices], self.critic.amp_dtype).squeeze(-1)
                    if cached_features is not None
                    else self.critic(flat_input[indices])
                )
                value_loss = ((prediction - flat_targets[indices]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        return value_loss

    # ------------------------------------------------------------------
    # Serial update (sync trainer)
    # ------------------------------------------------------------------

    def apply_lr(self, iteration: int | None = None) -> float:
        """Apply the cosine learning-rate schedule to all three optimizers."""
        lr = self.lr_at(self.iteration if iteration is None else iteration)
        for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr

    def fit_wm(self, *, defer_loss_sync: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit ``wm_batches`` world-model batches on real replay windows.

        Returns the final batch's normalized (loss, error). With
        ``defer_loss_sync`` every batch but the last keeps its loss tensor
        alive for the async trainer's late metric readback; the default
        detaches eagerly so the sync trainer can report floats right away.
        """
        wm_loss = wm_error = torch.zeros((), device=self.device)
        for batch in range(self.cfg.wm_batches):
            wm_loss, wm_error = self.wm_update(return_tensor=defer_loss_sync and batch < self.cfg.wm_batches - 1)
        return wm_loss, wm_error

    def imagine_and_update_policy(self) -> dict[str, float]:
        """Imagine with the frozen world model and update the policy and critic."""
        pi_loss, value_loss, imagined_reward, grad_norm, survival = self.policy_update()
        return {
            "policy_loss": float(pi_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "imagined_reward": float(imagined_reward.detach()),
            "policy_grad_norm": float(grad_norm.detach()),
            "imagined_survival": float(survival.detach()),
        }

    def update(self) -> dict[str, float]:
        """Run one serial training iteration: schedule, WM batches, policy.

        Thin composition of the three stages in the default ordering; the
        sync trainer drives the stages directly.
        """
        metrics = {"learning_rate": self.apply_lr()}
        wm_loss, wm_error = self.fit_wm()
        metrics.update({"wm_loss": float(wm_loss.detach()), "wm_last_error": float(wm_error.detach())})
        if self.iteration >= self.cfg.wm_only_iters:
            metrics.update(self.imagine_and_update_policy())
        self.iteration += 1
        return metrics

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
            sampling = self.head.sampling_state()
            if sampling is not None:
                checkpoint["sampling"] = sampling
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
            raise ValueError("APG resume requires a full checkpoint with optimizer states")
        self.iteration = int(checkpoint["iteration"])
        self.wall_time = float(checkpoint["wall_time"])
        self.sim_steps = int(checkpoint["sim_steps"])
        if "sampling" in checkpoint:
            self.head.load_sampling_state(checkpoint["sampling"])

    def initialize_from(self, checkpoint: dict[str, Any]) -> None:
        """Transfer network parameters and running norms into a fresh run."""
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])


__all__ = ["ApgLearner"]

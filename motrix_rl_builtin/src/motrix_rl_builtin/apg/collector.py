# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Real-environment collector and transition assembler for APG.

No recorder: the assembler consumes the derived modeling view directly
(P2.3/P2.2 artifacts). Per step it clones the declared read set before
stepping, steps the env with the actor's (optionally noisy) command, then
fills each field's successor from the post-step read views (alive rows) and
``FinalState`` (done rows — the pre-reset terminal the auto-reset would have
destroyed). Exogenous channel fields come from the resident command term
leaves (post-advance) with done rows from ``FinalState.commands``; episode
ids need no capture — the replay derives them from the done flags (P3.1
T-C). All actor inference runs under ``no_grad``; no autograd graph crosses
a real environment step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np
import torch

from motrix_env_core.manager.kernel_data import iter_leaves
from motrix_rl.console import EpisodeWindow, EpisodeWindowStats
from motrix_rl.normalize import EmpiricalNormalization
from motrix_rl.replay import SequenceReplay
from motrix_rl_builtin.apg.assemble import Assembly


def _term_leaves(term: Any, num_envs: int) -> dict[str, torch.Tensor]:
    """Dotted-path per-environment leaves of one resident command term."""
    return {
        ".".join(path): value
        for path, _annotation, value in iter_leaves(term)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == num_envs
    }


class ApgCollector:
    """Steps the environment under the actor and fills the sequence replay."""

    def __init__(
        self,
        *,
        env,
        view,
        assembly: Assembly,
        replay: SequenceReplay,
        device: torch.device,
        rng: np.random.Generator,
        actor: torch.nn.Module,
        obs_norm: EmpiricalNormalization,
        action_limit: float,
        action_bounds: tuple[float, float] | None,
        history_length: int,
        policy_group: str,
        exploration_noise_std: float = 0.0,
        exploration_clip: float = 1.0,
        on_transition: Callable[[dict, torch.Tensor, dict], None] | None = None,
    ):
        self.env = env
        self.view = view
        self.assembly = assembly
        self.replay = replay
        self.device = device
        self.rng = rng
        self.actor = actor
        self.obs_norm = obs_norm
        self.action_limit = action_limit
        self.action_bounds = action_bounds
        self.history_length = history_length
        self.policy_group = policy_group
        self.exploration_noise_std = exploration_noise_std
        self.exploration_clip = exploration_clip
        self.on_transition = on_transition
        self.channels = tuple(view.exogenous)
        self._term_name = {channel: channel.split(".", 1)[0] for channel in self.channels}
        # Rollout episode summaries for the console panel / curves.
        self.episode_window = EpisodeWindow(max_episodes=100)
        self._episode_return = np.zeros(env.num_envs, dtype=np.float64)
        self._history: torch.Tensor | None = None
        self._state_pin: torch.Tensor | None = None
        self._state_dev: torch.Tensor | None = None
        self._command_pin: torch.Tensor | None = None
        self._env_device: torch.device | None = None
        self._command_event = torch.cuda.Event() if device.type == "cuda" else None

    # -- actor ------------------------------------------------------------------

    def _push_history(self, obs: torch.Tensor) -> torch.Tensor:
        """Roll the per-lane observation history; episode starts broadcast-fill."""
        if self._history is None or self._history.shape[0] != obs.shape[0] or self._history.shape[-1] != obs.shape[-1]:
            self._history = obs.new_zeros((obs.shape[0], self.history_length, obs.shape[-1]))
        pushed = torch.cat((self._history[:, 1:], obs.unsqueeze(1)), dim=1)
        fresh = torch.as_tensor(self.env.state.episode_steps == 0, device=obs.device, dtype=torch.bool)
        filled = obs.unsqueeze(1).expand_as(pushed)
        self._history = torch.where(fresh[:, None, None], filled, pushed)
        return self._history

    def _normalize_history(self, history: torch.Tensor, *, update_from: torch.Tensor | None) -> torch.Tensor:
        """Normalize each frame of a stack, then flatten; optional stat fold."""
        batch, length, dim = history.shape
        if update_from is not None:
            self.obs_norm.update(update_from.reshape(-1, dim))
        flat = self.obs_norm(history.reshape(batch * length, dim), update=False)
        return flat.reshape(batch, length * dim)

    def _policy_command(self, *, track_norm: bool) -> torch.Tensor:
        """The actor's command for the current observation history."""
        raw_obs = getattr(self.env.state.obs, self.policy_group)
        self._env_device = raw_obs.device
        obs = raw_obs.to(self.device, non_blocking=True)
        history = self._push_history(obs)
        features = self._normalize_history(history, update_from=obs if track_norm else None)
        with torch.no_grad():
            command = self.action_limit * self.actor(features)
            if self._command_pin is None or self._command_pin.shape != command.shape:
                self._command_pin = torch.empty(
                    command.shape, dtype=torch.float32, pin_memory=self.device.type == "cuda"
                )
            self._command_pin.copy_(command, non_blocking=True)
            if self._command_event is not None:
                self._command_event.record()
        if self._command_event is not None:
            self._command_event.synchronize()
        return self._command_pin.to(self.device)

    # -- assembly -----------------------------------------------------------------

    def _read_fields(self) -> dict[str, torch.Tensor]:
        """Clone the declared read set (pre-step state on the caller's clock)."""
        return {key: self.env.sim_data.view(key).clone() for key in self.env.sim_data.keys}

    def _read_channels(self) -> dict[str, torch.Tensor]:
        """Current exogenous channel values from the resident command terms."""
        values = {}
        for channel in self.channels:
            term = self.env.command_terms[self._term_name[channel]]
            path = channel.split(".", 1)[1]
            values[channel] = _term_leaves(term, self.env.num_envs)[path].to(torch.float32)
        return values

    def _assemble(
        self,
        state_fields: dict[str, torch.Tensor],
        channels: dict[str, torch.Tensor],
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> None:
        """Fill the replay with one full-batch transition assembled post-step."""
        env = self.env
        state = env.state
        final = state.final_state
        done = state.terminated | state.truncated
        row_of = None
        if final is not None:
            row_of = torch.zeros(env.num_envs, dtype=torch.long)
            row_of[final.env_ids] = torch.arange(final.env_ids.shape[0])
        successor: dict[str, torch.Tensor] = {}
        for key in env.sim_data.keys:
            value = env.sim_data.view(key).clone()
            if row_of is not None and bool(done.any()):
                value[done] = final.sim[key][row_of[done]]
            successor[key] = value
        channel_successor: dict[str, torch.Tensor] = {}
        for channel in self.channels:
            term = env.command_terms[self._term_name[channel]]
            path = channel.split(".", 1)[1]
            value = _term_leaves(term, env.num_envs)[path].to(torch.float32)
            if row_of is not None and bool(done.any()):
                value[done] = final.commands[channel][row_of[done]].to(torch.float32)
            channel_successor[channel] = value
        batch: dict[str, torch.Tensor] = {}
        for key, value in state_fields.items():
            batch[key] = value.to(self.device, non_blocking=True)
            batch[f"next_{key}"] = successor[key].to(self.device, non_blocking=True)
        for channel, value in channels.items():
            batch[channel] = value.to(self.device, non_blocking=True)
            batch[f"next_{channel}"] = channel_successor[channel].to(self.device, non_blocking=True)
        batch["actions"] = action.to(self.device, non_blocking=True)
        batch["rewards"] = state.reward.to(self.device, non_blocking=True)
        batch["terminated"] = state.terminated.to(self.device, torch.int64, non_blocking=True)
        batch["truncated"] = state.truncated.to(self.device, torch.int64, non_blocking=True)
        batch["obs"] = obs
        self.replay.extend(batch)
        if self.on_transition is not None:
            self.on_transition(
                {key: batch[key] for key in state_fields},
                batch["actions"],
                {key: batch[f"next_{key}"] for key in state_fields},
            )

    def _track_episodes(self) -> None:
        """Fold per-lane episode returns into the window on done rows."""
        state = self.env.state
        self._episode_return += state.reward.double().cpu().numpy()
        done = (state.terminated | state.truncated).cpu().numpy()
        if done.any():
            lengths = (state.episode_steps[torch.as_tensor(done)] + 1).double().numpy()
            self.episode_window.add(self._episode_return[done].tolist(), lengths.tolist())
            self._episode_return[done] = 0.0

    # -- loops ----------------------------------------------------------------------

    def collect(self, num_steps: int) -> dict[str, float]:
        """Collect ``num_steps`` policy transitions into the replay."""
        rewards: list[float] = []
        terminations = 0
        noise_std = self.exploration_noise_std * self.action_limit
        for _ in range(num_steps):
            obs = getattr(self.env.state.obs, self.policy_group).clone()
            command = self._policy_command(track_norm=True)
            if noise_std > 0.0:
                noise = torch.as_tensor(
                    self.rng.normal(0.0, noise_std, size=tuple(command.shape)),
                    dtype=command.dtype,
                    device=command.device,
                )
                command = command + noise
                limit = self.exploration_clip * self.action_limit
                low, high = self.action_bounds if self.action_bounds is not None else (-limit, limit)
                command = torch.clamp(command, float(low), float(high))
            state_fields = self._read_fields()
            channels = self._read_channels()
            self.env.step(command.to(self._env_device))
            self._assemble(state_fields, channels, command, obs)
            self._track_episodes()
            state = self.env.state
            rewards.append(float(state.reward.mean()))
            terminations += int(state.terminated.sum())
        lane_steps = num_steps * self.env.num_envs
        return {
            "apg.rollout_reward": float(np.mean(rewards)) if rewards else 0.0,
            "apg.rollout_terminated_rate": float(terminations) / max(lane_steps, 1),
        }

    def evaluate(self, num_steps: int) -> dict[str, float]:
        """Deterministic real rollout; measures survival without touching the replay."""
        episode_ends: list[int] = []
        terminations = 0
        truncations = 0
        self._history = None
        for _ in range(num_steps):
            command = self._policy_command(track_norm=False)
            self.env.step(command.to(self._env_device))
            state = self.env.state
            done = state.terminated | state.truncated
            terminations += int(state.terminated.sum())
            truncations += int(state.truncated.sum())
            if bool(done.any()):
                episode_ends.extend((state.episode_steps[done] + 1).tolist())
        mean_length = float(np.mean(episode_ends)) if episode_ends else float(num_steps)
        lane_steps = num_steps * self.env.num_envs
        return {
            "apg.eval_mean_episode_length": mean_length,
            "apg.eval_completed_episodes": float(len(episode_ends)),
            "apg.eval_termination_rate": float(terminations) / max(lane_steps, 1),
            "apg.eval_truncation_rate": float(truncations) / max(lane_steps, 1),
        }

    def episode_summary(self) -> EpisodeWindowStats:
        """Windowed episode statistics for the panel and curves."""
        return self.episode_window.snapshot()

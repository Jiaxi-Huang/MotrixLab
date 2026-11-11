# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Motion start-frame samplers — pluggable start-step curricula, torch-native.

A start sampler owns one strategy for choosing the reference frame an episode
(or a clip wrap) starts from. State is plain torch tensors, so a sampler can
nest inside a manager command term as kernel data: the command's hooks call
the protocol methods below inside the compiled kernels, and curriculum gauges
publish through ``metric()`` fields. Implementations are pure math over
pre-drawn uniforms — the owning command performs every RNG draw with a fixed
per-mode count, keeping the per-lane stream layout stable across modes.

Bins are per clip: each clip of the concatenated corpus is binned on its own
frame span (one bin per ``env_fps`` frames, same ``len // fps + 1`` count the
global-axis sampler used), so failure or return statistics never straddle a
clip boundary. A single-clip corpus reduces exactly to the former global-axis
math, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.manager import ManagerContext, metric
from motrix_envs.motion.tracked import WbtMotionClip


@dataclass(frozen=True, slots=True)
class MotionStartSampler:
    """Pluggable start-frame curriculum consumed by a motion command term.

    Hook protocol (all methods whole-batch tensor math, traced in-kernel):

    - ``draw_count``: uniforms consumed per lane per sampling call — a fixed
      constant per mode so per-lane RNG streams stay comparable across
      configurations (one unit draw, optionally one within-bin draw, then one
      start-at-zero draw).
    - ``fold_step(ctx, steps)``: per-transition fold, called from the
      command's ``advance`` (before frame advancement; ``ctx.reward``
      already carries this step's reward total, ``steps`` the frames being
      tracked).
    - ``fold_episode_stats(idx, reset_episode, terminated, steps)``: fold the
      finished reset rows' statistics, called at the head of the command's
      ``reset`` hook; ``idx`` are the pass's compact rows, ``steps`` their
      dying frame indices.
    - ``rebuild()``: recompute the sampling distribution; called right after
      the episode fold, before the reset rows draw their start frames.
    - ``sample(draws, start_at_timestep_zero_prob)``: map pre-drawn uniforms
      ``(K, draw_count)`` in ``[-1, 1)`` to start frame indices ``(K,)``.

    The base implementations are no-ops: a mode without curriculum state
    simply leaves them untouched.
    """

    # -- Protocol hooks ---------------------------------------------------------

    @property
    def draw_count(self) -> int:
        raise NotImplementedError

    def fold_step(self, ctx: ManagerContext, steps: torch.Tensor) -> None:
        """Per-transition fold (advance hook)."""

    def fold_episode_stats(
        self,
        idx: torch.Tensor,
        reset_episode: torch.Tensor,
        terminated: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Fold finished reset rows into the curriculum statistics (reset hook)."""

    def rebuild(self) -> None:
        """Recompute the sampling distribution from the statistics (reset hook)."""

    def sample(self, draws: torch.Tensor, start_at_timestep_zero_prob: float) -> torch.Tensor:
        """Map pre-drawn uniforms to start frame indices."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class UniformTimestepsSampler(MotionStartSampler):
    """Stateless uniform start sampling over the concatenated frame axis.

    Equivalent to the former ``sampling_cdf.numel() == 0`` path: the unit
    draw maps linearly onto ``[0, num_frames - 1)`` of the whole corpus, so
    longer clips receive proportionally more starts.
    """

    num_frames: int

    @property
    def draw_count(self) -> int:
        return 2

    def sample(self, draws: torch.Tensor, start_at_timestep_zero_prob: float) -> torch.Tensor:
        unit = (draws[:, :1] + 1.0) * 0.5
        step_value = unit[:, 0] * (self.num_frames - 1)
        step_value = torch.clamp(step_value, min=0.0, max=float(self.num_frames - 2))
        steps = step_value.to(torch.int64)
        zero_unit = (draws[:, -1:] + 1.0) * 0.5
        return torch.where(zero_unit[:, 0] < start_at_timestep_zero_prob, torch.zeros_like(steps), steps)


def _clip_geometry(clip: WbtMotionClip) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip ``(start, end)`` global frame bounds from the per-frame boundaries."""
    ends = np.unique(clip.frame_clip_end.numpy())
    starts = np.concatenate(([0], ends[:-1] + 1)).astype(np.int64)
    return starts, ends.astype(np.int64)


@dataclass(frozen=True, slots=True)
class AdaptiveTimestepsSampler(MotionStartSampler):
    """Failure-biased start sampling with per-clip bins.

    Each clip is divided into ``length // env_fps + 1`` bins (~one second of
    motion each). Episodes that terminate early scatter their failure mass
    into the bin they were playing (episode fold at reset); the histogram is
    EMA-smoothed once per transition (``fold_step``), an exponential spatial
    kernel smears it across neighbouring bins, and a uniform floor keeps
    every bin reachable (``rebuild``). Sampling picks a bin through the CDF
    (search-left, equality selects the equal bin) and a phase within it.

    On a single-clip corpus this reproduces the former global-axis sampler
    bit for bit: same bin count, same within-bin phase math, same RNG draws.
    """

    # Per-clip geometry on the concatenated frame axis, ``C`` clips.
    clip_frame_start: torch.Tensor  # (C,) int64
    clip_frame_end: torch.Tensor  # (C,) int64
    clip_bins: torch.Tensor  # (C,) int64 bins per clip
    bin_clip_offset: torch.Tensor  # (C,) int64 first flat bin of each clip

    # Curriculum statistics over the flat concatenated bin axis, ``B`` bins.
    failed_count: torch.Tensor  # (B,) float32 EMA failure mass
    current_failed: torch.Tensor  # (B,) float32 per-step accumulator
    sampling_cdf: torch.Tensor  # (B,) float32

    uniform_ratio: float
    alpha: float
    kernel_size: int
    kernel_lambda: float

    # Batch-level gauges written by the rebuild pass.
    entropy: torch.Tensor = metric(name="adaptive_sampling_entropy", scalar=True)
    top1_prob: torch.Tensor = metric(name="adaptive_sampling_top1_prob", scalar=True)
    top1_bin: torch.Tensor = metric(name="adaptive_sampling_top1_bin", scalar=True)
    failure_mass: torch.Tensor = metric(name="adaptive_sampling_failure_mass", scalar=True)

    @property
    def draw_count(self) -> int:
        return 3

    @staticmethod
    def from_clip(
        clip: WbtMotionClip,
        env_fps: int,
        *,
        uniform_ratio: float,
        alpha: float,
        kernel_size: int,
        kernel_lambda: float,
    ) -> AdaptiveTimestepsSampler:
        """Build the sampler over a clip corpus, statistics zero-initialized."""
        starts, ends = _clip_geometry(clip)
        lengths = ends - starts + 1
        bins = lengths // max(int(env_fps), 1) + 1
        offsets = np.concatenate(([0], np.cumsum(bins)[:-1]))
        num_bins = int(bins.sum())
        return AdaptiveTimestepsSampler(
            clip_frame_start=torch.from_numpy(starts),
            clip_frame_end=torch.from_numpy(ends),
            clip_bins=torch.from_numpy(bins),
            bin_clip_offset=torch.from_numpy(offsets),
            failed_count=torch.zeros((num_bins,), dtype=torch.float32),
            current_failed=torch.zeros((num_bins,), dtype=torch.float32),
            sampling_cdf=torch.cumsum(torch.full((num_bins,), 1.0 / num_bins), dim=0),
            uniform_ratio=float(uniform_ratio),
            alpha=float(alpha),
            kernel_size=int(kernel_size),
            kernel_lambda=float(kernel_lambda),
            entropy=torch.zeros((), dtype=torch.float32),
            top1_prob=torch.zeros((), dtype=torch.float32),
            top1_bin=torch.zeros((), dtype=torch.float32),
            failure_mass=torch.zeros((), dtype=torch.float32),
        )

    # -- Sampling ------------------------------------------------------------

    def sample(self, draws: torch.Tensor, start_at_timestep_zero_prob: float) -> torch.Tensor:
        unit = (draws[:, :1] + 1.0) * 0.5
        bin_id = (self.sampling_cdf[None, :] < unit).sum(dim=1, dtype=torch.int64)
        bin_unit = (draws[:, 1:2] + 1.0) * 0.5
        clip = torch.searchsorted(self.bin_clip_offset, bin_id, right=True) - 1
        local_bin = bin_id - self.bin_clip_offset[clip]
        clip_start = self.clip_frame_start[clip]
        clip_length = self.clip_frame_end[clip] - clip_start + 1
        # Within-clip phase, the former global-axis math scoped to the clip:
        # (local_bin + u) / clip_bins * clip_length, clamped to a non-terminal
        # frame of the clip.
        step_value = (local_bin.to(torch.float32) + bin_unit[:, 0]) / self.clip_bins[clip]
        step_value = step_value * clip_length.to(torch.float32)
        step_value = torch.minimum(torch.clamp(step_value, min=0.0), (clip_length - 2).to(torch.float32))
        steps = clip_start + step_value.to(torch.int64)
        zero_unit = (draws[:, -1:] + 1.0) * 0.5
        return torch.where(zero_unit[:, 0] < start_at_timestep_zero_prob, torch.zeros_like(steps), steps)

    # -- Curriculum folds -------------------------------------------------------

    def fold_step(self, ctx: ManagerContext, steps: torch.Tensor) -> None:
        """EMA-fold the step failure histogram (once per transition)."""
        self.failed_count.mul_(1.0 - self.alpha).add_(self.current_failed, alpha=self.alpha)
        self.current_failed.zero_()

    def fold_episode_stats(
        self,
        idx: torch.Tensor,
        reset_episode: torch.Tensor,
        terminated: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Scatter this pass's episode failures into their per-clip bins."""
        episode_failed = reset_episode & terminated[idx]
        clip = torch.searchsorted(self.clip_frame_end, steps, right=False)
        clip_length = self.clip_frame_end[clip] - self.clip_frame_start[clip] + 1
        local_bin = (steps - self.clip_frame_start[clip]) * self.clip_bins[clip] // clip_length
        failed_bins = self.bin_clip_offset[clip] + local_bin
        failed_bins = torch.clamp(failed_bins, 0, self.failed_count.numel() - 1)
        histogram = torch.zeros_like(self.failed_count)
        histogram.scatter_add_(0, failed_bins, episode_failed.to(torch.float32))
        self.current_failed.add_(histogram)

    def rebuild(self) -> None:
        """Rebuild the sampling CDF from the smoothed failure statistics."""
        num_bins = self.failed_count.numel()
        probabilities = self.failed_count + self.uniform_ratio / num_bins
        kernel_size = max(self.kernel_size, 1)
        if kernel_size > 1:
            kernel = torch.tensor([self.kernel_lambda**i for i in range(kernel_size)], dtype=torch.float32)
            kernel /= kernel.sum()
            padded = torch.nn.functional.pad(probabilities[None, None, :], (0, kernel_size - 1), mode="rep")
            probability = (padded.unfold(2, kernel_size, 1) @ kernel)[0, 0]
        else:
            probability = probabilities
        total = probability.sum()
        uniform = torch.full((num_bins,), 1.0 / num_bins, dtype=torch.float32)
        positive = total > 0.0
        probabilities = torch.where(
            positive, probability / torch.where(positive, total, torch.ones_like(total)), uniform
        )
        self.entropy.copy_(-(probabilities * torch.log(probabilities + 1e-12)).sum() / math.log(max(num_bins, 2)))
        top_bin = torch.argmax(probabilities)
        # Gather (not ``probabilities[top_bin]``): tensor indexing would sync
        # the index to a Python int and break the graph.
        self.top1_prob.copy_(torch.gather(probabilities, 0, top_bin.reshape(1)).reshape(()))
        self.top1_bin.copy_(top_bin.to(torch.float32) / float(num_bins))
        self.failure_mass.copy_(self.failed_count.sum())
        # Per-clip phase addressing keeps every bin reachable inside its clip
        # (starts clamp to a non-terminal frame), so no tail masking applies.
        cdf = torch.cumsum(probabilities, dim=0).to(torch.float32)
        cdf[-1] = 1.0
        self.sampling_cdf.copy_(cdf)


def per_clip_bins(clip: WbtMotionClip, bin_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed one-per-``bin_frames`` start windows per clip: ``(start, span)``.

    Windows are integer global-frame ranges inside one clip; each span ends
    before its clip's terminal frame (span clamped to the clip end and to one
    frame minimum), so a drawn start never begins on a terminal frame while a
    nonterminal one exists in that clip.
    """
    ends_per_frame = clip.frame_clip_end.numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ends_per_frame) != 0) + 1)).astype(np.int64)
    ends = ends_per_frame[starts]
    bin_start = []
    bin_span = []
    for start, end in zip(starts, ends):
        for frame in range(int(start), max(int(end), int(start) + 1), int(bin_frames)):
            bin_start.append(frame)
            bin_span.append(min(max(int(end) - frame, 1), int(bin_frames)))
    return np.asarray(bin_start, dtype=np.int64), np.asarray(bin_span, dtype=np.int64)


def max_discounted_return(ctrl_dt: float, positive_reward_sum: float, sampling_gamma: float) -> float:
    """Softmax normalizer: the task's largest achievable discounted return.

    Derived from the configured reward weights (no copied value): the per-step
    maximum reward summed over the discount horizon.
    """
    return ctrl_dt * positive_reward_sum / max(1.0 - sampling_gamma, 1e-6)


@dataclass(frozen=True, slots=True)
class ReturnTimestepsSampler(MotionStartSampler):
    """Return-based start sampling: prefer starts the policy performs worst on.

    The per-bin statistic is upstream supermimic's return estimator: every
    ``advance`` rolls a per-lane window of the last ``window`` transitions
    (reward, tracked frame's bin, episode-end kind; ``fold_step``), and when
    an episode ends (``fold_episode_stats``) the window's per-position
    discounted returns — falls bootstrap 0, truncations continue from the
    value map at the successor bin — fold into a memory-capped incremental
    mean keyed by EACH position's bin. The estimator is bounded below by
    construction (short windows, value-based continuations), which the
    whole-episode accumulator it replaced was not: under a curriculum flood
    that variant assigned deeply negative values (~-3) that locked the
    0.4 low-return focus onto single bins (2026-11-11, twice-reproduced).
    The sampling distribution mixes a uniform prior, a softmax that favors
    low-value bins, and a slow history prior (``rebuild``).

    Bins are the fixed integer windows of :func:`per_clip_bins`; sampling
    addresses them by integer offsets, so corpora beyond float32 frame
    precision stay exact.
    """

    bin_start: torch.Tensor  # (B,) int64 global start frame of each bin
    bin_span: torch.Tensor  # (B,) int64 frames covered by each bin
    sampling_cdf: torch.Tensor  # (B,) float32
    bin_return: torch.Tensor  # (B,) float32 per-bin value map (memory-capped mean)
    bin_count: torch.Tensor  # (B,) float32 samples folded per bin (uncapped counter)
    bin_history: torch.Tensor  # (B,) float32 slow sampling-frequency prior (uniform)

    # Per-lane rolling estimator windows, ``N`` lanes x ``window`` steps,
    # newest last. Done kinds: 0 open, 1 fall (terminated), 2 truncated.
    lane_win_rewards: torch.Tensor  # (N, W) float32 per-step reward totals
    lane_win_bins: torch.Tensor  # (N, W) int64 bin of the tracked frame
    lane_win_done_kind: torch.Tensor  # (N, W) int64 episode-end kind per step

    num_frames: int
    uniform_mix: float
    return_mix: float
    history_mix: float
    sampling_gamma: float
    window: int
    # Per-bin sample-memory cap of the value statistic, as upstream's
    # ``memory``: a bin's value averages its most recent ~memory samples
    # (exponential forgetting at 1/memory per new sample).
    memory: int
    g_max: float

    # Batch-level gauges written by the rebuild pass.
    mean_bin_return: torch.Tensor = metric(name="return_sampling_mean_bin_return", scalar=True)
    entropy: torch.Tensor = metric(name="return_sampling_entropy", scalar=True)
    top1_prob: torch.Tensor = metric(name="return_sampling_top1_prob", scalar=True)
    top1_bin: torch.Tensor = metric(name="return_sampling_top1_bin", scalar=True)

    @property
    def draw_count(self) -> int:
        return 3

    @staticmethod
    def from_clip(
        clip: WbtMotionClip,
        env_fps: int,
        num_envs: int,
        *,
        uniform_mix: float,
        return_mix: float,
        history_mix: float,
        sampling_gamma: float,
        window: int,
        memory: int,
        g_max: float,
    ) -> ReturnTimestepsSampler:
        """Build the sampler over a clip corpus, windows zero-initialized."""
        bin_start, bin_span = per_clip_bins(clip, env_fps)
        num_bins = int(bin_start.size)
        num_frames = clip.joint_pos.shape[0]
        return ReturnTimestepsSampler(
            bin_start=torch.from_numpy(bin_start),
            bin_span=torch.from_numpy(bin_span),
            sampling_cdf=torch.cumsum(torch.full((num_bins,), 1.0 / num_bins), dim=0),
            bin_return=torch.zeros((num_bins,), dtype=torch.float32),
            bin_count=torch.zeros((num_bins,), dtype=torch.float32),
            bin_history=torch.full((num_bins,), 1.0 / max(num_bins, 1), dtype=torch.float32),
            lane_win_rewards=torch.zeros((num_envs, window), dtype=torch.float32),
            lane_win_bins=torch.zeros((num_envs, window), dtype=torch.int64),
            lane_win_done_kind=torch.zeros((num_envs, window), dtype=torch.int64),
            num_frames=num_frames,
            uniform_mix=float(uniform_mix),
            return_mix=float(return_mix),
            history_mix=float(history_mix),
            sampling_gamma=float(sampling_gamma),
            window=int(window),
            memory=int(memory),
            g_max=float(g_max),
            mean_bin_return=torch.zeros((), dtype=torch.float32),
            entropy=torch.zeros((), dtype=torch.float32),
            top1_prob=torch.zeros((), dtype=torch.float32),
            top1_bin=torch.zeros((), dtype=torch.float32),
        )

    # -- Sampling ------------------------------------------------------------

    def sample(self, draws: torch.Tensor, start_at_timestep_zero_prob: float) -> torch.Tensor:
        unit = (draws[:, :1] + 1.0) * 0.5
        bin_id = (self.sampling_cdf[None, :] < unit).sum(dim=1, dtype=torch.int64)
        bin_unit = (draws[:, 1:2] + 1.0) * 0.5
        # Integer addressing: keep the offset arithmetic in int64 so corpora
        # beyond float32 frame precision stay exact.
        offset = (bin_unit[:, 0] * self.bin_span[bin_id]).to(torch.int64)
        steps = self.bin_start[bin_id] + offset
        steps = torch.clamp(steps, min=0, max=self.num_frames - 2)
        zero_unit = (draws[:, -1:] + 1.0) * 0.5
        return torch.where(zero_unit[:, 0] < start_at_timestep_zero_prob, torch.zeros_like(steps), steps)

    # -- Curriculum folds -------------------------------------------------------

    def _bin_of(self, steps: torch.Tensor) -> torch.Tensor:
        """Bin index of each global frame (bins sorted by global start)."""
        return (torch.searchsorted(self.bin_start, steps, right=True) - 1).clamp(min=0)

    def fold_step(self, ctx: ManagerContext, steps: torch.Tensor) -> None:
        """Roll the per-lane estimator windows one transition.

        The reward, the tracked frame's bin, and a provisional open-step kind
        enter at the newest position; ``fold_episode_stats`` overwrites the
        dying step's kind when its episode ends.
        """
        rewards, bins, kinds = self.lane_win_rewards, self.lane_win_bins, self.lane_win_done_kind
        rewards[:, :-1] = rewards[:, 1:]
        rewards[:, -1] = ctx.reward
        bins[:, :-1] = bins[:, 1:]
        bins[:, -1] = self._bin_of(steps)
        kinds[:, :-1] = kinds[:, 1:]
        kinds[:, -1] = 0

    def fold_episode_stats(
        self,
        idx: torch.Tensor,
        reset_episode: torch.Tensor,
        terminated: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Fold the ended episodes' estimator windows into the value map.

        Upstream's ``_returns``/``_update`` semantics: per-position discounted
        returns over the window, chained across open steps and reset at
        episode boundaries — falls continue from 0, truncations from the
        value map at the successor bin (the next window position's bin; the
        dying lane's post-advance frame for the newest step). Every position
        contributes a sample keyed by its OWN bin (deviation: no
        per-(episode, bin) visit averaging — dwell time weights bins
        mildly), so mid-episode frames update the map, not just episode
        starts. Values are bounded below by construction: short windows and
        value-based continuations cannot manufacture the deeply negative
        bin readings that locked the sampler under curriculum floods.
        """
        # The dying step's end kind on episode rows: 1 fall, 2 truncated (time
        # limit or clip end — clip-end continuation approximates as the
        # successor bin; its incidence is ~1e-4 under the 10 s episode cap).
        # Mask-expressed over the full compact rows: the reset kernel traces
        # under Dynamo, so no data-dependent shapes.
        dying = torch.where(terminated[idx], 1, 2)
        current = self.lane_win_done_kind[idx, -1]
        self.lane_win_done_kind[idx, -1] = torch.where(reset_episode, dying, current)

        rewards = self.lane_win_rewards[idx]  # (K, W)
        bins = self.lane_win_bins[idx]
        kinds = self.lane_win_done_kind[idx]
        gamma = self.sampling_gamma
        window = rewards.shape[1]

        next_bins = torch.cat((bins[:, 1:], self._bin_of(steps).unsqueeze(1)), dim=1)
        next_value = torch.where(kinds == 1, 0.0, self.bin_return[next_bins])

        returns = torch.empty_like(rewards)
        returns[:, -1] = rewards[:, -1] + gamma * next_value[:, -1]
        for position in range(window - 2, -1, -1):
            continuation = torch.where(kinds[:, position] > 0, next_value[:, position], returns[:, position + 1])
            returns[:, position] = rewards[:, position] + gamma * continuation

        # Non-episode rows of the compact pass contribute no mass.
        mask = reset_episode.to(torch.float32)[:, None]
        by_bin = torch.zeros_like(self.bin_return)
        counts = torch.zeros_like(self.bin_return)
        by_bin.index_add_(0, bins.reshape(-1), (returns * mask).reshape(-1))
        counts.index_add_(0, bins.reshape(-1), mask.expand(-1, window).reshape(-1))
        touched = counts > 0
        # Memory-capped incremental mean (upstream ``_update``).
        capped = self.bin_count.clamp(max=self.memory) + counts
        updated = self.bin_return + (by_bin - counts * self.bin_return) / capped.clamp(min=1.0)
        self.bin_return.copy_(torch.where(touched, updated, self.bin_return))
        self.bin_count.copy_(capped)

    def rebuild(self) -> None:
        """Rebuild the sampling CDF from the per-bin return statistics."""
        num_bins = self.bin_return.numel()
        # Return-weighted preference: low-return bins get more mass.
        scaled = -25.0 * self.bin_return / max(self.g_max, 1e-9)
        soft = torch.exp(scaled - scaled.max())
        soft = soft / soft.sum()
        history = self.bin_history / self.bin_history.sum()
        probability = self.uniform_mix / num_bins + self.return_mix * soft + self.history_mix * history
        probability = probability / probability.sum()
        self.mean_bin_return.copy_(self.bin_return.mean())
        self.entropy.copy_(-(probability * torch.log(probability + 1e-12)).sum() / math.log(max(num_bins, 2)))
        top_bin = torch.argmax(probability)
        self.top1_prob.copy_(torch.gather(probability, 0, top_bin.reshape(1)).reshape(()))
        self.top1_bin.copy_(top_bin.to(torch.float32) / float(num_bins))
        cdf = torch.cumsum(probability, dim=0).to(torch.float32)
        cdf[-1] = 1.0
        self.sampling_cdf.copy_(cdf)


__all__ = [
    "AdaptiveTimestepsSampler",
    "MotionStartSampler",
    "ReturnTimestepsSampler",
    "UniformTimestepsSampler",
    "max_discounted_return",
    "per_clip_bins",
]

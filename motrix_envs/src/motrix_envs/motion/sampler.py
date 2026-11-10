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
    - ``fold_step(ctx)``: per-transition fold, called from the command's
      ``advance`` (before frame advancement; ``ctx.reward`` already carries
      this step's reward total).
    - ``fold_episode_stats(idx, reset_episode, terminated, steps)``: fold the
      finished reset rows' statistics, called at the head of the command's
      ``reset`` hook; ``idx`` are the pass's compact rows, ``steps`` their
      dying frame indices.
    - ``rebuild()``: recompute the sampling distribution; called right after
      the episode fold, before the reset rows draw their start frames.
    - ``sample(draws, start_at_timestep_zero_prob)``: map pre-drawn uniforms
      ``(K, draw_count)`` in ``[-1, 1)`` to start frame indices ``(K,)``.
    - ``mark_episode_starts(idx, reset_episode, steps)``: open new episodes
      on the episode-reset rows once their start frames are in place.

    The base implementations are no-ops: a mode without curriculum state
    simply leaves them untouched.
    """

    # -- Protocol hooks ---------------------------------------------------------

    @property
    def draw_count(self) -> int:
        raise NotImplementedError

    def fold_step(self, ctx: ManagerContext) -> None:
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

    def mark_episode_starts(
        self,
        idx: torch.Tensor,
        reset_episode: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Open new episodes on episode-reset rows (after start frames are set)."""

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

    def fold_step(self, ctx: ManagerContext) -> None:
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


__all__ = [
    "AdaptiveTimestepsSampler",
    "MotionStartSampler",
    "UniformTimestepsSampler",
]

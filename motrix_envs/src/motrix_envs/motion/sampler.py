# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Adaptive timestep sampler — start-frame curriculum over a motion timeline.

Buckets a motion clip into bins and biases episode start frames toward bins with
more recent failures. Pure numpy; independent of any robot or simulation model.
"""

from __future__ import annotations

import numpy as np


class AdaptiveTimestepsSampler:
    """Curriculum sampler that biases motion-clip start frames toward harder regions.

    The motion clip is divided into ``num_bins`` equal-width bins. Each env
    step records which bins caused early termination (``record_failures``);
    once per step the failure histogram is folded into a running EMA
    (``update``). ``sample`` derives per-bin probabilities from that smoothed
    histogram (with an exponential spatial kernel and a uniform floor), then
    draws start-frame indices.

    Typical per-step usage (``motion_steps`` tracks each env's current
    frame index into the clip; ``terminated`` marks failed episodes)::

        failed = motion_steps[env_ids][terminated[env_ids]]
        sampler.record_failures(failed)
        sampler.update()
        start_steps = sampler.sample(num_resets)
    """

    def __init__(
        self,
        motion_time_step_total: int,
        env_fps: int,
        kernel_size: int,
        kernel_lambda: float,
        uniform_ratio: float,
        alpha: float,
    ):
        """Configure the sampler.

        Args:
            motion_time_step_total: Length of the motion clip in sim steps.
            env_fps: Control frequency (Hz); the bin width is ~1 second of
                motion, i.e. ``num_bins = motion_time_step_total // env_fps + 1``.
            kernel_size: Width of the exponential smoothing kernel applied to
                the failure histogram when computing probabilities. ``1``
                disables spatial smoothing.
            kernel_lambda: Base of the exponential kernel
                (``kernel[i] = kernel_lambda**i``, then L1-normalized).
                Smaller → sharper peak around failed bins; ``1.0`` → flat
                averaging over the window.
            uniform_ratio: Exploration floor. Each bin receives an additive
                ``uniform_ratio / num_bins`` mass before normalization, so the
                policy never fully abandons any region of the clip. Pass ``0``
                for pure failure-driven sampling (risks starvation).
            alpha: EMA weight in ``update`` — fraction of the new step's
                failure count blended into the running histogram. ``1.0`` uses
                only the most recent step; small values give a longer memory.
        """
        # Clip length in sim steps.
        self.motion_time_step_total = int(motion_time_step_total)
        # Histogram buckets — roughly one second of motion each.
        self.num_bins = self.motion_time_step_total // max(env_fps, 1) + 1
        # Exploration floor (fraction of total probability mass).
        self.uniform_ratio = uniform_ratio
        # EMA weight on the current step's failures (see ``update``).
        self.alpha = alpha
        # Smoothed failure histogram — what sampling probabilities are built from.
        self.bin_failed_count = np.zeros((self.num_bins,), dtype=np.float32)
        # Per-step failure accumulator; zeroed by ``update`` after each commit.
        self.current_bin_failed_count = np.zeros((self.num_bins,), dtype=np.float32)

        # L1-normalized exponential kernel used to smear failure mass across
        # neighboring bins in ``sampling_probabilities``.
        kernel_size = max(kernel_size, 1)
        kernel = np.asarray([kernel_lambda**i for i in range(kernel_size)], dtype=np.float32)
        self.kernel = kernel / np.sum(kernel)

    def record_failures(self, failed_steps: np.ndarray) -> None:
        """Accumulate this step's failures into the per-step histogram.

        Maps each failed motion-step index to its bin and increments
        ``current_bin_failed_count`` in place. Call once per env step, before
        ``update``; empty input is a no-op.

        Args:
            failed_steps: 1D array of motion-step indices where episodes
                terminated this step (typically
                ``motion_steps[env_ids][terminated[env_ids]]``).
        """
        if failed_steps.size == 0:
            return
        failed_bins = (failed_steps.astype(np.int64) * self.num_bins) // max(self.motion_time_step_total, 1)
        failed_bins = np.clip(failed_bins, 0, self.num_bins - 1)
        self.current_bin_failed_count += np.bincount(failed_bins, minlength=self.num_bins).astype(np.float32)

    def update(self) -> None:
        """Commit the per-step histogram into the smoothed one via EMA.

        ``bin_failed_count = alpha * current + (1 - alpha) * bin_failed_count``,
        then ``current_bin_failed_count`` is zeroed for the next cycle. Call
        once per env step after ``record_failures`` and before ``sample``.
        """
        self.bin_failed_count = (self.alpha * self.current_bin_failed_count) + (
            (1.0 - self.alpha) * self.bin_failed_count
        )
        self.current_bin_failed_count.fill(0.0)

    @property
    def sampling_probabilities(self) -> np.ndarray:
        """Per-bin sampling probabilities, sums to 1.

        Pipeline: smoothed histogram → add uniform floor → convolve with
        exponential kernel → renormalize. Falls back to a uniform
        distribution when total mass is zero (e.g. before any failures have
        been recorded).

        Returns:
            ``float32`` array of shape ``(num_bins,)``.
        """
        prob = self.bin_failed_count + self.uniform_ratio / self.num_bins
        if self.kernel.shape[0] > 1:
            padded = np.pad(prob, (0, self.kernel.shape[0] - 1), mode="edge")
            prob = np.asarray(
                [np.sum(padded[i : i + self.kernel.shape[0]] * self.kernel) for i in range(self.num_bins)],
                dtype=np.float32,
            )
        prob_sum = np.sum(prob)
        if prob_sum <= 0.0:
            return np.full((self.num_bins,), 1.0 / self.num_bins, dtype=np.float32)
        return (prob / prob_sum).astype(np.float32)

    def sample(self, num_samples: int, max_step_exclusive: int | None = None) -> np.ndarray:
        """Draw motion-step indices to start episodes from.

        A bin is chosen from ``sampling_probabilities`` (optionally truncated
        to ``[0, max_step_exclusive)``), then a uniformly random phase inside
        that bin is converted to a motion-step index.

        Args:
            num_samples: Number of start frames to draw.
            max_step_exclusive: If given, restrict sampling to motion steps
                in ``[0, max_step_exclusive)``. Bins fully beyond the cutoff
                are zeroed out and probabilities renormalized. Clamped to a
                minimum of 2 to leave at least one drawable step.

        Returns:
            ``int64`` array of shape ``(num_samples,)`` in
            ``[0, motion_time_step_total - 1]``.
        """
        probabilities = self.sampling_probabilities
        max_step = self.motion_time_step_total - 1
        if max_step_exclusive is not None:
            max_step = int(np.clip(max_step_exclusive, 2, self.motion_time_step_total - 1))
            max_bin = int(np.ceil(max_step * self.num_bins / max(self.motion_time_step_total, 1)))
            max_bin = int(np.clip(max_bin, 1, self.num_bins))
            probabilities = probabilities.copy()
            probabilities[max_bin:] = 0.0
            prob_sum = np.sum(probabilities)
            if prob_sum <= 0.0:
                probabilities[:max_bin] = 1.0 / max_bin
            else:
                probabilities /= prob_sum
        sampled_bins = np.random.choice(self.num_bins, size=num_samples, replace=True, p=probabilities)
        phase = (
            sampled_bins.astype(np.float32) + np.random.uniform(size=num_samples).astype(np.float32)
        ) / self.num_bins
        steps = (phase * self.motion_time_step_total).astype(np.int64)
        return np.clip(steps, 0, max_step - 1)

    def stats(self) -> dict[str, float]:
        """Summary statistics for logging/monitoring.

        Returns:
            Dict with four float fields:

            - ``adaptive_sampling_entropy``: Shannon entropy of the sampling
              distribution normalized to ``[0, 1]`` (1 = uniform, 0 = all mass
              on one bin).
            - ``adaptive_sampling_top1_prob``: probability mass on the hottest
              bin.
            - ``adaptive_sampling_top1_bin``: normalized location of the
              hottest bin in ``[0, 1]`` — where in the clip failures
              concentrate.
            - ``adaptive_failure_mass``: total smoothed failure count
              (``sum(bin_failed_count)``); a rough gauge of how much is
              failing.
        """
        probabilities = self.sampling_probabilities
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        entropy_norm = entropy / float(np.log(max(self.num_bins, 2)))
        top_bin = int(np.argmax(probabilities))
        return {
            "adaptive_sampling_entropy": entropy_norm,
            "adaptive_sampling_top1_prob": float(probabilities[top_bin]),
            "adaptive_sampling_top1_bin": float(top_bin) / float(self.num_bins),
            "adaptive_failure_mass": float(np.sum(self.bin_failed_count)),
        }

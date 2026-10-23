# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Success Guided Sampling — arXiv 2610.12465's success-rate curriculum.

This module ports the paper's SGS sampler onto SONIC's per-source bin layout.
The paper samples *task configurations*; here a configuration is one
source-clip-local bin (the same ``bin_size``-frame binning the adaptive
sampler uses), and one draw picks a bin plus a uniform frame inside it.

Two adaptations, both forced by the tracking task:

- Outcomes replace the paper's boolean task success: an episode that reaches
  its time limit without a failure termination is a success, a terminated one
  is not. Outcomes credit the bin where the episode ended — the same
  attribution the failure-rate sampler ships — so mid-episode clip-end
  re-anchoring never creates extra outcomes and a draining lane keeps
  reporting to its own source clip across rotation swaps.
- Statistics keep the paper's last-``H``-outcomes window (a sliding mean that
  keeps adapting as the policy learns), stored as one int8 ring buffer per
  bin. Bins without recorded outcomes carry the uninformative 0.5 rate, at
  which the Beta kernel weight is flat — sampling starts uniform, exactly
  like the paper's pre-training phase.

Unlike the failure-rate curriculum, no uniform floor or per-clip mass
equalization is applied: the paper's ``epsilon`` already keeps every bin's
probability non-zero, and each bin is one configuration sampled i.i.d.
"""

from __future__ import annotations

import numpy as np

from motrix_envs.motion.adaptive import clip_segment_bounds

__all__ = [
    "SuccessGuidedSampler",
    "build_sgs_probabilities",
]

# Uninformative success-rate prior for bins without recorded outcomes.
_UNSEEN_SUCCESS_RATE = 0.5


def build_sgs_probabilities(
    success_rates: np.ndarray,
    target: float,
    concentration: float,
    epsilon: float,
    temperature: float,
) -> np.ndarray:
    """Beta-shaped kernel over success rates, sampled through a softmax.

    Port of the paper's Eqs. 1-2: ``w_i = (p_i + eps)^(kappa*t) *
    (1 - p_i + eps)^(kappa*(1-t))`` and ``p(i) = softmax(log w_i / T)`` with
    mode ``t`` and concentration ``kappa``. ``epsilon`` prevents singularities
    at ``p in {0, 1}`` and keeps every configuration's probability non-zero,
    so success-rate estimates keep updating even for bins the policy cannot
    yet solve.
    """
    rates = np.clip(np.asarray(success_rates, np.float64), 0.0, 1.0)
    log_weights = concentration * (target * np.log(rates + epsilon) + (1.0 - target) * np.log(1.0 - rates + epsilon))
    logits = (log_weights + epsilon) / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    return (weights / np.sum(weights)).astype(np.float32)


class SuccessGuidedSampler:
    """Host-side SGS sampler shared by resident and rotated SONIC corpora.

    Same command-term contract as
    :class:`~motrix_envs.motion.adaptive.AdaptiveMotionSampler`: publish the
    per-bin CDF plus active bin frame ranges for the reset kernel, drain the
    per-lane outcome handoff each tick, and keep statistics per source clip so
    rotation windows and draining lanes attribute outcomes correctly. The
    probability model is the paper's Beta-kernel softmax over each bin's
    last-``H``-outcomes success rate instead of the failure-rate mixture.
    """

    def __init__(
        self,
        *,
        bin_size: int,
        history: int = 100,
        target: float = 0.66,
        concentration: float = 5.0,
        epsilon: float = 1e-8,
        temperature: float = 2.0,
        update_period: int = 128,
    ):
        self._bin_size = int(bin_size)
        self._history = int(history)
        self._target = float(target)
        self._concentration = float(concentration)
        self._epsilon = float(epsilon)
        self._temperature = float(temperature)
        self._update_period = int(update_period)
        # Flat per-bin statistics: an int8 last-outcomes ring buffer plus its
        # running success sum (the sliding mean never re-reduces the ring),
        # and dense per-source lookup tables over the same flat layout as the
        # adaptive sampler.
        self._outcomes = np.empty((0, self._history), np.int8)
        self._success_sums = np.empty(0, np.float64)
        self._counts = np.empty(0, np.int64)
        self._cursor = np.empty(0, np.int64)
        self._offset_lut = np.empty(0, np.int64)
        self._count_lut = np.empty(0, np.int64)
        self._frames_lut = np.empty(0, np.int64)
        self._steps_since_update = 0
        self._dirty = False
        self._projected: tuple[int, int] | None = None
        self._probabilities = np.empty(0, np.float32)
        self._weights_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def attach(self, term) -> None:
        """Project the term's current window and publish the initial tables."""
        self._project(term)

    def tick(self, term) -> dict[str, float]:
        """Republish on window changes, drain outcomes, maybe republish stats.

        Called once per control step with the canonical motion command; the
        returned metrics ride the step result into the trainer's curves.
        """
        window = (int(term.window_range[0]), int(term.window_range[1]))
        if window != self._projected:
            self._project(term)
        self._drain_outcomes(term)
        if self._dirty:
            self._steps_since_update += 1
            if self._steps_since_update >= self._update_period:
                self._project(term)
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        """Sampling metrics for the trainer curves, cached at each republish."""
        num_bins = self._probabilities.size
        if num_bins == 0:
            return {
                "sgs_sampling_entropy": 1.0,
                "sgs_sampling_top1_prob": 0.0,
                "sgs_mean_success_rate": 0.0,
            }
        entropy = -float(np.sum(self._probabilities * np.log(self._probabilities + 1e-12)))
        seen = self._counts > 0
        recorded = int(np.sum(self._counts[seen]))
        mean_rate = float(np.sum(self._success_sums[seen]) / max(recorded, 1))
        return {
            "sgs_sampling_entropy": entropy / float(np.log(max(num_bins, 2))),
            "sgs_sampling_top1_prob": float(np.max(self._probabilities)),
            "sgs_mean_success_rate": mean_rate,
        }

    def source_sampling_weights(self, num_sources: int) -> np.ndarray:
        """Clip-level sampling weights over the whole corpus.

        SGS counterpart of the adaptive sampler's difficulty weights: apply
        the same Beta-kernel to each source's mean bin success rate (unseen
        sources keep the 0.5 prior), so rotation spends part of each window's
        clip budget on clips near the policy's frontier. The kernel's
        ``epsilon`` already guarantees every source a non-zero weight.

        Cached between projections; the caller (the rotator, on the manager
        thread) only asks at decode start.
        """
        if self._weights_cache is not None and self._weights_cache.shape[0] == num_sources:
            return self._weights_cache
        rates = np.full(num_sources, _UNSEEN_SUCCESS_RATE)
        registered = np.flatnonzero(self._offset_lut >= 0)
        if registered.size:
            offsets = self._offset_lut[registered]
            counts = self._counts
            sums = np.where(counts > 0, self._success_sums, 0.0)
            totals = np.where(counts > 0, counts, 0)
            # Registered sources tile the flat arrays contiguously in id
            # order, so one reduceat per table yields per-source aggregates.
            block_success = np.add.reduceat(sums, offsets)
            block_totals = np.add.reduceat(totals, offsets)
            seen = block_totals > 0
            rates[registered[seen]] = block_success[seen] / block_totals[seen]
        weights = build_sgs_probabilities(rates, self._target, self._concentration, self._epsilon, self._temperature)
        weights /= np.sum(weights)
        self._weights_cache = weights
        return weights

    def bin_success_rate(self, source_id: int, bin_index: int) -> float | None:
        """Source-local bin success rate for tests and diagnostics."""
        offset = int(self._offset_lut[source_id])
        flat = offset + bin_index
        if self._counts[flat] == 0:
            return None
        return float(self._success_sums[flat] / self._counts[flat])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _register(self, source_id: int, frames: int) -> None:
        if source_id < self._offset_lut.size and self._offset_lut[source_id] >= 0:
            return
        count = -(-frames // self._bin_size)
        if source_id >= self._offset_lut.size:
            grown = source_id + 1 - self._offset_lut.size
            self._offset_lut = np.concatenate([self._offset_lut, np.full(grown, -1, np.int64)])
            self._count_lut = np.concatenate([self._count_lut, np.zeros(grown, np.int64)])
            self._frames_lut = np.concatenate([self._frames_lut, np.zeros(grown, np.int64)])
        self._outcomes = np.concatenate([self._outcomes, np.zeros((count, self._history), np.int8)])
        self._success_sums = np.concatenate([self._success_sums, np.zeros(count, np.float64)])
        self._counts = np.concatenate([self._counts, np.zeros(count, np.int64)])
        self._cursor = np.concatenate([self._cursor, np.zeros(count, np.int64)])
        self._offset_lut[source_id] = self._counts.size - count
        self._count_lut[source_id] = count
        self._frames_lut[source_id] = frames

    def _lane_bins(self, source_ids: np.ndarray, local_steps: np.ndarray) -> np.ndarray:
        """Map per-lane (source, local step) onto flat bin indices."""
        bins = np.minimum(local_steps // self._bin_size, self._count_lut[source_ids] - 1)
        return self._offset_lut[source_ids] + bins

    def _drain_outcomes(self, term) -> None:
        pending = term.adaptive_failure_pending
        rows = np.flatnonzero(pending)
        if rows.size == 0:
            return
        flat = self._lane_bins(
            term.adaptive_failure_source_id[rows].astype(np.int64),
            term.adaptive_failure_local_step[rows].astype(np.int64),
        )
        # Ring-buffer writes are order-sensitive within a bin; the batch holds
        # at most one outcome per lane, so a short loop stays exact.
        for bin_id, success in zip(flat, term.adaptive_outcome_success[rows]):
            cursor = int(self._cursor[bin_id])
            if self._counts[bin_id] == self._history:
                self._success_sums[bin_id] -= self._outcomes[bin_id, cursor]
            outcome = 1 if success else 0
            self._outcomes[bin_id, cursor] = outcome
            self._success_sums[bin_id] += outcome
            self._counts[bin_id] = min(int(self._counts[bin_id]) + 1, self._history)
            self._cursor[bin_id] = (cursor + 1) % self._history
        pending[rows] = False
        self._dirty = True

    def _bin_success_rates(self, flat: np.ndarray) -> np.ndarray:
        counts = self._counts[flat]
        rates = np.full(flat.size, _UNSEEN_SUCCESS_RATE)
        seen = counts > 0
        rates[seen] = self._success_sums[flat[seen]] / counts[seen]
        return rates

    def _project(self, term) -> None:
        clip = term.clip
        base = int(term.window_range[0])
        frames = int(term.window_range[1])
        bin_starts: list[int] = []
        bin_ends: list[int] = []
        bin_flat: list[int] = []
        for start, end in clip_segment_bounds(clip.frame_clip_end, base, frames):
            source_id = int(clip.frame_source_id[start])
            self._register(source_id, end - start)
            offset = int(self._offset_lut[source_id])
            for index in range(int(self._count_lut[source_id])):
                bin_start = start + index * self._bin_size
                bin_end = min(start + (index + 1) * self._bin_size, end)
                if bin_start < bin_end:
                    bin_starts.append(bin_start)
                    bin_ends.append(bin_end)
                    bin_flat.append(offset + index)

        count = len(bin_starts)
        assert count <= int(term.sampling_cdf.size)
        term.adaptive_active_bin_count[0] = count
        if count:
            flat = np.asarray(bin_flat, np.int64)
            probabilities = build_sgs_probabilities(
                self._bin_success_rates(flat),
                self._target,
                self._concentration,
                self._epsilon,
                self._temperature,
            )
            term.sampling_cdf[:count] = np.cumsum(probabilities, dtype=np.float32)
            term.sampling_cdf[count - 1] = 1.0
            term.adaptive_bin_start[:count] = bin_starts
            term.adaptive_bin_end[:count] = bin_ends
            self._probabilities = probabilities
        else:
            self._probabilities = np.empty(0, np.float32)
        self._projected = (base, frames)
        self._dirty = False
        self._steps_since_update = 0
        self._weights_cache = None

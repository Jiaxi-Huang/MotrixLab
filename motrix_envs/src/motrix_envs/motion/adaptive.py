# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-source adaptive sampling — gear_sonic's failure-rate curriculum.

This module answers one question for SONIC training: given everything episodes
have shown so far, where should the next episode start?  gear_sonic
(``motion_lib_base.py``) answers it with cumulative per-bin statistics over
every *source clip* in the corpus, and this module ports that contract onto
MotrixLab's host-side command terms:

- Bins are source-clip-local: every clip is split into ``bin_size``-frame
  bins (a 50 Hz corpus with ``bin_size=50`` gets one bin per second), so a
  bin never straddles two clips and failure credit always returns to the
  clip that produced it.
- Statistics are cumulative counters — ``failures`` and ``episodes`` both
  start at ``init_count`` and only ever grow.  There is no EMA and no
  spatial kernel: a cumulative rate estimate is already smooth, cannot
  self-reinforce through sampling volume alone, and upstream ships neither.
- Exposure is upstream's episode accounting: each lane-step contributes
  ``1 / bin_length`` episodes to the bin it occupies, so one lane traversing
  a full bin is one episode of exposure regardless of bin length.
- Probabilities are uniform-blended, outlier-clipped failure rates reshaped
  by sequence-length-agnostic per-clip bin weights, optionally capped per
  bin (water-filled so the bound survives normalization).

The :class:`AdaptiveMotionSampler` owns the statistics and projects them onto
a command term's active sampling window.  Resident corpora and rotated
windows look identical from here — a frame range plus per-frame source
provenance — so both paths share one mechanism, and a draining lane's
statistics stay with its source clip across window swaps.
"""

from __future__ import annotations

import numpy as np

from motrix_envs.motion.sampler import adaptive_sampling_metrics

__all__ = [
    "AdaptiveMotionSampler",
    "build_adaptive_probabilities",
    "cap_per_bin_probability",
    "clip_segment_bounds",
    "count_segment_bins",
]


# Exponential decay per control step on the realized-rate counters; a
# half-life of ~7k steps (~2.5 min at 50 Hz) spans many episodes while still
# tracking distribution shifts from adaptive sampling and rotation swaps.
_REALIZED_RATE_DECAY = 1e-4


def cap_per_bin_probability(probability: np.ndarray, max_prob_per_bin: np.float32) -> np.ndarray:
    """Cap every bin at ``max_prob_per_bin / num_bins`` via water-filling.

    Clipping plus a single renormalization cannot hold the bound — the mass
    pushed off the capped bins inflates the remaining ones back above the cap.
    Iteratively clip and redistribute the surplus over the uncapped bins until
    every bin sits at or below its fair-share multiple. Requires
    ``max_prob_per_bin >= 1`` so the cap is at or above the uniform level and
    the final normalization preserves it.
    """
    num_bins = probability.size
    if num_bins == 0:
        return probability
    cap = max_prob_per_bin / np.float32(num_bins)
    p = probability.astype(np.float32, copy=True)
    for _ in range(32):
        over = p > cap
        if not over.any():
            break
        surplus = float(np.sum(p[over] - cap))
        p[over] = cap
        free = ~over
        free_mass = float(np.sum(p[free]))
        if free_mass <= 0.0:
            p[:] = cap
            break
        p[free] *= np.float32((free_mass + surplus) / free_mass)
        if not np.any(p > cap):
            break
    total = float(np.sum(p))
    if total > 0.0 and abs(total - 1.0) > 1e-6:
        p *= np.float32(1.0 / total)
    return p


def build_adaptive_probabilities(
    failures: np.ndarray,
    episodes: np.ndarray,
    bin_weights: np.ndarray,
    uniform_ratio: np.float32,
    failure_rate_max_over_mean: np.float32,
    max_prob_per_bin: np.float32,
) -> np.ndarray:
    """Build per-bin start probabilities from cumulative failure rates.

    Port of gear_sonic ``update_adaptive_sampling_probabilities``: the signal
    is the per-bin failure rate ``failures / episodes`` (both counters start
    at one and accumulate for the collector's lifetime, so the estimate is
    exposure-normalized).  The rate is clipped at ``mean(rate) *
    failure_rate_max_over_mean``, blended with a uniform floor, and shaped by
    ``bin_weights`` (which equalize the total probability mass of every clip).
    An optional final water-filled cap bounds any single bin at
    ``max_prob_per_bin`` times the uniform level; upstream ships the knob but
    leaves it unset in the release config.
    """
    num_bins = failures.size
    if num_bins == 0:
        return np.empty((0,), dtype=np.float32)

    rate = failures / np.maximum(episodes, np.float32(1e-6))
    mean_rate = float(np.mean(rate))
    if mean_rate > 0.0 and failure_rate_max_over_mean > 0.0:
        rate = np.clip(rate, np.float32(0.0), np.float32(mean_rate * failure_rate_max_over_mean))

    total = float(np.sum(rate))
    uniform = np.full((num_bins,), 1.0 / num_bins, dtype=np.float32)
    if total <= 0.0 or not np.isfinite(total):
        return uniform
    failure_based = rate / total
    probability = (np.float32(1.0) - uniform_ratio) * failure_based + uniform_ratio * uniform
    probability = probability * bin_weights
    total = float(np.sum(probability))
    if total <= 0.0 or not np.isfinite(total):
        return uniform
    probability = probability / total
    if max_prob_per_bin > 0.0:
        probability = cap_per_bin_probability(probability, max_prob_per_bin)
    return probability.astype(np.float32)


def clip_segment_bounds(frame_clip_end: np.ndarray, base: int, frames: int) -> list[tuple[int, int]]:
    """Walk per-frame clip-end boundaries into ``[base, base + frames)`` segments.

    A degenerate boundary (end at or before the segment start) would otherwise
    stall the walk into an unbounded loop, so it always advances by one frame
    — corrupt data degrades to wrong bins instead of exhausting memory.
    """
    bounds = []
    start = base
    limit = base + frames
    while start < limit:
        end = max(int(frame_clip_end[start]) + 1, start + 1)
        bounds.append((start, end))
        start = end
    return bounds


def count_segment_bins(bounds: list[tuple[int, int]], bin_size: int) -> int:
    """Exact bin count for a segment layout: one bin per ``ceil(frames / bin_size)``."""
    return sum(-(-(end - start) // bin_size) for start, end in bounds)


class AdaptiveMotionSampler:
    """Host-side failure-rate sampler shared by resident and rotated SONIC corpora.

    Statistics live in flat arrays addressed through per-source lookup tables,
    so per-step accumulation is one ``bincount`` over per-lane flat indices.
    The compiled command keeps only the sampling tables (CDF plus active bin
    frame ranges) that its reset kernel reads; this class republishes them
    every ``update_period`` control steps with pending statistics, and eagerly
    whenever the active window changes.
    """

    def __init__(
        self,
        *,
        bin_size: int,
        init_count: float = 1.0,
        uniform_ratio: float = 0.1,
        failure_rate_max_over_mean: float = 50.0,
        sequence_length_agnostic: bool = True,
        max_prob_per_bin: float | None = None,
        update_period: int = 128,
    ):
        self._bin_size = int(bin_size)
        self._init_count = np.float32(init_count)
        self._uniform_ratio = np.float32(uniform_ratio)
        self._failure_rate_max_over_mean = np.float32(failure_rate_max_over_mean)
        self._sequence_length_agnostic = sequence_length_agnostic
        self._max_prob_per_bin = np.float32(max_prob_per_bin) if max_prob_per_bin is not None else np.float32(0.0)
        self._update_period = int(update_period)
        # Flat per-bin counters plus dense per-source lookup tables (source
        # ids are corpus indices, so a dense LUT is cheap and O(1) per lane).
        self._failures = np.empty(0, np.float32)
        self._episodes = np.empty(0, np.float32)
        self._offset_lut = np.empty(0, np.int64)
        self._count_lut = np.empty(0, np.int64)
        self._frames_lut = np.empty(0, np.int64)
        self._steps_since_update = 0
        self._dirty = False
        self._projected: tuple[int, int] | None = None
        self._probabilities = np.empty(0, np.float32)
        self._weights_cache: np.ndarray | None = None
        # Hot-path state: the per-step cost stays O(envs + window bins)
        # regardless of corpus size.  Exposure accumulates into a window-
        # local histogram (folded into the global counters at each
        # republish), failures scatter directly and keep an incremental
        # total, and the metrics dict is cached between republishes.
        self._failure_total = 0.0
        self._metrics_cache: dict[str, float] | None = None
        self._pending_exposure = np.empty(0, np.float64)
        self._window_flat = np.empty(0, np.int64)
        self._window_sources = np.empty(0, np.int64)
        # Realized performance under the current sampling distribution:
        # failures and episode exposure accumulated with the same per-step
        # decay, so their ratio is the policy's failure rate over what
        # adaptive sampling is actually serving right now.
        self._rate_decay = 1.0 - _REALIZED_RATE_DECAY
        self._rate_failures = 0.0
        self._rate_exposure = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def attach(self, term) -> None:
        """Project the term's current window and publish the initial tables."""
        self._project(term)

    def tick(self, term) -> dict[str, float]:
        """Republish on window changes, drain failures, accumulate exposure.

        Called once per control step with the canonical motion command; the
        returned metrics ride the step result into the trainer's curves. The
        window check runs first so a just-published window's sources are
        registered before any lane can reference them — lanes migrate through
        resets at any time relative to the publish.
        """
        window = (int(term.window_range[0]), int(term.window_range[1]))
        if window != self._projected:
            # A window publication changes the sampling domain; republish
            # immediately so the next reset never sees stale tables.
            self._project(term)
        self._rate_failures *= self._rate_decay
        self._rate_exposure *= self._rate_decay
        self._drain_failures(term)
        self._accumulate_exposure(term)
        if self._dirty:
            self._steps_since_update += 1
            if self._steps_since_update >= self._update_period:
                self._project(term)
        return self.metrics()

    def metrics(self) -> dict[str, float]:
        """Sampling metrics cached at each republish, plus the realized
        ``adaptive_failure_rate`` — failures over episode exposure under the
        current sampling distribution, exponentially decayed (half-life
        ~7k control steps). Read it against ``adaptive_sampling_entropy`` /
        ``top1_prob`` to tell policy progress from distribution shift: a
        falling rate while the sampler concentrates means the policy is
        mastering what it is being served.
        """
        base = self._metrics_cache
        if base is None:
            base = adaptive_sampling_metrics(self._probabilities, self._failure_total)
        rate = self._rate_failures / max(self._rate_exposure, 1e-6)
        return {**base, "adaptive_failure_rate": float(rate)}

    def source_sampling_weights(self, num_sources: int) -> np.ndarray:
        """Clip-level sampling weights over the whole corpus.

        Port of gear_sonic ``update_adaptive_sampling_motion_sequences``
        (what their motion resample draws the next motion batch from):
        aggregate each source's per-bin failure rates into a clip score —
        unseen sources keep the initial rate estimate of one — clip at the
        same mean x ``failure_rate_max_over_mean`` bound, blend with a
        ``uniform_ratio`` uniform floor, and normalize. Rotation spends part
        of each window's clip budget on draws from this distribution, so hard
        clips reappear more often across windows.

        Cached between projections; the caller (the rotator, on the manager
        thread) only asks at decode start.
        """
        if self._weights_cache is not None and self._weights_cache.shape[0] == num_sources:
            return self._weights_cache
        self._fold_exposure()
        scores = np.ones(num_sources, np.float64)
        if self._failures.size:
            rates = (self._failures / self._episodes).astype(np.float64)
            mean_rate = float(np.mean(rates))
            if mean_rate > 0.0 and self._failure_rate_max_over_mean > 0.0:
                rates = np.clip(rates, 0.0, mean_rate * float(self._failure_rate_max_over_mean))
            registered = np.flatnonzero(self._offset_lut >= 0)
            bin_owner = np.repeat(registered, self._count_lut[registered])
            totals = np.bincount(bin_owner, weights=rates, minlength=num_sources)
            counts = np.bincount(bin_owner, minlength=num_sources)
            seen = counts > 0
            scores[seen] = totals[seen] / counts[seen]
        uniform = np.full(num_sources, 1.0 / num_sources)
        share = scores / float(np.sum(scores))
        weights = (1.0 - float(self._uniform_ratio)) * share + float(self._uniform_ratio) * uniform
        weights /= float(np.sum(weights))
        self._weights_cache = weights
        return weights

    @property
    def failure_mass(self) -> float:
        """Failures accumulated beyond the ``init_count`` prior."""
        return self._failure_total

    def source_failure_mass(self, source_id: int) -> float:
        """Source-local failure mass for tests and diagnostics."""
        offset = int(self._offset_lut[source_id])
        count = int(self._count_lut[source_id])
        return float(np.sum(self._failures[offset : offset + count]) - self._init_count * count)

    def source_episode_mass(self, source_id: int) -> float:
        """Source-local episode exposure, including not-yet-folded pending mass."""
        offset = int(self._offset_lut[source_id])
        count = int(self._count_lut[source_id])
        total = float(np.sum(self._episodes[offset : offset + count]) - self._init_count * count)
        if self._window_sources.size:
            mask = self._window_sources == source_id
            if mask.any():
                total += float(np.sum(self._pending_exposure[: self._window_sources.size][mask]))
        return total

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
        block = np.full(count, self._init_count, np.float32)
        self._failures = np.concatenate([self._failures, block])
        self._episodes = np.concatenate([self._episodes, block])
        self._offset_lut[source_id] = self._failures.size - count
        self._count_lut[source_id] = count
        self._frames_lut[source_id] = frames

    def _lane_bins(self, source_ids: np.ndarray, local_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map per-lane (source, local step) onto flat bin indices and bin lengths."""
        bins = np.minimum(local_steps // self._bin_size, self._count_lut[source_ids] - 1)
        lengths = np.minimum(self._bin_size, self._frames_lut[source_ids] - bins * self._bin_size)
        return self._offset_lut[source_ids] + bins, lengths

    @staticmethod
    def _scatter_add(target: np.ndarray, flat: np.ndarray, weights: np.ndarray) -> None:
        """Compact ``target[flat] += weights`` without touching the whole array."""
        unique, inverse = np.unique(flat, return_inverse=True)
        target[unique] += np.bincount(inverse, weights=weights)

    def _fold_exposure(self) -> None:
        """Fold the window-local exposure histogram into the global counters."""
        count = int(self._window_flat.size)
        if count:
            self._episodes[self._window_flat] += self._pending_exposure[:count]
            self._pending_exposure[:count] = 0.0

    def _drain_failures(self, term) -> None:
        pending = term.adaptive_failure_pending
        rows = np.flatnonzero(pending)
        if rows.size == 0:
            return
        flat, _ = self._lane_bins(
            term.adaptive_failure_source_id[rows].astype(np.int64),
            term.adaptive_failure_local_step[rows].astype(np.int64),
        )
        self._scatter_add(self._failures, flat, np.ones(rows.size))
        self._failure_total += float(rows.size)
        self._rate_failures += float(rows.size)
        pending[rows] = False
        self._dirty = True

    def _accumulate_exposure(self, term) -> None:
        """Account one episode exposure unit per lane-step, upstream-style.

        Lanes inside the active window accumulate into the window-local
        histogram (searchsorted over the published bin table).  Lanes still
        draining a retired window scatter straight into their own source's
        global bins — source attribution survives window swaps either way.
        """
        steps = term.steps[:, 0].astype(np.int64)
        base = int(term.window_range[0])
        frames = int(term.window_range[1])
        count = int(term.adaptive_active_bin_count[0])
        in_window = (steps >= base) & (steps < base + frames)
        if count and in_window.any():
            starts = np.asarray(term.adaptive_bin_start[:count])
            ends = np.asarray(term.adaptive_bin_end[:count])
            slot = np.searchsorted(starts, steps[in_window], side="right") - 1
            lengths = (ends[slot] - starts[slot]).astype(np.float64)
            weights = 1.0 / lengths
            self._pending_exposure[:count] += np.bincount(slot, weights=weights, minlength=count)
            self._rate_exposure += float(np.sum(weights))
        if not in_window.all():
            outside = ~in_window
            source_ids = term.clip.frame_source_id[steps[outside]].astype(np.int64)
            local_steps = (steps[outside] - term.clip.frame_source_start[steps[outside]]).astype(np.int64)
            flat, lengths = self._lane_bins(source_ids, local_steps)
            weights = 1.0 / lengths
            self._scatter_add(self._episodes, flat, weights)
            self._rate_exposure += float(np.sum(weights))
        self._dirty = True

    def _project(self, term) -> None:
        # Fold the retiring window's pending exposure first: probabilities
        # below must see every accounted episode.
        self._fold_exposure()
        clip = term.clip
        base = int(term.window_range[0])
        frames = int(term.window_range[1])
        bin_starts: list[int] = []
        bin_ends: list[int] = []
        bin_sources: list[int] = []
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
                    bin_sources.append(source_id)
                    bin_flat.append(offset + index)

        count = len(bin_starts)
        assert count <= int(term.sampling_cdf.size)
        term.adaptive_active_bin_count[0] = count
        if count:
            flat = np.asarray(bin_flat, np.int64)
            lengths = np.asarray(np.asarray(bin_ends) - np.asarray(bin_starts), np.float64)
            sources = np.asarray(bin_sources, np.int64)
            _, inverse, peer_counts = np.unique(sources, return_inverse=True, return_counts=True)
            # gear_sonic bin weights: bin_length / mean / peer count.  The
            # peer division equalizes total mass per clip; upstream's
            # ``sequence_length_agnostic`` flag gates it.
            weights = lengths / float(np.mean(lengths))
            if self._sequence_length_agnostic:
                weights = weights / peer_counts[inverse]
            probabilities = build_adaptive_probabilities(
                self._failures[flat],
                self._episodes[flat],
                weights.astype(np.float32),
                self._uniform_ratio,
                self._failure_rate_max_over_mean,
                self._max_prob_per_bin,
            )
            term.sampling_cdf[:count] = np.cumsum(probabilities, dtype=np.float32)
            term.sampling_cdf[count - 1] = 1.0
            term.adaptive_bin_start[:count] = bin_starts
            term.adaptive_bin_end[:count] = bin_ends
            self._probabilities = probabilities
        else:
            self._probabilities = np.empty(0, np.float32)
        # Window bookkeeping for the deferred exposure fold: slot -> flat bin
        # and slot -> source, plus the window-local histogram itself.  The
        # fold above already emptied the retiring window's mass, so growing
        # the buffer never loses anything.
        self._window_flat = np.asarray(bin_flat, np.int64)
        self._window_sources = np.asarray(bin_sources, np.int64)
        if self._pending_exposure.size < term.sampling_cdf.size:
            self._pending_exposure = np.zeros(term.sampling_cdf.size, np.float64)
        self._metrics_cache = adaptive_sampling_metrics(self._probabilities, self._failure_total)
        self._projected = (base, frames)
        self._dirty = False
        self._steps_since_update = 0
        self._weights_cache = None

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime corpus rotation — stream a large motion dataset through a fixed clip.

Off-policy training (FastSAC) replays transitions from a buffer, so the env
does not need the whole corpus resident: :class:`CorpusRotator` splits the
corpus into windows of clips and publishes them into alternating halves of one
pre-allocated :class:`~motrix_envs.motion.sonic.SonicMotionClip`. Compiled
kernels receive the clip arrays as arguments on every call, so in-place writes
are live, while array objects must never be replaced — hence the fixed
capacity.

The rotation contract:

- Coverage: windows consume a seeded without-replacement permutation of the
  full corpus, so every clip is visited once per pass over the permutation.
- Speed: window decode runs on background threads into private staging arrays;
  publication is one in-place copy into the retiring half. Physics stepping
  only ever blocks for that copy.
- Stability: publication flips the reset-sampling domain to the new window but
  never force-resets envs. Lanes migrate through their natural episode resets
  and clip-end re-anchoring, and the retiring half is overwritten only once no
  lane's step index references it, so no artificial truncation enters the
  replay buffer and the buffer's uniform sampling smooths each swap into a
  gradual mixture shift.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motrix_env_core.numba.kernel_data import Map
from motrix_envs.motion.library import (
    FRAME_AXIS_FIELDS,
    assemble_segment_arrays,
    check_motion_matches_corpus,
    expand_motion_paths,
    load_motion_segments,
    motion_corpus_contract,
)
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.sonic import SonicMotionClip
from motrix_envs.motion.tracked import WbtMotionClip

logger = logging.getLogger(__name__)

# Channels SONIC hoists from ``extensions`` onto dedicated clip fields; the
# rotator stages and publishes them alongside the frame-axis fields.
_SONIC_EXTENSION_CHANNELS = ("smpl_joints", "smpl_root_quat")

# Extra frame headroom granted to later windows over the first window's size,
# so clip-length variance across windows rarely forces a trim.
_WINDOW_HEADROOM = 1.25

_IDLE = "idle"
_DECODING = "decoding"
_STAGING = "staging"


@dataclass(frozen=True)
class _DecodedSegment:
    """One decoded corpus member paired with its stable source entry id."""

    source_id: int
    clip: WbtMotionClip


def _assemble_window(
    segments: list[_DecodedSegment],
    include_sources: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Assemble one window and retain per-frame source provenance sidecars."""
    clips = [segment.clip for segment in segments]
    fields, frame_clip_end, extensions = assemble_segment_arrays(clips, _SONIC_EXTENSION_CHANNELS)
    if not include_sources:
        return fields, frame_clip_end, extensions, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    source_id = np.empty(frame_clip_end.size, dtype=np.int64)
    source_start = np.empty(frame_clip_end.size, dtype=np.int64)
    start = 0
    for segment in segments:
        end = start + segment.clip.joint_pos.shape[0]
        source_id[start:end] = segment.source_id
        source_start[start:end] = start
        start = end
    return fields, frame_clip_end, extensions, source_id, source_start


class AdaptiveRotationSampler:
    """Host-side, source-stable adaptive state for streamed SONIC corpora.

    Statistics are allocated only after a corpus entry is decoded.  The compiled
    command receives just the CDF and frame ranges of the active window, while
    failure events retain their source id so retired-window lanes stay valid.
    """

    def __init__(
        self,
        *,
        bin_size: int,
        alpha: float,
        uniform_ratio: float,
        kernel_size: int,
        kernel_lambda: float,
        update_period: int = 128,
    ):
        self._bin_size = bin_size
        self._alpha = np.float32(alpha)
        self._uniform_ratio = np.float32(uniform_ratio)
        self._kernel_size = kernel_size
        self._kernel_lambda = np.float32(kernel_lambda)
        self._kernel = np.asarray([kernel_lambda**i for i in range(max(kernel_size, 1))], dtype=np.float32)
        self._kernel /= np.sum(self._kernel)
        self._update_period = update_period
        self._steps_since_update = 0
        self._dirty = False
        self._history: dict[int, np.ndarray] = {}
        self._current: dict[int, np.ndarray] = {}
        self.entropy = 1.0
        self.top1_prob = 0.0
        self.top1_bin = 0.0
        self.failure_mass = 0.0

    def attach(self, term) -> None:
        self._write_active_distribution(term)
        self._steps_since_update = 0
        self._dirty = False

    def tick(self, term) -> None:
        pending = term.adaptive_failure_pending
        if np.any(pending):
            failed_rows = np.flatnonzero(pending)
            touched: set[int] = set()
            for row in failed_rows:
                source_id = int(term.adaptive_failure_source_id[row])
                local_step = int(term.adaptive_failure_local_step[row])
                history = self._history[source_id]
                current = self._current[source_id]
                current[min(local_step // self._bin_size, current.size - 1)] += 1.0
                touched.add(source_id)
            pending[failed_rows] = False

            for source_id in touched:
                history = self._history[source_id]
                current = self._current[source_id]
                # A source may be absent for many rotation periods.  Accumulate
                # its observed failure evidence rather than applying the legacy
                # per-control-step decay, which would erase it before return.
                history += self._alpha * current
                current.fill(0.0)
            self._dirty = True

        if self._dirty:
            self._steps_since_update += 1
            if self._steps_since_update >= self._update_period:
                self._write_active_distribution(term)
                self._steps_since_update = 0
                self._dirty = False

    def source_failure_mass(self, source_id: int) -> float:
        """Return source-local state for rotation tests and diagnostics."""
        return float(np.sum(self._history.get(source_id, 0.0)))

    def _window_segments(self, term):
        start = int(term.window_range[0])
        limit = start + int(term.window_range[1])
        while start < limit:
            end = int(term.clip.frame_clip_end[start]) + 1
            yield int(term.clip.frame_source_id[start]), start, end
            start = end

    def _ensure_source(self, source_id: int, start: int, end: int) -> np.ndarray:
        history = self._history.get(source_id)
        if history is None:
            # The final motion frame cannot be a reset target; at least one
            # bin remains for the two-frame minimum clip contract.
            count = max(1, math.ceil((end - start - 1) / self._bin_size))
            history = np.zeros(count, dtype=np.float32)
            self._history[source_id] = history
            self._current[source_id] = np.zeros_like(history)
        return history

    def _smoothed(self, values: np.ndarray) -> np.ndarray:
        if self._kernel_size <= 1:
            return values
        return np.correlate(np.pad(values, (0, self._kernel_size - 1), mode="edge"), self._kernel, mode="valid")

    def _write_active_distribution(self, term) -> None:
        values: list[np.ndarray] = []
        starts: list[int] = []
        ends: list[int] = []
        for source_id, start, end in self._window_segments(term):
            history = self._ensure_source(source_id, start, end)
            safe_end = end - 1
            for bin_id in range(history.size):
                bin_start = start + bin_id * self._bin_size
                bin_end = min(bin_start + self._bin_size, safe_end)
                if bin_start < bin_end:
                    starts.append(bin_start)
                    ends.append(bin_end)
                    values.append(self._smoothed(history)[bin_id : bin_id + 1])

        count = len(starts)
        assert count <= term.sampling_cdf.size
        term.adaptive_active_bin_count[0] = count
        if not count:
            return
        adaptive = np.concatenate(values)
        total = float(np.sum(adaptive))
        if total > 0.0:
            adaptive /= total
        else:
            adaptive.fill(1.0 / count)
        probabilities = (np.float32(1.0) - self._uniform_ratio) * adaptive + self._uniform_ratio / count
        term.sampling_cdf[:count] = np.cumsum(probabilities, dtype=np.float32)
        term.sampling_cdf[count - 1] = 1.0
        term.adaptive_bin_start[:count] = starts
        term.adaptive_bin_end[:count] = ends
        entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
        self.entropy = entropy / float(np.log(max(count, 2)))
        top = int(np.argmax(probabilities))
        self.top1_prob = float(probabilities[top])
        self.top1_bin = float(top) / float(count)
        self.failure_mass = float(sum(np.sum(value) for value in self._history.values()))


class CorpusRotator:
    """Swap windows of a motion corpus through a two-window frame axis.

    The frame axis is split into two halves of ``frame_budget`` frames. The
    active half serves reset sampling (its base/frames are published to the
    command's ``window_range``); the other half receives the next window once
    every lane has left it. :meth:`tick` is called once per control step from
    the command's host transition hook and drives the
    idle → decode → stage → publish state machine.
    """

    def __init__(
        self,
        *,
        entries: tuple[Path, ...],
        order: np.ndarray,
        rng: np.random.Generator,
        frame_budget: int,
        first_frames: int,
        window_clips: int,
        period: int,
        workers: int,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        fps: int,
        corpus_contract: tuple[frozenset[str], frozenset[str], tuple[tuple[int, ...], ...], Path],
        track_sources: bool,
    ) -> None:
        self._entries = entries
        self._order = order
        self._rng = rng
        # Window 0 already consumed the permutation prefix during build.
        self._cursor = window_clips
        # Entries consumed from the permutation within the current pass;
        # resets when the permutation wraps and reshuffles.
        self._pass_consumed = window_clips
        self._frame_budget = frame_budget
        self._window_clips = window_clips
        self._period = period
        self._workers = workers
        self._joint_names = joint_names
        self._tracked_body_names = tracked_body_names
        self._reference_body_name = reference_body_name
        self._root_body_name = root_body_name
        self._fps = fps
        self._corpus_contract = corpus_contract
        self._track_sources = track_sources

        self._pool: ThreadPoolExecutor | None = None
        self._decode_futures: list[Future[_DecodedSegment | None]] = []
        self._assemble_future: Future | None = None
        self._pending_clips = 0
        self._carry: list[_DecodedSegment] = []
        self._adaptive: AdaptiveRotationSampler | None = None
        self._phase = _IDLE
        self._ticks = 0
        self._last_swap_tick = 0
        self._decode_started_tick = 0
        self._staging_started_tick = 0
        self._active_half = 0
        self._half_frames = [first_frames, 0]
        self._window_index = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        paths: Sequence[str | Path],
        window_clips: int,
        period: int,
        workers: int,
        seed: int,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        track_sources: bool = False,
        fps: int = 50,
    ) -> tuple[SonicMotionClip, CorpusRotator | None]:
        """Decode the first window and allocate the two-window capacity axis.

        The first window holds exactly ``window_clips`` clips (validated as one
        batch, like a static corpus load) and defines the frame budget; the
        axis is sized for both halves up front because the arrays must never
        be reallocated mid-training. A corpus that already fits one window
        returns ``(static clip, None)`` — rotation is pointless there.
        """
        entries = expand_motion_paths(paths)
        if len(entries) <= window_clips:
            # A corpus that fits one window is fully resident already; keep
            # smoke corpora on the plain static path instead of failing.
            logger.info(
                "Corpus rotation disabled: %d motion entries fit one window of %d clips.",
                len(entries),
                window_clips,
            )
            segments = load_motion_segments(
                list(entries),
                joint_names=joint_names,
                tracked_body_names=tracked_body_names,
                reference_body_name=reference_body_name,
                root_body_name=root_body_name,
                fps=fps,
                extension_channels=_SONIC_EXTENSION_CHANNELS,
            )
            frames = int(sum(segment.joint_pos.shape[0] for segment in segments))
            first_window = [_DecodedSegment(index, segment) for index, segment in enumerate(segments)]
            return _allocate_sonic_clip(frames, first_window, track_sources), None
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(entries))
        first_entries = [entries[int(index)] for index in order[:window_clips]]
        segments = load_motion_segments(
            first_entries,
            joint_names=joint_names,
            tracked_body_names=tracked_body_names,
            reference_body_name=reference_body_name,
            root_body_name=root_body_name,
            fps=fps,
            extension_channels=_SONIC_EXTENSION_CHANNELS,
        )
        # Identity contract every later window is checked against; re-reading
        # the first file is a one-time cost that keeps the window-0 decode on
        # the shared batch path.
        contract = motion_corpus_contract(
            MotrixMotion(first_entries[0]), fps=fps, extension_channels=_SONIC_EXTENSION_CHANNELS
        )
        first_frames = int(sum(segment.joint_pos.shape[0] for segment in segments))
        frame_budget = int(math.ceil(first_frames * _WINDOW_HEADROOM))
        first_segments = [
            _DecodedSegment(int(index), segment) for index, segment in zip(order[:window_clips], segments)
        ]
        clip = _allocate_sonic_clip(2 * frame_budget, first_segments, track_sources)
        rotator = cls(
            entries=entries,
            order=order,
            rng=rng,
            frame_budget=frame_budget,
            first_frames=first_frames,
            window_clips=window_clips,
            period=period,
            workers=workers,
            joint_names=joint_names,
            tracked_body_names=tracked_body_names,
            reference_body_name=reference_body_name,
            root_body_name=root_body_name,
            fps=fps,
            corpus_contract=contract,
            track_sources=track_sources,
        )
        return clip, rotator

    @property
    def active_frames(self) -> int:
        """Frames of the currently published window."""
        return self._half_frames[self._active_half]

    @property
    def active_base(self) -> int:
        """Base frame index of the currently published window."""
        return self._active_half * self._frame_budget

    @property
    def window_index(self) -> int:
        """Number of published rotations; the first resident window is 0."""
        return self._window_index

    @property
    def pass_coverage(self) -> float:
        """Fraction of the corpus consumed in the current permutation pass."""
        return self._pass_consumed / len(self._entries)

    def max_adaptive_bins(self, bin_size: int) -> int:
        """Fixed CDF capacity for any window without inspecting the full corpus."""
        return math.ceil(self._frame_budget / bin_size) + self._window_clips

    def enable_adaptive(
        self,
        *,
        bin_size: int,
        alpha: float,
        uniform_ratio: float,
        kernel_size: int,
        kernel_lambda: float,
        update_period: int = 128,
    ) -> None:
        self._adaptive = AdaptiveRotationSampler(
            bin_size=bin_size,
            alpha=alpha,
            uniform_ratio=uniform_ratio,
            kernel_size=kernel_size,
            kernel_lambda=kernel_lambda,
            update_period=update_period,
        )

    def attach(self, term) -> None:
        """Bind host-only adaptive state once the canonical command exists."""
        if self._adaptive is not None:
            self._adaptive.attach(term)

    @property
    def adaptive(self) -> AdaptiveRotationSampler | None:
        return self._adaptive

    # ------------------------------------------------------------------
    # Per-step driver
    # ------------------------------------------------------------------
    def tick(self, term) -> None:
        """Advance the rotation state machine; called once per control step.

        ``term`` is the owning :class:`SonicMotionCommand`; only its host-side
        arrays (``steps``, ``window_range``, adaptive accumulators) are read or
        written here, never from a decoding thread.
        """
        self._ticks += 1
        if self._adaptive is not None:
            self._adaptive.tick(term)
        if self._phase == _IDLE:
            if self._ticks - self._last_swap_tick >= self._period:
                self._begin_decode()
        elif self._phase == _DECODING:
            if all(future.done() for future in self._decode_futures):
                self._begin_staging()
        elif self._phase == _STAGING:
            assert self._assemble_future is not None
            if self._assemble_future.done() and self._retired_half_drained(term.steps):
                self._publish(term, self._assemble_future.result())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _begin_decode(self) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="motion-rotation")
        self._decode_started_tick = self._ticks
        # The cursor advances on the manager thread only, keeping permutation
        # state single-threaded while the pool decodes files in parallel.
        batch = [self._next_entry() for _ in range(self._window_clips - len(self._carry))]
        self._decode_futures = [self._pool.submit(self._decode_or_skip, source_id, path) for source_id, path in batch]
        self._phase = _DECODING

    def _begin_staging(self) -> None:
        segments = [future.result() for future in self._decode_futures]
        self._decode_futures = []
        segments = [segment for segment in segments if segment is not None] + self._carry
        self._carry = []
        if not segments:
            self._skip_cycle("decoded no usable clips")
            return
        # Trim to the frame budget. Everything from the first non-fitting
        # segment onward stays decoded in the carry list and leads the next
        # window, so budget trims never lose corpus coverage.
        kept: list[_DecodedSegment] = []
        used = 0
        for index, segment in enumerate(segments):
            frames = int(segment.clip.joint_pos.shape[0])
            if kept and used + frames > self._frame_budget:
                self._carry = segments[index:]
                break
            if not kept and frames > self._frame_budget:
                logger.warning(
                    "Skipping motion clip of %d frames: it alone exceeds the %d-frame window budget.",
                    frames,
                    self._frame_budget,
                )
                continue
            kept.append(segment)
            used += frames
        if not kept:
            self._skip_cycle("every clip exceeded the window budget")
            return
        assert self._pool is not None
        self._pending_clips = len(kept)
        if self._track_sources:
            self._assemble_future = self._pool.submit(_assemble_window, kept, True)
        else:
            # Keep uniform rotation's staging call exactly on its original
            # path; source provenance is an adaptive-only allocation.
            self._assemble_future = self._pool.submit(
                assemble_segment_arrays, [segment.clip for segment in kept], _SONIC_EXTENSION_CHANNELS
            )
        self._staging_started_tick = self._ticks
        self._phase = _STAGING

    def _skip_cycle(self, reason: str) -> None:
        logger.warning("Motion rotation window %d skipped (%s); retrying next period.", self._window_index + 1, reason)
        self._phase = _IDLE
        self._last_swap_tick = self._ticks

    def _next_entry(self) -> tuple[int, Path]:
        source_id = int(self._order[self._cursor])
        entry = self._entries[source_id]
        self._cursor += 1
        self._pass_consumed += 1
        if self._cursor >= len(self._order):
            # One pass over the corpus is complete; reshuffle and start the
            # next pass from the continuing seeded stream.
            self._order = self._rng.permutation(len(self._entries))
            self._cursor = 0
            self._pass_consumed = 0
        return source_id, entry

    def _retired_half_drained(self, steps: np.ndarray) -> bool:
        target = 1 - self._active_half
        previous = self._half_frames[target]
        if previous == 0:
            return True
        base = target * self._frame_budget
        return not bool(np.any((steps >= base) & (steps < base + previous)))

    def _publish(self, term, staging) -> None:
        if self._track_sources:
            fields, frame_clip_end, extensions, source_id, source_start = staging
        else:
            fields, frame_clip_end, extensions = staging
        target = 1 - self._active_half
        base = target * self._frame_budget
        frames = int(frame_clip_end.shape[0])
        # ``term.clip`` is the env-owned (canonical) copy the compiled kernels
        # read; in-place writes here are immediately live for them.
        clip = term.clip
        for field in FRAME_AXIS_FIELDS:
            getattr(clip, field)[base : base + frames] = fields[field]
        clip.frame_clip_end[base : base + frames] = frame_clip_end + base
        if self._track_sources:
            clip.frame_source_id[base : base + frames] = source_id
            clip.frame_source_start[base : base + frames] = source_start + base
        for name in _SONIC_EXTENSION_CHANNELS:
            getattr(clip, name)[base : base + frames] = extensions[name]
        term.window_range[:] = (base, frames)
        num_bins = int(term.adaptive_bin_failed_count.size)
        if num_bins and self._adaptive is None:
            # Bins live on the window's own frame timeline, so failure mass
            # cannot carry across a swap; restart from the uniform floor.
            term.adaptive_bin_failed_count.fill(0.0)
            term.adaptive_current_bin_failed_count.fill(0.0)
            term.sampling_cdf[:] = np.arange(1, num_bins + 1, dtype=np.float32) / np.float32(num_bins)
        self._half_frames[target] = frames
        self._active_half = target
        self._window_index += 1
        self._assemble_future = None
        self._phase = _IDLE
        self._last_swap_tick = self._ticks
        if self._adaptive is not None:
            self._adaptive.attach(term)
        logger.info(
            "Motion rotation: window %d published (%d clips, %d frames at base %d; "
            "decode+stage %d ticks, drain %d ticks, pass coverage %.1f%%).",
            self._window_index,
            self._pending_clips,
            frames,
            base,
            self._staging_started_tick - self._decode_started_tick,
            self._ticks - self._staging_started_tick,
            100.0 * self.pass_coverage,
        )

    def _decode_or_skip(self, source_id: int, path: Path) -> _DecodedSegment | None:
        try:
            motion = MotrixMotion(path)
            identity = motion_corpus_contract(motion, fps=self._fps, extension_channels=_SONIC_EXTENSION_CHANNELS)
            check_motion_matches_corpus(motion, identity, self._corpus_contract)
            return _DecodedSegment(
                source_id,
                WbtMotionClip.create(
                    motion,
                    self._joint_names,
                    self._tracked_body_names,
                    self._reference_body_name,
                    self._root_body_name,
                    _SONIC_EXTENSION_CHANNELS,
                ),
            )
        except Exception as error:  # noqa: BLE001 - one corrupt file must not kill training
            logger.warning("Skipping unreadable motion file %s: %s", path, error)
            return None


def _allocate_sonic_clip(capacity: int, first_window: list[_DecodedSegment], track_sources: bool) -> SonicMotionClip:
    """Allocate capacity-sized clip arrays with window 0 written at frame 0."""
    if track_sources:
        fields, frame_clip_end, extensions, source_id, source_start = _assemble_window(first_window, True)
    else:
        fields, frame_clip_end, extensions = assemble_segment_arrays(
            [segment.clip for segment in first_window], _SONIC_EXTENSION_CHANNELS
        )
    first_frames = int(frame_clip_end.shape[0])
    capacity_fields = {
        field: np.zeros((capacity, *array.shape[1:]), dtype=array.dtype) for field, array in fields.items()
    }
    for field, array in fields.items():
        capacity_fields[field][:first_frames] = array
    capacity_clip_end = np.zeros(capacity, dtype=np.int64)
    capacity_clip_end[:first_frames] = frame_clip_end
    capacity_source_id = np.full(capacity, -1, dtype=np.int64) if track_sources else np.empty(0, dtype=np.int64)
    capacity_source_start = np.zeros(capacity, dtype=np.int64) if track_sources else np.empty(0, dtype=np.int64)
    if track_sources:
        capacity_source_id[:first_frames] = source_id
        capacity_source_start[:first_frames] = source_start
    smpl_joints = np.zeros((capacity, *extensions["smpl_joints"].shape[1:]), dtype=np.float32)
    smpl_root_quat = np.zeros((capacity, *extensions["smpl_root_quat"].shape[1:]), dtype=np.float32)
    smpl_joints[:first_frames] = extensions["smpl_joints"]
    smpl_root_quat[:first_frames] = extensions["smpl_root_quat"]
    return SonicMotionClip(
        **capacity_fields,
        frame_clip_end=capacity_clip_end,
        extensions=Map({}),
        smpl_joints=smpl_joints,
        smpl_root_quat=smpl_root_quat,
        frame_source_id=capacity_source_id,
        frame_source_start=capacity_source_start,
    )


__all__ = ["CorpusRotator"]

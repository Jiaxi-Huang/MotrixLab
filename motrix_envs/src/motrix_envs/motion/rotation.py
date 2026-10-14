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
    ) -> None:
        self._entries = entries
        self._order = order
        self._rng = rng
        self._cursor = 0
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

        self._pool: ThreadPoolExecutor | None = None
        self._decode_futures: list[Future[WbtMotionClip | None]] = []
        self._assemble_future: Future[tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]] | None = None
        self._pending_clips = 0
        self._carry: list[WbtMotionClip] = []
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
            return _allocate_sonic_clip(frames, segments), None
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
        clip = _allocate_sonic_clip(2 * frame_budget, segments)
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
        self._decode_futures = [self._pool.submit(self._decode_or_skip, path) for path in batch]
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
        kept: list[WbtMotionClip] = []
        used = 0
        for index, segment in enumerate(segments):
            frames = int(segment.joint_pos.shape[0])
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
        self._assemble_future = self._pool.submit(assemble_segment_arrays, kept, _SONIC_EXTENSION_CHANNELS)
        self._staging_started_tick = self._ticks
        self._phase = _STAGING

    def _skip_cycle(self, reason: str) -> None:
        logger.warning("Motion rotation window %d skipped (%s); retrying next period.", self._window_index + 1, reason)
        self._phase = _IDLE
        self._last_swap_tick = self._ticks

    def _next_entry(self) -> Path:
        entry = self._entries[int(self._order[self._cursor])]
        self._cursor += 1
        if self._cursor >= len(self._order):
            # One pass over the corpus is complete; reshuffle and start the
            # next pass from the continuing seeded stream.
            self._order = self._rng.permutation(len(self._entries))
            self._cursor = 0
        return entry

    def _retired_half_drained(self, steps: np.ndarray) -> bool:
        target = 1 - self._active_half
        previous = self._half_frames[target]
        if previous == 0:
            return True
        base = target * self._frame_budget
        return not bool(np.any((steps >= base) & (steps < base + previous)))

    def _publish(self, term, staging) -> None:
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
        for name in _SONIC_EXTENSION_CHANNELS:
            getattr(clip, name)[base : base + frames] = extensions[name]
        term.window_range[:] = (base, frames)
        num_bins = int(term.adaptive_bin_failed_count.size)
        if num_bins:
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
        logger.info(
            "Motion rotation: window %d published (%d clips, %d frames at base %d; "
            "decode+stage %d ticks, drain %d ticks).",
            self._window_index,
            self._pending_clips,
            frames,
            base,
            self._staging_started_tick - self._decode_started_tick,
            self._ticks - self._staging_started_tick,
        )

    def _decode_or_skip(self, path: Path) -> WbtMotionClip | None:
        try:
            motion = MotrixMotion(path)
            identity = motion_corpus_contract(motion, fps=self._fps, extension_channels=_SONIC_EXTENSION_CHANNELS)
            check_motion_matches_corpus(motion, identity, self._corpus_contract)
            return WbtMotionClip.create(
                motion,
                self._joint_names,
                self._tracked_body_names,
                self._reference_body_name,
                self._root_body_name,
                _SONIC_EXTENSION_CHANNELS,
            )
        except Exception as error:  # noqa: BLE001 - one corrupt file must not kill training
            logger.warning("Skipping unreadable motion file %s: %s", path, error)
            return None


def _allocate_sonic_clip(capacity: int, first_window: list[WbtMotionClip]) -> SonicMotionClip:
    """Allocate capacity-sized clip arrays with window 0 written at frame 0."""
    fields, frame_clip_end, extensions = assemble_segment_arrays(first_window, _SONIC_EXTENSION_CHANNELS)
    first_frames = int(frame_clip_end.shape[0])
    capacity_fields = {
        field: np.zeros((capacity, *array.shape[1:]), dtype=array.dtype) for field, array in fields.items()
    }
    for field, array in fields.items():
        capacity_fields[field][:first_frames] = array
    capacity_clip_end = np.zeros(capacity, dtype=np.int64)
    capacity_clip_end[:first_frames] = frame_clip_end
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
    )


__all__ = ["CorpusRotator"]

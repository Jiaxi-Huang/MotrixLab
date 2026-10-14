# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotionLibrary — assemble multiple MotrixLab motion NPZ v1 files into one clip.

Each file holds one clip (schema v1). The library reorders every clip to the
task joint/body order, loads only the task-declared ``ext_`` channels, validates
the corpus-level contracts, and concatenates the clips onto one global frame
axis with per-frame clip boundaries. A single file is the N=1 special case.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from motrix_env_core.numba.kernel_data import Map
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.tracked import MotionChannel, WbtMotionClip

# Per-frame array fields of :class:`WbtMotionClip` that concatenate along the
# frame axis when clips are joined.
FRAME_AXIS_FIELDS = (
    "joint_pos",
    "joint_vel",
    "tracked_bodies_pos_w",
    "tracked_bodies_quat_w",
    "tracked_bodies_lin_vel_w",
    "tracked_bodies_ang_vel_w",
    "root_body_pos_w",
    "root_body_quat_w",
    "root_body_lin_vel_w",
    "root_body_ang_vel_w",
    "reference_body_pos_w",
    "reference_body_quat_w",
)

_REQUIRED_FRAME_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class MotionLibrary:
    """Load an ordered corpus of motion NPZ v1 files and assemble one WBT clip.

    Entries may be files or directories; a directory expands to its ``.npz``
    files in sorted order, so the corpus order is fixed and reproducible.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        joint_names: list[str],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
        fps: int,
        extension_channels: tuple[str, ...] = (),
        num_motions: int | None = None,
        sample_seed: int = 0,
    ) -> None:
        """Configure the corpus.

        Args:
            paths: Ordered motion files and/or directories.
            joint_names: Desired output joint order (task order).
            tracked_body_names: Desired tracked-body order.
            reference_body_name: Body used to align the motion to the robot.
            root_body_name: Floating-base root body.
            fps: Control rate every clip must be recorded at.
            extension_channels: Names (without the ``ext_`` prefix) of channels
                to load from every file; undeclared channels are not loaded.
            num_motions: Randomly subsample the expanded corpus to this many
                clips before loading; ``None`` loads everything. The draw is
                deterministic for a fixed ``sample_seed``.
            sample_seed: Seed driving the ``num_motions`` subsample.
        """
        entries = expand_motion_paths(paths)
        self.entries = subsample_motion_entries(entries, num_motions, sample_seed)
        self.joint_names = joint_names
        self.tracked_body_names = tracked_body_names
        self.reference_body_name = reference_body_name
        self.root_body_name = root_body_name
        self.fps = fps
        self.extension_channels = tuple(extension_channels)

    def assemble(self) -> WbtMotionClip:
        """Load, validate, and concatenate the corpus into one kernel-data clip."""
        segments = load_motion_segments(
            self.entries,
            joint_names=self.joint_names,
            tracked_body_names=self.tracked_body_names,
            reference_body_name=self.reference_body_name,
            root_body_name=self.root_body_name,
            fps=self.fps,
            extension_channels=self.extension_channels,
        )
        return assemble_segment_clip(segments, self.extension_channels)


def load_motion_segments(
    entries: Sequence[Path],
    *,
    joint_names: list[str],
    tracked_body_names: tuple[str, ...],
    reference_body_name: str,
    root_body_name: str,
    fps: int,
    extension_channels: tuple[str, ...] = (),
) -> list[WbtMotionClip]:
    """Decode and validate corpus entries into per-file task-order segments.

    Each entry is one schema v1 clip reindexed to the task joint/body order;
    the corpus-level contracts (fps, name sets, finiteness, extension-channel
    shapes) are enforced across the batch exactly like a full
    :class:`MotionLibrary` load.
    """
    motions = [MotrixMotion(path) for path in entries]
    reference: tuple[frozenset[str], frozenset[str], tuple[tuple[int, ...], ...], Path] | None = None
    for motion in motions:
        identity = motion_corpus_contract(motion, fps=fps, extension_channels=extension_channels)
        if reference is None:
            reference = identity
        else:
            check_motion_matches_corpus(motion, identity, reference)
    return [
        WbtMotionClip.create(
            motion,
            joint_names,
            tracked_body_names,
            reference_body_name,
            root_body_name,
            extension_channels,
        )
        for motion in motions
    ]


def motion_corpus_contract(
    motion: MotrixMotion,
    *,
    fps: int,
    extension_channels: tuple[str, ...] = (),
) -> tuple[frozenset[str], frozenset[str], tuple[tuple[int, ...], ...], Path]:
    """Validate one motion against the per-file corpus contract.

    Returns the motion's identity tuple — (joint-name set, body-name set,
    extension-channel trailing shapes, source path) — of which corpus members
    must agree on the first three.
    """
    if motion.fps != fps:
        raise ValueError(f"Motion fps must match the control rate {fps}: {motion.path} is recorded at {motion.fps}.")
    for field in _REQUIRED_FRAME_ARRAYS:
        if not np.all(np.isfinite(getattr(motion, field))):
            raise ValueError(f"{field} in {motion.path} contains non-finite values.")
    shapes = []
    for name in extension_channels:
        if name not in motion.extensions:
            raise ValueError(f"Motion file {motion.path} does not provide declared extension channel 'ext_{name}'.")
        shapes.append(motion.extensions[name].shape[1:])
    return frozenset(motion.joint_names), frozenset(motion.body_names), tuple(shapes), motion.path


def check_motion_matches_corpus(
    motion: MotrixMotion,
    identity: tuple[frozenset[str], frozenset[str], tuple[tuple[int, ...], ...], Path],
    reference: tuple[frozenset[str], frozenset[str], tuple[tuple[int, ...], ...], Path],
) -> None:
    """Reject a decoded motion whose identity differs from the corpus reference."""
    if identity[0] != reference[0]:
        raise ValueError(f"Joint name set of {motion.path} differs from {reference[3]}.")
    if identity[1] != reference[1]:
        raise ValueError(f"Body name set of {motion.path} differs from {reference[3]}.")
    if identity[2] != reference[2]:
        raise ValueError(
            f"Extension channel must keep one shape across the corpus; {motion.path} has trailing "
            f"shapes {identity[2]}, reference {reference[3]} has {reference[2]}."
        )


def assemble_segment_clip(segments: list[WbtMotionClip], extension_channels: tuple[str, ...]) -> WbtMotionClip:
    """Concatenate per-file segments into one kernel-data clip."""
    fields, frame_clip_end, extensions = assemble_segment_arrays(segments, extension_channels)
    return WbtMotionClip(
        **fields,
        frame_clip_end=frame_clip_end,
        extensions=Map({name: MotionChannel(data) for name, data in extensions.items()}),
    )


def assemble_segment_arrays(
    segments: list[WbtMotionClip],
    extension_channels: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Concatenate segments into plain frame-axis arrays starting at frame 0.

    Returns the per-field arrays (keys match :data:`FRAME_AXIS_FIELDS`), the
    per-frame local clip-end boundaries, and the concatenated extension
    channels. ``frame_clip_end`` values are local to the returned assembly; a
    caller placing it at a nonzero base offset must add the base itself.
    """
    clip_lengths = np.asarray([segment.joint_pos.shape[0] for segment in segments], dtype=np.int64)
    clip_offsets = np.zeros_like(clip_lengths)
    np.cumsum(clip_lengths[:-1], out=clip_offsets[1:])
    fields = {field: np.concatenate([getattr(segment, field) for segment in segments]) for field in FRAME_AXIS_FIELDS}
    frame_clip_end = np.concatenate(
        [segment.frame_clip_end + offset for segment, offset in zip(segments, clip_offsets)]
    )
    extensions = {
        name: np.concatenate([segment.extensions[name].data for segment in segments]) for name in extension_channels
    }
    return fields, frame_clip_end, extensions


def expand_motion_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand motion sources to an ordered tuple of ``.npz`` file paths.

    Directory entries expand to their ``.npz`` children in sorted order, so
    the corpus order of a directory source is fixed and reproducible.
    """
    entries: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            entries.extend(sorted(child for child in path.iterdir() if child.suffix == ".npz"))
        else:
            entries.append(path)
    if not entries:
        raise ValueError("No motion files were found in the given sources.")
    return tuple(entries)


def subsample_motion_entries(entries: tuple[Path, ...], num_motions: int | None, sample_seed: int) -> tuple[Path, ...]:
    """Deterministically subsample the expanded corpus to ``num_motions`` clips.

    The selected files keep the corpus's sorted order, so the assembled clip
    and its ``frame_clip_end`` layout stay reproducible run to run.
    """
    if num_motions is None or num_motions >= len(entries):
        return entries
    if num_motions <= 0:
        raise ValueError(f"num_motions must be a positive integer, got {num_motions}.")
    picked = np.random.default_rng(sample_seed).choice(len(entries), size=num_motions, replace=False)
    return tuple(entries[i] for i in sorted(picked))


__all__ = [
    "FRAME_AXIS_FIELDS",
    "MotionLibrary",
    "assemble_segment_arrays",
    "assemble_segment_clip",
    "check_motion_matches_corpus",
    "expand_motion_paths",
    "load_motion_segments",
    "motion_corpus_contract",
    "subsample_motion_entries",
]

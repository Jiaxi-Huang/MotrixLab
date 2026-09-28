# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SONIC motion clip assembly over MotrixLab motion NPZ v1 files.

SONIC trains from a corpus of schema v1 npz clips carrying the
``ext_smpl_joints`` / ``ext_smpl_root_quat`` reference channels (produced by
the BONES-SEED and LAFAN converters). :class:`SonicMotionClip` is the
kernel-facing view of that corpus: the multi-clip :class:`WbtMotionClip` data
plane plus the two SMPL reference arrays as dedicated fields the observation
kernels index directly. A corpus is the only motion source — a single clip is
the one-file special case of the same path.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from motrix_env_core.numba.kernel_data import Map, SharedArray, kernel_data
from motrix_envs.motion.library import MotionLibrary
from motrix_envs.motion.tracked import WbtMotionClip


@kernel_data
class SonicMotionClip(WbtMotionClip):
    """Kernel-compatible WBT clip with SONIC's SMPL reference channels.

    ``frame_clip_end`` is inherited from the multi-clip :class:`WbtMotionClip`;
    the SMPL arrays stay dedicated SONIC fields consumed directly by the
    observation kernels.
    """

    smpl_joints: SharedArray
    smpl_root_quat: SharedArray

    @classmethod
    def from_corpus(
        cls,
        paths: Sequence[str | Path],
        *,
        joint_names: tuple[str, ...],
        tracked_body_names: tuple[str, ...],
        reference_body_name: str,
        root_body_name: str,
    ) -> SonicMotionClip:
        """Assemble a SONIC corpus from MotrixLab motion NPZ v1 files.

        Each file is one clip (schema v1) carrying the ``ext_smpl_joints`` and
        ``ext_smpl_root_quat`` reference channels; :class:`MotionLibrary`
        reorders, validates, and concatenates them onto one global frame axis.
        """
        library = MotionLibrary(
            paths,
            joint_names=list(joint_names),
            tracked_body_names=tracked_body_names,
            reference_body_name=reference_body_name,
            root_body_name=root_body_name,
            fps=50,
            extension_channels=("smpl_joints", "smpl_root_quat"),
        )
        base = library.assemble()
        smpl_joints = base.extensions["smpl_joints"].data
        smpl_root_quat = base.extensions["smpl_root_quat"].data
        if smpl_joints.shape[1:] != (24, 3):
            raise ValueError(f"ext_smpl_joints must have trailing shape (24, 3), got {smpl_joints.shape[1:]}")
        if smpl_root_quat.shape[1:] != (4,):
            raise ValueError(f"ext_smpl_root_quat must have trailing shape (4,), got {smpl_root_quat.shape[1:]}")
        return cls(
            joint_pos=base.joint_pos,
            joint_vel=base.joint_vel,
            tracked_bodies_pos_w=base.tracked_bodies_pos_w,
            tracked_bodies_quat_w=base.tracked_bodies_quat_w,
            tracked_bodies_lin_vel_w=base.tracked_bodies_lin_vel_w,
            tracked_bodies_ang_vel_w=base.tracked_bodies_ang_vel_w,
            root_body_pos_w=base.root_body_pos_w,
            root_body_quat_w=base.root_body_quat_w,
            root_body_lin_vel_w=base.root_body_lin_vel_w,
            root_body_ang_vel_w=base.root_body_ang_vel_w,
            reference_body_pos_w=base.reference_body_pos_w,
            reference_body_quat_w=base.reference_body_quat_w,
            frame_clip_end=base.frame_clip_end,
            extensions=Map({}),
            smpl_joints=smpl_joints,
            smpl_root_quat=smpl_root_quat,
        )


__all__ = ["SonicMotionClip"]

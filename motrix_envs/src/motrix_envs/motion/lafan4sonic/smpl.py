# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Materialize SMPL pose parameters into the SONIC reference channels.

Forward-kinematics the 55-joint SMPL hierarchy (rest offsets and parents from
gear_sonic's ``human_joints_info.pkl``) from the pose parameters produced by
``bvh.py``, then move everything into the SONIC SMPL frame contract — the same
contract the BONES-SEED SMPL PKLs already carry:

- ``smpl_joints``: 24 joints (SMPL 0..21 plus both hand joints 39/54),
  **local to the SMPL root joint** (root position subtracted);
- ``smpl_root_quat``: world orientation of the SMPL root, as
  ``heading_correction ∘ y_up→z_up ∘ root ∘ canonical`` (xyzw).

The LAFAN-specific ``heading_correction`` (a +90° rotation about z) aligns the
BVH-derived SMPL world frame with the retargeted G1 frame, which is rotated
90° apart. It is a world-frame transform, so it must apply to both the joints
and the root orientation — touching only one would corrupt
``root⁻¹ · joints``, the exact feature the SONIC SMPL encoder consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motrix_env_core.math import quaternion
from motrix_envs.motion.lafan4sonic._math import axis_angle_to_quat_xyzw

_NUM_SMPL_JOINTS = 55
_NUM_SMPL_BODY_JOINTS = 21
# SONIC's 24-joint selection from the 55-joint hierarchy: the 22 body joints
# plus the two hand roots (39 = left, 54 = right).
SMPL_24_JOINTS = tuple(range(22)) + (39, 54)

# y-up → z-up (rotation about x by +90°), xyzw.
_Y_UP_TO_Z_UP = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float64)
# LAFAN BVH→SMPL heading correction: +90° about z, xyzw.
_HEADING_CORRECTION = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float64)
# Fixed canonicalization rotation, xyzw (same constant the BONES-SEED
# converter composes onto pose_aa[:, 0]).
_CANONICAL_ROTATION = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float64)


@dataclass(frozen=True)
class SmplSkeleton:
    """Rest pose of the 55-joint SMPL hierarchy (y-up)."""

    rest_joints: np.ndarray  # (55, 3)
    parents: np.ndarray  # (55,) intp; parents[0] == -1

    def __post_init__(self) -> None:
        if self.rest_joints.shape != (_NUM_SMPL_JOINTS, 3) or self.parents.shape != (_NUM_SMPL_JOINTS,):
            raise ValueError(
                f"SMPL skeleton must carry ({_NUM_SMPL_JOINTS}, 3) rest joints and "
                f"({_NUM_SMPL_JOINTS},) parents, got {self.rest_joints.shape} / {self.parents.shape}"
            )


def load_smpl_skeleton(info_path: str | Path) -> SmplSkeleton:
    """Load gear_sonic's ``human_joints_info.pkl`` (J + parents_list).

    The pickle is a torch-serialized file; torch is imported lazily here since
    it is only needed on this cold path (SONIC training pulls it in anyway).
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Loading human_joints_info.pkl requires torch (part of the SONIC training stack)") from error
    info = torch.load(str(info_path), weights_only=False, map_location="cpu")
    rest = np.asarray(info["J"], dtype=np.float64)
    parents = np.asarray(info["parents_list"], dtype=np.intp)
    return SmplSkeleton(rest_joints=rest, parents=parents)


def materialize_smpl_channels(
    root_rotvec: np.ndarray, body_rotvec: np.ndarray, skeleton: SmplSkeleton
) -> tuple[np.ndarray, np.ndarray]:
    """SMPL pose parameters -> SONIC ``(smpl_joints, smpl_root_quat)`` channels.

    ``root_rotvec`` is ``(N, 3)`` and ``body_rotvec`` is ``(N, 21, 3)`` in the
    y-up SMPL rest frame (see :func:`motrix_envs.motion.lafan4sonic.bvh.smpl_pose_from_bvh`).
    Output dtypes are float32; joints are root-local, the root quat is the
    world SONIC SMPL root orientation (xyzw).
    """
    root_rotvec = np.asarray(root_rotvec, dtype=np.float64)
    body_rotvec = np.asarray(body_rotvec, dtype=np.float64)
    if root_rotvec.ndim != 2 or root_rotvec.shape[1] != 3:
        raise ValueError(f"root_rotvec must have shape (N, 3), got {root_rotvec.shape}")
    frames = len(root_rotvec)
    if body_rotvec.shape != (frames, _NUM_SMPL_BODY_JOINTS, 3):
        raise ValueError(f"body_rotvec must have shape ({frames}, 21, 3), got {body_rotvec.shape}")

    local_quat = np.zeros((frames, _NUM_SMPL_JOINTS, 4), dtype=np.float64)
    local_quat[..., 3] = 1.0
    local_quat[:, 0] = axis_angle_to_quat_xyzw(root_rotvec)
    local_quat[:, 1 : 1 + _NUM_SMPL_BODY_JOINTS] = axis_angle_to_quat_xyzw(body_rotvec)

    rest = skeleton.rest_joints
    parents = skeleton.parents
    rel = np.zeros_like(rest)
    rel[1:] = rest[1:] - rest[parents[1:]]
    global_quat = np.empty_like(local_quat)
    joints = np.empty((frames, _NUM_SMPL_JOINTS, 3), dtype=np.float64)
    global_quat[:, 0] = local_quat[:, 0]
    joints[:, 0] = rest[0]
    for joint in range(1, _NUM_SMPL_JOINTS):
        parent = parents[joint]
        global_quat[:, joint] = quaternion.mul(global_quat[:, parent], local_quat[:, joint])
        joints[:, joint] = joints[:, parent] + quaternion.rotate_vector(global_quat[:, parent], rel[joint])

    selection = np.asarray(SMPL_24_JOINTS, dtype=np.intp)
    joints = joints[:, selection]
    frame = quaternion.mul(_HEADING_CORRECTION, _Y_UP_TO_Z_UP)
    joints = quaternion.rotate_vector(frame, joints)
    root_quat = quaternion.mul(
        frame,
        quaternion.mul(local_quat[:, 0], _CANONICAL_ROTATION),
    )
    # SONIC stores the joints local to the SMPL root joint.
    joints = joints - joints[:, :1, :]
    return joints.astype(np.float32), root_quat.astype(np.float32)

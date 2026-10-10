# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""LAFAN1 BVH parsing and the BVH → SMPL pose pathway.

Extracts SMPL pose parameters (root + 21 body rotation vectors, in the y-up
SMPL rest frame) from raw LAFAN1 BVH rotations. The mapping and the per-bone
rest-orientation corrections are adapted from ``jaraujo98/lafan_to_smplx``
revision d465aa7202ddc94f2cd557b682854437573e0dca (MIT, Copyright 2025 Joao
Pedro Araujo) via the UniLab port; quaternions here are ``xyzw`` throughout.

The BVH root translation is deliberately discarded: SONIC stores SMPL joints
local to the SMPL root (see ``smpl.py``), and the paired G1 retarget CSV owns
the world root trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motrix_env_core.math import quaternion
from motrix_envs.motion.lafan4sonic._math import quat_to_rotvec, slerp_track

# SMPL body joints 1..21 -> LAFAN bone names.
SMPL_TO_LAFAN: dict[int, str] = {
    1: "LeftUpLeg",
    2: "RightUpLeg",
    3: "Spine",
    4: "LeftLeg",
    5: "RightLeg",
    6: "Spine1",
    7: "LeftFoot",
    8: "RightFoot",
    9: "Spine2",
    10: "LeftToe",
    11: "RightToe",
    12: "Neck",
    13: "LeftShoulder",
    14: "RightShoulder",
    15: "Head",
    16: "LeftArm",
    17: "RightArm",
    18: "LeftForeArm",
    19: "RightForeArm",
    20: "LeftHand",
    21: "RightHand",
}


@dataclass(frozen=True)
class BvhMotion:
    """Parsed LAFAN BVH skeleton + per-frame local joint rotations."""

    names: tuple[str, ...]
    parents: np.ndarray  # (J,) int32; parents[0] == -1
    local_quat_xyzw: np.ndarray  # (T, J, 4) unit xyzw
    fps: float


def _axis_quat(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Unit xyzw quaternion of a fixed-axis rotation (angle in radians)."""
    quat = quaternion.from_angle_axis(angle, np.asarray(axis, dtype=np.float64))
    return np.asarray(quat, dtype=np.float64)


def lafan_frame_offsets() -> dict[str, np.ndarray]:
    """Rest-orientation corrections mapping LAFAN bone frames onto SMPL.

    Each entry is composed as ``Rz(a) @ Ry(b)`` (or the single-axis form)
    exactly as in the lafan_to_smplx reference, expressed as xyzw quaternions.
    """
    return {
        "Hips": quaternion.mul(_axis_quat((0.0, 0.0, 1.0), -np.pi / 2), _axis_quat((0.0, 1.0, 0.0), -np.pi / 2)),
        **{
            name: quaternion.mul(_axis_quat((0.0, 0.0, 1.0), np.pi / 2), _axis_quat((0.0, 1.0, 0.0), np.pi / 2))
            for name in ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg")
        },
        **{
            name: quaternion.mul(
                _axis_quat((0.0, 0.0, 1.0), 0.37117860986509),
                _axis_quat((0.0, 1.0, 0.0), np.pi / 2),
            )
            for name in ("LeftFoot", "RightFoot")
        },
        **{name: _axis_quat((0.0, 1.0, 0.0), np.pi / 2) for name in ("LeftToe", "RightToe")},
        **{
            name: quaternion.mul(_axis_quat((0.0, 0.0, 1.0), -np.pi / 2), _axis_quat((0.0, 1.0, 0.0), -np.pi / 2))
            for name in ("Spine", "Spine1", "Spine2", "Neck", "Head")
        },
        **{
            name: _axis_quat((1.0, 0.0, 0.0), np.pi / 2)
            for name in ("LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand")
        },
        **{
            name: quaternion.mul(_axis_quat((0.0, 0.0, 1.0), np.pi), _axis_quat((1.0, 0.0, 0.0), -np.pi / 2))
            for name in ("RightShoulder", "RightArm", "RightForeArm", "RightHand")
        },
    }


def _euler_zyx_to_quat_xyzw(euler_zyx_deg: np.ndarray) -> np.ndarray:
    """(..., 3) [Z, Y, X] euler degrees -> (..., 4) xyzw (R = Rz @ Ry @ Rx)."""
    z, y, x = (np.deg2rad(euler_zyx_deg[..., i]) for i in range(3))
    qz = np.zeros(euler_zyx_deg.shape[:-1] + (4,), dtype=np.float64)
    qz[..., 2] = np.sin(0.5 * z)
    qz[..., 3] = np.cos(0.5 * z)
    qy = np.zeros_like(qz)
    qy[..., 1] = np.sin(0.5 * y)
    qy[..., 3] = np.cos(0.5 * y)
    qx = np.zeros_like(qz)
    qx[..., 0] = np.sin(0.5 * x)
    qx[..., 3] = np.cos(0.5 * x)
    return quaternion.mul(quaternion.mul(qz, qy), qx)


def load_bvh(path: str | Path) -> BvhMotion:
    """Parse a LAFAN1 BVH file (Z/Y/X rotation channels, 3 root position channels)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    names: list[str] = []
    parents: list[int] = []
    active = -1
    end_site = False
    rotation_orders: list[tuple[str, ...]] = []
    frames_line = frame_time_line = -1
    num_frames = 0
    frame_time = 0.0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("ROOT ", "JOINT ")):
            names.append(stripped.split()[1])
            parents.append(active)
            active = len(names) - 1
        elif stripped == "End Site":
            end_site = True
        elif stripped.startswith("CHANNELS "):
            parts = stripped.split()
            channels = tuple(parts[2 : 2 + int(parts[1])])
            rotations = tuple(channel for channel in channels if channel.endswith("rotation"))
            if rotations != ("Zrotation", "Yrotation", "Xrotation"):
                raise ValueError(f"Unsupported BVH rotation channels in {path}: {rotations}")
            rotation_orders.append(rotations)
        elif stripped == "}":
            if end_site:
                end_site = False
            elif active >= 0:
                active = parents[active]
        elif stripped.startswith("Frames:"):
            frames_line = index
            num_frames = int(stripped.split(":", 1)[1])
        elif stripped.startswith("Frame Time:"):
            frame_time_line = index
            frame_time = float(stripped.split(":", 1)[1])
            break
    if len(rotation_orders) != len(names) or frames_line < 0 or frame_time_line < 0:
        raise ValueError(f"Malformed LAFAN BVH hierarchy or MOTION header in {path}")
    values = np.loadtxt(lines[frame_time_line + 1 :], dtype=np.float64, ndmin=2)
    expected_columns = 6 + 3 * (len(names) - 1)
    if values.shape != (num_frames, expected_columns) or not np.isfinite(values).all():
        raise ValueError(
            f"Invalid BVH motion shape {values.shape} in {path}; expected {(num_frames, expected_columns)}"
        )
    # Drop the three root position channels; the rest is one Z/Y/X euler triple
    # per joint in hierarchy order.
    euler_zyx = values[:, 3:].reshape(num_frames, len(names), 3)
    local_quat = _euler_zyx_to_quat_xyzw(euler_zyx)
    return BvhMotion(
        names=tuple(names),
        parents=np.asarray(parents, dtype=np.int32),
        local_quat_xyzw=local_quat,
        fps=1.0 / frame_time,
    )


def _global_quaternions(local: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Chain local quats (T, J, 4) through the skeleton to world rotations."""
    result = np.empty_like(local)
    result[:, 0] = local[:, 0]
    for joint in range(1, local.shape[1]):
        result[:, joint] = quaternion.mul(result[:, parents[joint]], local[:, joint])
    return result


def smpl_pose_from_bvh(motion: BvhMotion, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample a BVH and extract SMPL pose parameters on ``target_times``.

    Returns ``(root_rotvec (N, 3), body_rotvec (N, 21, 3))`` — SMPL joint 0 and
    joints 1..21 as rotation vectors in the y-up SMPL rest frame, ready for
    :func:`motrix_envs.motion.lafan4sonic.smpl.materialize_smpl_channels`.
    """
    source_times = np.arange(motion.local_quat_xyzw.shape[0], dtype=np.float64) / motion.fps
    local = slerp_track(motion.local_quat_xyzw, source_times, target_times)
    global_quat = _global_quaternions(local, motion.parents)
    indices = {name: index for index, name in enumerate(motion.names)}
    missing = sorted((set(SMPL_TO_LAFAN.values()) | {"Hips"}) - set(indices))
    if missing:
        raise ValueError(f"LAFAN BVH is missing required joints: {missing}")
    offsets = lafan_frame_offsets()
    corrected = {
        name: quaternion.mul(global_quat[:, index], quat)
        for name, index in indices.items()
        if (quat := offsets.get(name)) is not None
    }
    root = corrected["Hips"]
    body_pose = np.zeros((len(target_times), 21, 3), dtype=np.float64)
    for smpl_index, name in SMPL_TO_LAFAN.items():
        parent_name = motion.names[motion.parents[indices[name]]]
        body_pose[:, smpl_index - 1] = quat_to_rotvec(
            quaternion.mul(quaternion.conjugate(corrected[parent_name]), corrected[name])
        )
    return quat_to_rotvec(root), body_pose

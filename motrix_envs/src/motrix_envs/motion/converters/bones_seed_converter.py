# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert BONES-SEED G1 CSV + SMPL PKL pairs to MotrixLab motion NPZ v1.

Sources:
    - G1 retargeted clips: ``bones-studio/seed`` on HuggingFace (``g1.tar.gz``,
      extracted CSVs)
    - SMPL reference clips: ``nvidia/GEAR-SONIC`` on HuggingFace
      (``bones_seed_smpl`` split tar, extracted PKLs)
Both sides are matched by file stem; ``scripts/motion/download_bone_seed.py``
materializes such a paired raw subset.

BONES-SEED CSV layout (header row + rows at 120 fps, 36 columns):
    [0]     Frame index
    [1:4]   root_translateX/Y/Z   centimeters
    [4:7]   root_rotateX/Y/Z      degrees, XYZ euler
    [7:36]  29 ``*_joint_dof``    degrees, columns named after the G1 joints

Every 4th row is kept (120 Hz -> 30 Hz), then the clip is resampled to the
50 Hz SONIC control rate (lerp on positions/joints, slerp on the root quat).
Body poses/velocities come from batched forward kinematics on the G1 model, so
``body_names`` line up with the task's ``get_link(...)`` bindings — the same
route the LAFAN converter takes.

SMPL PKL layout (joblib): ``{"fps": 50, "pose_aa": (T, 24, 3), "smpl_joints":
(T, 24, 3)}``.  The root orientation channel is derived from ``pose_aa[:, 0]``
with the SONIC y-up→z-up and canonicalization rotations.

Output is one schema v1 npz per clip carrying the SONIC reference channels
``ext_smpl_joints`` (T, 24, 3) and ``ext_smpl_root_quat`` (T, 4, xyzw); the
SONIC command consumes such directories directly via ``SonicMotionClip.from_corpus``.
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import motrixsim as mtx
import numpy as np

from motrix_env_core.math import quaternion
from motrix_envs.motion.schema import SCHEMA_VERSION
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR

# BONES-SEED CSVs are recorded at 120 fps; the validated SONIC pipeline keeps
# every 4th row (30 Hz) before resampling to the 50 Hz control rate.
_BONES_CSV_FPS = 120.0
_BONES_CSV_STRIDE = 4
_SONIC_FPS = 50
# SONIC observations read 10 future reference frames; shorter clips cannot feed
# the command.
_MIN_FRAMES = 10
_G1_ROOT_BODY = "pelvis"
_SONIC_REFERENCE_BODY = "pelvis"

_ROOT_COLUMNS = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
)
_EXPECTED_JOINTS = 29


def _natural_sort_key(value: str | Path) -> list[int | str]:
    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def resolve_bones_seed_pairs(robot_root: str | Path, smpl_root: str | Path) -> list[tuple[Path, Path]]:
    """Match BONES-SEED CSV and SMPL PKL clips by file stem (sorted, natural order)."""
    robots = {path.stem: path for path in Path(robot_root).expanduser().rglob("*.csv")}
    humans = {path.stem: path for path in Path(smpl_root).expanduser().rglob("*.pkl")}
    if not robots or not humans:
        raise FileNotFoundError(
            f"No BONES-SEED sources found under {robot_root} / {smpl_root}; "
            "run scripts/motion/download_bone_seed.py first"
        )
    names = sorted(set(robots) & set(humans), key=lambda n: _natural_sort_key(robots[n].name))
    if not names:
        raise FileNotFoundError("No paired BONES-SEED CSV/SMPL clips share a stem")
    return [(robots[name], humans[name]) for name in names]


def _parse_csv_header(csv_path: Path) -> tuple[list[str], np.ndarray]:
    header = [part.strip() for part in csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")]
    if tuple(header[: len(_ROOT_COLUMNS)]) != _ROOT_COLUMNS:
        raise ValueError(f"Unexpected BONES-SEED root columns in {csv_path}: {header[:7]}")
    joints = [name.removesuffix("_dof") for name in header[len(_ROOT_COLUMNS) :]]
    if len(joints) != _EXPECTED_JOINTS or len(set(joints)) != len(joints):
        raise ValueError(f"{csv_path} must carry {_EXPECTED_JOINTS} unique joint columns, got {len(joints)}")
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
    if raw.shape[1] != len(header) or raw.shape[0] < 2:
        raise ValueError(f"Invalid BONES-SEED CSV shape {raw.shape} in {csv_path}")
    return joints, raw


def _slerp_track(quat: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Slerp a (T, 4) xyzw track onto ``target_times`` (indices from ``source_times``)."""
    upper = np.clip(np.searchsorted(source_times, target_times, side="right"), 1, len(source_times) - 1)
    lower = upper - 1
    q0 = quat[lower]
    q1 = quat[upper].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    fraction = ((target_times - source_times[lower]) / (source_times[upper] - source_times[lower]))[:, None]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe = sin_theta > 1.0e-7
    scale0 = np.where(safe, np.sin((1.0 - fraction) * theta) / np.where(safe, sin_theta, 1.0), 1.0 - fraction)
    scale1 = np.where(safe, np.sin(fraction * theta) / np.where(safe, sin_theta, 1.0), fraction)
    out = scale0 * q0 + scale1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def _resample(
    root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray, source_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(len(root_pos), dtype=np.float64) / source_fps
    target = np.arange(0.0, times[-1], 1.0 / target_fps, dtype=np.float64)
    pos = np.stack([np.interp(target, times, root_pos[:, axis]) for axis in range(3)], axis=-1)
    quat = _slerp_track(root_quat, times, target)
    joints = np.stack([np.interp(target, times, dof[:, axis]) for axis in range(dof.shape[1])], axis=-1)
    return pos, quat, joints


def _quat_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw unit quats -> (N, 3) rotation vectors (atan2 form, w >= 0)."""
    q = quat_xyzw.copy()
    negative = q[:, 3] < 0.0
    q[negative] *= -1.0
    xyz, w = q[:, :3], q[:, 3:4]
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half = np.arctan2(norms, w)
    axis = np.where(norms > 1.0e-12, xyz / np.where(norms > 1.0e-12, norms, 1.0), 0.0)
    return axis * (2.0 * half)


def _angular_velocity_world(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity of a (T, 4) xyzw sequence via shortest-arc central diffs."""
    result = quat_xyzw.copy()
    for i in range(1, len(result)):
        if float(np.dot(result[i - 1], result[i])) < 0.0:
            result[i] *= -1.0
    omega = np.zeros((len(result), 3), dtype=np.float64)
    if len(result) > 2:
        q_rel = quaternion.mul(result[2:], quaternion.conjugate(result[:-2]))
        omega[1:-1] = _quat_to_rotvec(q_rel) / (2.0 * dt)
        omega[0] = omega[1]
        omega[-1] = omega[-2]
    return omega


def _axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1.0e-8, np.sin(half) / np.where(angle > 1.0e-8, angle, 1.0), 0.5 - angle * angle / 48.0)
    return np.concatenate((axis_angle * scale, np.cos(half)), axis=-1)


def _load_smpl_pkl(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = joblib.load(pkl_path)
    if isinstance(payload, dict) and len(payload) == 1 and "smpl_joints" not in payload:
        nested = next(iter(payload.values()))
        if isinstance(nested, dict):
            payload = nested
    if not isinstance(payload, dict) or "smpl_joints" not in payload or "pose_aa" not in payload:
        raise ValueError(f"Unsupported BONES-SEED SMPL payload in {pkl_path}")
    if int(round(float(payload["fps"]))) != _SONIC_FPS:
        raise ValueError(f"Expected {_SONIC_FPS} Hz SMPL data in {pkl_path}")
    pose = np.asarray(payload["pose_aa"], dtype=np.float32).reshape(-1, 24, 3)
    joints = np.asarray(payload["smpl_joints"], dtype=np.float32)
    if joints.shape != (len(pose), 24, 3):
        raise ValueError(f"Expected SMPL joints {(len(pose), 24, 3)}, got {joints.shape}")
    root = _axis_angle_to_quat_xyzw(pose[:, 0].astype(np.float64))
    y_up_to_z_up = np.broadcast_to(np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]), root.shape)
    fixed = np.broadcast_to(np.array([-0.5, -0.5, -0.5, 0.5]), root.shape)
    root_quat = quaternion.mul(quaternion.mul(y_up_to_z_up, root), fixed)
    return joints, root_quat.astype(np.float32)


def _convert_pair(model, csv_path: Path, pkl_path: Path, output_path: Path) -> dict[str, object]:
    """Convert one paired BONES-SEED clip into a MotrixLab motion NPZ v1."""
    model_joints = [str(name) for name in model.joint_names]
    body_names = [str(name) for name in model.link_names]

    csv_joints, raw = _parse_csv_header(csv_path)
    if set(csv_joints) != set(model_joints):
        raise ValueError(f"BONES joint set in {csv_path} does not match the G1 model")

    subsampled = raw[::_BONES_CSV_STRIDE]
    root_pos = subsampled[:, 1:4] / 100.0
    euler = np.deg2rad(subsampled[:, 4:7])
    root_quat = quaternion.from_euler(euler[:, 0], euler[:, 1], euler[:, 2]).astype(np.float64)
    dof_csv = np.deg2rad(subsampled[:, 7:])
    root_pos, root_quat, dof_csv = _resample(
        root_pos,
        np.asarray(root_quat, dtype=np.float64),
        dof_csv,
        _BONES_CSV_FPS / _BONES_CSV_STRIDE,
        _SONIC_FPS,
    )
    csv_to_model = [csv_joints.index(name) for name in model_joints]
    dof = dof_csv[:, csv_to_model]

    smpl_joints, smpl_root_quat = _load_smpl_pkl(pkl_path)
    num_frames = min(len(dof), len(smpl_joints))
    if num_frames < _MIN_FRAMES:
        raise ValueError(f"Paired BONES-SEED clip {csv_path.stem!r} is too short: {num_frames} frames")

    dt = 1.0 / _SONIC_FPS
    joint_vel = np.gradient(dof, dt, axis=0)
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = _angular_velocity_world(np.asarray(root_quat, dtype=np.float64), dt)

    selection = slice(0, num_frames)
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    qvel = np.concatenate([root_lin_vel, root_ang_vel, joint_vel], axis=1).astype(np.float32)
    if qpos.shape[1] != model.num_dof_pos or qvel.shape[1] != model.num_dof_vel:
        raise ValueError(
            f"qpos/qvel width ({qpos.shape[1]}/{qvel.shape[1]}) != model dof ({model.num_dof_pos}/{model.num_dof_vel})"
        )

    data = mtx.SceneData(model, batch=[num_frames])
    data.set_dof_pos(qpos[selection], model)
    data.set_dof_vel(qvel[selection])
    model.forward_kinematic(data)
    poses = np.asarray(model.get_link_poses(data), dtype=np.float32)  # (T, B, 7) xyzw

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(_SONIC_FPS),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(model_joints),
        "body_names": np.asarray(body_names),
        "joint_pos": np.ascontiguousarray(dof[selection], dtype=np.float32),
        "joint_vel": np.ascontiguousarray(joint_vel[selection], dtype=np.float32),
        "body_pos_w": np.ascontiguousarray(poses[:, :, 0:3]),
        "body_quat_w": np.ascontiguousarray(poses[:, :, 3:7]),
        "body_lin_vel_w": np.asarray(model.get_link_linear_velocities(data), dtype=np.float32),
        "body_ang_vel_w": np.asarray(model.get_link_angular_velocities(data), dtype=np.float32),
        "root_body_name": np.asarray(_G1_ROOT_BODY),
        "reference_body_name": np.asarray(_SONIC_REFERENCE_BODY),
        "clip_name": np.asarray(csv_path.stem),
        "ext_smpl_joints": np.ascontiguousarray(smpl_joints[selection], dtype=np.float32),
        "ext_smpl_root_quat": np.ascontiguousarray(smpl_root_quat[selection], dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    return {
        "num_frames": num_frames,
        "num_joints": len(model_joints),
        "num_bodies": len(body_names),
        "output_path": str(output_path),
    }


def convert_bones_seed_dataset(
    robot_root: str | Path,
    smpl_root: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_clips: int | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert every paired clip under the raw roots into ``output_dir``.

    The FK model is loaded once and shared across clips; existing outputs are
    skipped unless ``overwrite``; a clip that fails conversion is recorded and
    the batch continues. The corpus directory is ready for
    ``SonicMotionClip.from_corpus`` once done.
    """
    pairs = resolve_bones_seed_pairs(robot_root, smpl_root)
    if max_clips is not None:
        pairs = pairs[:max_clips]
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        Path(model_file).expanduser() if model_file is not None else UNITREE_G1_ASSET_DIR / "scene_g1_29dof.xml"
    )
    model = mtx.load_model(str(model_path))
    converted: list[str] = []
    skipped: list[str] = []
    failed: dict[str, str] = {}
    for index, (csv_path, pkl_path) in enumerate(pairs, start=1):
        destination = output_dir / f"{csv_path.stem}.npz"
        if destination.exists() and not overwrite:
            skipped.append(csv_path.stem)
            continue
        try:
            _convert_pair(model, csv_path, pkl_path, destination)
            converted.append(csv_path.stem)
        except (ValueError, OSError) as error:
            print(f"  failed {csv_path.stem}: {error}")
            failed[csv_path.stem] = str(error)
            destination.unlink(missing_ok=True)
        if index % 200 == 0:
            print(f"  {index}/{len(pairs)} pairs processed ({len(failed)} failed)")
    if not converted and not skipped:
        raise RuntimeError(f"Every BONES-SEED clip failed to convert under {robot_root} / {smpl_root}")
    return {"converted": converted, "skipped": skipped, "failed": failed, "output_dir": str(output_dir)}

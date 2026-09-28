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
Body poses come from batched forward kinematics on the G1 model, so
``body_names`` line up with the task's ``get_link(...)`` bindings — the same
route the LAFAN converter takes.

Velocity contract (mirrors upstream gear_sonic):

- ``joint_vel`` is the forward difference of the 50 Hz joint positions with
  the final frame's difference duplicated.
- ``body_lin_vel_w``/``body_ang_vel_w`` are derived by differentiating the FK
  poses (central differences + shortest-arc quaternion diffs) and gaussian
  smoothing the result (sigma = 2 frames, edge-replicated) — the same
  construction as the upstream ``_compute_velocity``. Velocities are NOT read
  from the engine's link-velocity outputs: MotrixSim's free-joint qvel layout
  is ``[lin_world(3), ang_body_local(3), joint_vel(N)]``, so feeding a
  world-frame root angular velocity through ``set_dof_vel`` returns body
  angular velocities rotated by the root orientation.

SMPL PKL layout (joblib): ``{"fps": 50, "pose_aa": (T, 24, 3), "smpl_joints":
(T, 24, 3)}``.  The root orientation channel is derived from ``pose_aa[:, 0]``
with the SONIC y-up→z-up and canonicalization rotations.

Output is one schema v1 npz per clip carrying the SONIC reference channels
``ext_smpl_joints`` (T, 24, 3) and ``ext_smpl_root_quat`` (T, 4, xyzw); the
SONIC command consumes such directories directly via ``SonicMotionClip.from_corpus``.

Clip filtering: upstream gear_sonic drops clips whose filename matches a
keyword list (scene props the robot cannot reproduce, plus extreme
acrobatics) before training — see ``BONES_SEED_FILTER_KEYWORDS``. Conversion
applies the same filter by default. Note the raw cache layout names
(``robot_filtered/``/``smpl_filtered/``) only mean "stem-paired subset" from
``scripts/motion/download_bone_seed.py``; content filtering happens here.
"""

from __future__ import annotations

import multiprocessing
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import motrixsim as mtx
import numpy as np

from motrix_env_core.math import quaternion
from motrix_envs.motion.schema import SCHEMA_VERSION
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR

# Upstream gear_sonic filter list (data_process/filter_and_copy_bones_data.py):
# scene props the robot cannot reproduce and extreme acrobatics whose
# retargeted clips destabilize training. Matched case-insensitively as
# substrings of the clip stem.
BONES_SEED_FILTER_KEYWORDS: tuple[str, ...] = (
    "bed",
    "bike",
    "chair",
    "climb",
    "com_up_50cm",
    "sitting",
    "step_on",
    "seat",
    "table",
    "_sit_",
    "sit_",
    "ladder",
    "crutch",
    "_bed_",
    "_ride_",
    "scooter",
    "stepdown",
    "acrobatics_",
    "box_HSPU",
    "cartwheel",
    "50cm_box_",
    "on_box",
    "fall_from",
    "handstand_ff_",
    "on_1m",
    "form_box",
    "off_1m",
    "230m",
    "jump_over_obstacle_",
    "lift_crate_come_up_",
    "jump_to_shoulder_roll",
    "kozak_dance",
    "stair",
    "handstand",
    "box_jump",
    "monkey_jump",
    "safety_roll",
    "box_dips",
    "walking_on_edge",
    "push_obstacle",
)

# BONES-SEED CSVs are recorded at 120 fps; the validated SONIC pipeline keeps
# every 4th row (30 Hz) before resampling to the 50 Hz control rate.
_BONES_CSV_FPS = 120.0
_BONES_CSV_STRIDE = 4
_SONIC_FPS = 50
# Upstream smooths pose-derived body velocities with an edge-replicated
# gaussian filter (sigma = 2 frames) before storing them.
_VELOCITY_SMOOTH_SIGMA = 2.0
_VELOCITY_SMOOTH_TRUNCATE = 4.0
# SONIC observations read 10 future reference frames; shorter clips cannot feed
# the command.
_MIN_FRAMES = 10
_G1_ROOT_BODY = "pelvis"
_SONIC_REFERENCE_BODY = "pelvis"
# MotrixSim never returns a released SceneData's native buffers (~23 KB/frame),
# so batched FK must reuse one fixed-batch data object for the whole dataset
# instead of allocating one per clip — a fresh per-clip SceneData leaks linearly
# with total frames and OOM-kills long conversion runs.
_FK_CHUNK_FRAMES = 2048

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
    """(..., 4) xyzw unit quats -> (..., 3) rotation vectors (atan2 form, w >= 0)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz, w = q[..., :3], q[..., 3:4]
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half = np.arctan2(norms, w)
    axis = np.where(norms > 1.0e-12, xyz / np.where(norms > 1.0e-12, norms, 1.0), 0.0)
    return axis * (2.0 * half)


def _angular_velocity_world(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity of a (T, 4) or (T, B, 4) xyzw sequence via shortest-arc central diffs."""
    result = np.asarray(quat_xyzw, dtype=np.float64).copy()
    for i in range(1, len(result)):
        flip = np.sum(result[i - 1] * result[i], axis=-1) < 0.0
        result[i] = np.where(flip[..., None], -result[i], result[i])
    omega = np.zeros(result.shape[:-1] + (3,), dtype=np.float64)
    if len(result) > 2:
        q_rel = quaternion.mul(result[2:], quaternion.conjugate(result[:-2]))
        omega[1:-1] = _quat_to_rotvec(q_rel) / (2.0 * dt)
        omega[0] = omega[1]
        omega[-1] = omega[-2]
    return omega


def _gaussian_smooth1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with edge replication.

    Matches ``scipy.ndimage.gaussian_filter1d(..., mode="nearest")`` without
    the scipy dependency.
    """
    radius = int(_VELOCITY_SMOOTH_TRUNCATE * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)],
        axis=0,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=0)
    return windows @ kernel


def _forward_difference_track(values: np.ndarray, dt: float) -> np.ndarray:
    """Forward differences with the final frame duplicated (upstream dof-vel contract)."""
    velocity = np.empty(values.shape, dtype=np.float64)
    velocity[:-1] = (values[1:] - values[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def _body_velocities_from_poses(
    body_pos_w: np.ndarray, body_quat_w: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame body velocities by differentiating the FK poses.

    Mirrors the upstream gear_sonic ``_compute_velocity`` contract: central
    differences followed by an edge-replicated gaussian smoothing (sigma = 2
    frames), so the stored velocity channels stay consistent with the stored
    poses. Velocities are NOT read from the engine's link-velocity outputs —
    see the free-joint qvel note in the module docstring.
    """
    lin = np.gradient(np.asarray(body_pos_w, dtype=np.float64), dt, axis=0)
    ang = _angular_velocity_world(body_quat_w, dt)
    sigma = _VELOCITY_SMOOTH_SIGMA
    return _gaussian_smooth1d(lin, sigma), _gaussian_smooth1d(ang, sigma)


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


def _batched_fk(model, data, qpos: np.ndarray) -> np.ndarray:
    """Run FK for a (T, dof) trajectory on the shared fixed-batch ``data``.

    Only link poses are read; velocities are derived from those poses by the
    caller (``_body_velocities_from_poses``). The trajectory is processed in
    ``data``-shaped chunks; the tail is padded with its last frame to fill the
    batch and the outputs are trimmed back. Chunk outputs are copied because
    the next chunk overwrites the buffers.
    """
    total = len(qpos)
    chunk = data.shape[0]
    pad = (-total) % chunk
    if pad:
        qpos = np.concatenate([qpos, np.repeat(qpos[-1:], pad, axis=0)])
    poses: list[np.ndarray] = []
    for start in range(0, len(qpos), chunk):
        block = slice(start, start + chunk)
        data.set_dof_pos(qpos[block], model)
        model.forward_kinematic(data)
        poses.append(np.array(model.get_link_poses(data), dtype=np.float32))
    return np.concatenate(poses)[:total]


_CONVERT_WORKER: tuple[object, object] | None = None


def _init_convert_worker(model_file: str) -> None:
    """Load the per-worker FK model and its fixed-batch buffer once (see ``_batched_fk``)."""
    global _CONVERT_WORKER
    model = mtx.load_model(model_file)
    _CONVERT_WORKER = (model, mtx.SceneData(model, batch=[_FK_CHUNK_FRAMES]))


def _convert_pair_job(item: tuple[Path, Path, Path]) -> None:
    model, fk_data = _CONVERT_WORKER
    csv_path, pkl_path, output_path = item
    _convert_pair(model, fk_data, csv_path, pkl_path, output_path)


def _convert_pair(model, fk_data, csv_path: Path, pkl_path: Path, output_path: Path) -> dict[str, object]:
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
    joint_vel = _forward_difference_track(dof, dt)

    selection = slice(0, num_frames)
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    if qpos.shape[1] != model.num_dof_pos:
        raise ValueError(f"qpos width ({qpos.shape[1]}) != model dof pos ({model.num_dof_pos})")

    poses = _batched_fk(model, fk_data, qpos[selection])
    body_pos_w = poses[:, :, 0:3]
    body_quat_w = poses[:, :, 3:7]
    body_lin_vel_w, body_ang_vel_w = _body_velocities_from_poses(body_pos_w, body_quat_w, dt)

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(_SONIC_FPS),
        "num_frames": np.int32(num_frames),
        "joint_names": np.asarray(model_joints),
        "body_names": np.asarray(body_names),
        "joint_pos": np.ascontiguousarray(dof[selection], dtype=np.float32),
        "joint_vel": np.ascontiguousarray(joint_vel[selection], dtype=np.float32),
        "body_pos_w": np.ascontiguousarray(body_pos_w, dtype=np.float32),
        "body_quat_w": np.ascontiguousarray(body_quat_w, dtype=np.float32),
        "body_lin_vel_w": np.ascontiguousarray(body_lin_vel_w, dtype=np.float32),
        "body_ang_vel_w": np.ascontiguousarray(body_ang_vel_w, dtype=np.float32),
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


def filter_bones_seed_pairs(
    pairs: list[tuple[Path, Path]], keywords: Sequence[str]
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Drop pairs whose clip stem contains any keyword (case-insensitive substring)."""
    lowered = tuple(keyword.lower() for keyword in keywords)
    kept: list[tuple[Path, Path]] = []
    filtered: list[str] = []
    for csv_path, pkl_path in pairs:
        stem = csv_path.stem.lower()
        if any(keyword in stem for keyword in lowered):
            filtered.append(csv_path.stem)
        else:
            kept.append((csv_path, pkl_path))
    return kept, filtered


def convert_bones_seed_dataset(
    robot_root: str | Path,
    smpl_root: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    max_clips: int | None = None,
    model_file: str | Path | None = None,
    filter_keywords: Sequence[str] | None = None,
    workers: int = 1,
) -> dict[str, object]:
    """Convert every paired clip under the raw roots into ``output_dir``.

    Existing outputs are skipped unless ``overwrite``; clips whose stem matches
    ``filter_keywords`` (upstream keyword list when None, no filtering when
    empty) are dropped before conversion; ``workers`` processes share the work,
    each loading its own model and fixed-batch FK buffer, so peak memory scales
    linearly with ``workers``. A clip that fails conversion is recorded and the
    batch continues. The corpus directory is ready for ``SonicMotionClip.from_corpus``
    once done.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    pairs = resolve_bones_seed_pairs(robot_root, smpl_root)
    keywords = BONES_SEED_FILTER_KEYWORDS if filter_keywords is None else filter_keywords
    pairs, filtered = filter_bones_seed_pairs(pairs, keywords)
    if max_clips is not None:
        pairs = pairs[:max_clips]
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        Path(model_file).expanduser() if model_file is not None else UNITREE_G1_ASSET_DIR / "scene_g1_29dof.xml"
    )

    todo: list[tuple[Path, Path, Path]] = []
    skipped: list[str] = []
    for csv_path, pkl_path in pairs:
        destination = output_dir / f"{csv_path.stem}.npz"
        if destination.exists() and not overwrite:
            skipped.append(csv_path.stem)
        else:
            todo.append((csv_path, pkl_path, destination))

    converted: list[str] = []
    failed: dict[str, str] = {}
    order = {item[0].stem: index for index, item in enumerate(todo)}

    def record(stem: str, error: BaseException | None) -> None:
        if error is None:
            converted.append(stem)
        else:
            print(f"  failed {stem}: {error}")
            failed[stem] = str(error)
            (output_dir / f"{stem}.npz").unlink(missing_ok=True)

    if workers == 1:
        model = mtx.load_model(str(model_path))
        fk_data = mtx.SceneData(model, batch=[_FK_CHUNK_FRAMES])
        for index, (csv_path, pkl_path, destination) in enumerate(todo, start=1):
            try:
                _convert_pair(model, fk_data, csv_path, pkl_path, destination)
                record(csv_path.stem, None)
            except (ValueError, OSError) as error:
                record(csv_path.stem, error)
            if index % 200 == 0:
                print(f"  {index}/{len(todo)} clips processed ({len(failed)} failed)")
    else:
        # Spawn (not fork): the MotrixSim runtime keeps native state that does
        # not survive a fork of a process which already loaded a model.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_convert_worker,
            initargs=(str(model_path),),
        ) as pool:
            futures = {pool.submit(_convert_pair_job, item): item[0].stem for item in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                stem = futures[future]
                try:
                    future.result()
                    record(stem, None)
                except (ValueError, OSError) as error:
                    record(stem, error)
                if index % 200 == 0:
                    print(f"  {index}/{len(todo)} clips processed ({len(failed)} failed)")
        converted.sort(key=order.__getitem__)

    if not converted and not skipped:
        raise RuntimeError(f"Every BONES-SEED clip failed to convert under {robot_root} / {smpl_root}")
    return {
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "filtered": filtered,
        "output_dir": str(output_dir),
    }

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert paired LAFAN1 G1 CSV / BVH clips into a SONIC NPZ corpus.

Every output file is one MotrixLab motion NPZ v1 clip carrying the SONIC
``ext_smpl_joints`` / ``ext_smpl_root_quat`` reference channels — the same
layout the BONES-SEED converter produces, directly consumable by
``SonicMotionClip.from_corpus`` (set ``SONIC_MOTION_DIR`` to the output
directory). Per clip, the two pathways meet:

- retarget: the G1 CSV is resampled 30→50 Hz and baked through MotrixSim FK
  (``retarget.py``);
- BVH→SMPL: the raw BVH rotations are mapped onto the SMPL hierarchy and
  materialized into the SONIC SMPL frame (``bvh.py`` + ``smpl.py``).

Both sides are resampled onto the same target grid, so the robot and SMPL
channels stay frame-aligned. ``scripts/motion/download_lafan4sonic.py``
materializes the paired raw sources (BVH + G1 CSV + human_joints_info.pkl).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from motrix_envs.motion.lafan4sonic import retarget as _retarget
from motrix_envs.motion.lafan4sonic._math import (
    body_velocities_from_poses,
    forward_difference_track,
    target_times,
)
from motrix_envs.motion.lafan4sonic.bvh import load_bvh, smpl_pose_from_bvh
from motrix_envs.motion.lafan4sonic.smpl import load_smpl_skeleton, materialize_smpl_channels
from motrix_envs.motion.schema import SCHEMA_VERSION
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR

SONIC_FPS = 50
# SONIC observations read 10 future reference frames; shorter clips cannot
# feed the command.
MIN_FRAMES = 10
_FPS_TOLERANCE = 1.0e-3
_G1_ROOT_BODY = "pelvis"
_SONIC_REFERENCE_BODY = "pelvis"


def _natural_sort_key(value: str | Path) -> list[int | str]:
    text = value.as_posix() if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def resolve_lafan4sonic_pairs(robot_root: str | Path, bvh_root: str | Path) -> list[tuple[Path, Path]]:
    """Match LAFAN G1 CSV and BVH clips by file stem (sorted, natural order)."""
    robots = {path.stem: path for path in Path(robot_root).expanduser().rglob("*.csv")}
    bvhs = {path.stem: path for path in Path(bvh_root).expanduser().rglob("*.bvh")}
    if not robots or not bvhs:
        raise FileNotFoundError(
            f"No LAFAN sources found under {robot_root} / {bvh_root}; run scripts/motion/download_lafan4sonic.py first"
        )
    names = sorted(set(robots) & set(bvhs), key=lambda n: _natural_sort_key(robots[n].name))
    if not names:
        raise FileNotFoundError("No paired LAFAN G1/BVH clips share a stem")
    return [(robots[name], bvhs[name]) for name in names]


def convert_lafan4sonic_pair(
    model, fk_data, skeleton, csv_path: Path, bvh_path: Path, output_path: Path
) -> dict[str, object]:
    """Convert one paired LAFAN clip into a MotrixLab motion NPZ v1."""
    model_joints = [str(name) for name in model.joint_names]
    body_names = [str(name) for name in model.link_names]

    root_pos, root_quat, dof_csv = _retarget.load_g1_csv(csv_path)
    bvh = load_bvh(bvh_path)
    if abs(bvh.fps - _retarget.LAFAN_SOURCE_FPS) > _FPS_TOLERANCE:
        raise ValueError(f"BVH {bvh_path.name} is not {_retarget.LAFAN_SOURCE_FPS:g} Hz (got {bvh.fps:.4f})")
    if len(bvh.local_quat_xyzw) != len(root_pos):
        raise ValueError(
            f"G1/BVH frame-count mismatch for {csv_path.stem}: {len(root_pos)} vs {len(bvh.local_quat_xyzw)}"
        )

    # Both pathways resample onto the same 50 Hz grid, so the robot and SMPL
    # channels stay frame-aligned.
    times = target_times(len(root_pos), _retarget.LAFAN_SOURCE_FPS, SONIC_FPS)
    root_pos, root_quat, dof_csv = _retarget.resample_robot_track(root_pos, root_quat, dof_csv, SONIC_FPS)
    root_rotvec, body_rotvec = smpl_pose_from_bvh(bvh, times)
    smpl_joints, smpl_root_quat = materialize_smpl_channels(root_rotvec, body_rotvec, skeleton)

    dof = dof_csv[:, _retarget.g1_csv_to_model_order(model_joints)]
    num_frames = min(len(dof), len(smpl_joints))
    if num_frames < MIN_FRAMES:
        raise ValueError(f"Paired LAFAN clip {csv_path.stem!r} is too short: {num_frames} frames")

    dt = 1.0 / SONIC_FPS
    joint_vel = forward_difference_track(dof, dt)

    selection = slice(0, num_frames)
    qpos = np.concatenate([root_pos, root_quat, dof], axis=1).astype(np.float32)
    if qpos.shape[1] != model.num_dof_pos:
        raise ValueError(f"qpos width ({qpos.shape[1]}) != model dof pos ({model.num_dof_pos})")

    poses = _retarget.batched_fk(model, fk_data, qpos[selection])
    body_pos_w = poses[:, :, 0:3]
    body_quat_w = poses[:, :, 3:7]
    body_lin_vel_w, body_ang_vel_w = body_velocities_from_poses(body_pos_w, body_quat_w, dt)

    output = {
        "schema_version": np.int32(SCHEMA_VERSION),
        "fps": np.int32(SONIC_FPS),
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


def convert_lafan4sonic_dataset(
    robot_root: str | Path,
    bvh_root: str | Path,
    output_dir: str | Path,
    *,
    human_joints_info: str | Path,
    overwrite: bool = False,
    max_clips: int | None = None,
    model_file: str | Path | None = None,
) -> dict[str, object]:
    """Convert every paired LAFAN clip under the raw roots into ``output_dir``.

    Existing outputs are skipped unless ``overwrite``; a clip that fails
    conversion is recorded and the batch continues. The corpus directory is
    ready for ``SonicMotionClip.from_corpus`` once done.
    """
    pairs = resolve_lafan4sonic_pairs(robot_root, bvh_root)
    if max_clips is not None:
        pairs = pairs[:max_clips]
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = (
        Path(model_file).expanduser() if model_file is not None else UNITREE_G1_ASSET_DIR / "scene_g1_29dof.xml"
    )

    info_path = Path(human_joints_info).expanduser()
    if not info_path.is_file():
        raise FileNotFoundError(f"human_joints_info.pkl not found: {info_path}")
    skeleton = load_smpl_skeleton(info_path)
    model, fk_data = _retarget.load_fk_model(model_path)

    todo: list[tuple[Path, Path, Path]] = []
    skipped: list[str] = []
    for csv_path, bvh_path in pairs:
        destination = output_dir / f"{csv_path.stem}.npz"
        if destination.exists() and not overwrite:
            skipped.append(csv_path.stem)
        else:
            todo.append((csv_path, bvh_path, destination))

    converted: list[str] = []
    failed: dict[str, str] = {}
    for index, (csv_path, bvh_path, destination) in enumerate(todo, start=1):
        stem = csv_path.stem
        try:
            convert_lafan4sonic_pair(model, fk_data, skeleton, csv_path, bvh_path, destination)
            converted.append(stem)
        except (ValueError, OSError) as error:
            print(f"  failed {stem}: {error}")
            failed[stem] = str(error)
            destination.unlink(missing_ok=True)
        if index % 10 == 0:
            print(f"  {index}/{len(todo)} clips processed ({len(failed)} failed)")

    if not converted and not skipped:
        raise RuntimeError(f"Every LAFAN clip failed to convert under {robot_root} / {bvh_root}")
    return {
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "output_dir": str(output_dir),
    }

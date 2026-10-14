# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The retarget pathway: LAFAN1_Retargeting_Dataset G1 CSVs → robot channels.

The dataset (https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
publishes headerless per-clip CSVs at 30 fps with 36 columns per row:

    [0:3]   root position   (x, y, z)                    meters
    [3:7]   root quaternion  (qx, qy, qz, qw)  -- xyzw
    [7:36]  29 joint angles  (radians)         -- G1_CSV_JOINT_ORDER

This module resamples the reduced state to the 50 Hz SONIC control rate and
bakes world body poses/velocities through MotrixSim forward kinematics with
the same velocity contract as the BONES-SEED converter, so both corpora are
interchangeable inside one ``SONIC_MOTION_DIR``.
"""

from __future__ import annotations

from pathlib import Path

import motrixsim as mtx
import numpy as np

from motrix_envs.motion.converters.lafan_converter import G1_CSV_JOINT_ORDER
from motrix_envs.motion.lafan4sonic._math import resample_track

LAFAN_SOURCE_FPS = 30.0
# MotrixSim never returns a released SceneData's native buffers (~23 KB/frame),
# so batched FK must reuse one fixed-batch data object for the whole dataset —
# a fresh per-clip SceneData leaks linearly with total frames and OOM-kills
# long conversion runs.
FK_CHUNK_FRAMES = 2048

_G1_NUM_JOINTS = len(G1_CSV_JOINT_ORDER)
_G1_CSV_COLUMNS = 3 + 4 + _G1_NUM_JOINTS


def load_g1_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a retargeted G1 CSV -> (root_pos, root_quat_xyzw, dof) at 30 fps.

    Joint columns come back in the dataset order (``G1_CSV_JOINT_ORDER``);
    remap to the model order with :func:`g1_csv_to_model_order`.
    """
    path = Path(path)
    values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if values.shape[1] != _G1_CSV_COLUMNS or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid LAFAN G1 CSV shape/content in {path}: {values.shape}")
    quat = values[:, 3:7]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError(f"Zero-length root quaternion in {path}")
    return values[:, 0:3], quat / norm, values[:, 7:_G1_CSV_COLUMNS]


def g1_csv_to_model_order(model_joint_names: list[str]) -> list[int]:
    """Column permutation taking dataset-order dof columns to model order."""
    return [G1_CSV_JOINT_ORDER.index(name) for name in model_joint_names]


def resample_robot_track(
    root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray, target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample the 30 fps robot track onto the shared target grid (lerp/slerp)."""
    return resample_track(root_pos, root_quat, dof, LAFAN_SOURCE_FPS, target_fps)


def batched_fk(model, data, qpos: np.ndarray) -> np.ndarray:
    """Run FK for a (T, dof) trajectory on the shared fixed-batch ``data``.

    Only link poses are read; velocities are derived from those poses by the
    caller. The trajectory is processed in ``data``-shaped chunks; the tail is
    padded with its last frame and the outputs trimmed back. Chunk outputs are
    copied because the next chunk overwrites the buffers.
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


def load_fk_model(model_file: str | Path):
    """Load the G1 model and its fixed-batch FK buffer for corpus conversion."""
    model = mtx.load_model(str(model_file))
    model_joints = [str(name) for name in model.joint_names]
    if sorted(model_joints) != sorted(G1_CSV_JOINT_ORDER):
        raise ValueError(
            "Model joints do not match the LAFAN G1 joint set.\n"
            f"  model: {model_joints}\n  csv:   {list(G1_CSV_JOINT_ORDER)}"
        )
    return model, mtx.SceneData(model, batch=[FK_CHUNK_FRAMES])

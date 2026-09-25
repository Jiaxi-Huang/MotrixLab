# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_envs.motion import MotrixMotion
from motrix_envs.motion.sonic import SonicMotionClip, SonicPackedMotion, pack_sonic_store


def test_sonic_from_motion_requires_smpl_extensions(tmp_path) -> None:
    identity = np.tile(np.asarray([0, 0, 0, 1], np.float32), (4, 2, 1))
    path = tmp_path / "m.npz"
    np.savez(
        path,
        schema_version=np.int32(1),
        fps=np.int32(50),
        num_frames=np.int32(4),
        joint_names=np.asarray(["j0", "j1"]),
        body_names=np.asarray(["b0", "b1"]),
        joint_pos=np.zeros((4, 2), np.float32),
        joint_vel=np.zeros((4, 2), np.float32),
        body_pos_w=np.zeros((4, 2, 3), np.float32),
        body_quat_w=identity,
        body_lin_vel_w=np.zeros((4, 2, 3), np.float32),
        body_ang_vel_w=np.zeros((4, 2, 3), np.float32),
    )

    with pytest.raises(ValueError, match="smpl"):
        SonicMotionClip.from_motion(MotrixMotion(path), ("j0", "j1"), ("b0", "b1"), "b0", "b0")


def test_sonic_packed_roundtrip(tmp_path):
    joints = ("j0", "j1")
    bodies = ("b0", "b1")
    clip = {
        "joint_pos": np.zeros((10, 2), np.float32),
        "joint_vel": np.ones((10, 2), np.float32),
        "body_pos_w": np.zeros((10, 2, 3), np.float32),
        "body_quat_w": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 2, 1)),
        "body_lin_vel_w": np.zeros((10, 2, 3), np.float32),
        "body_ang_vel_w": np.zeros((10, 2, 3), np.float32),
        "smpl_joints": np.zeros((10, 24, 3), np.float32),
        "smpl_root_quat": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1)),
    }
    root = pack_sonic_store(tmp_path / "store", clips=[clip], joint_names=joints, body_names=bodies)
    loaded = SonicPackedMotion(root, joint_names=joints, body_names=bodies)
    assert loaded.num_frames == 10
    assert loaded.clip_lengths.tolist() == [10]
    assert isinstance(loaded.joint_pos, np.memmap)
    np.testing.assert_array_equal(loaded.joint_vel, clip["joint_vel"])

    with pytest.raises(ValueError, match="joint order"):
        SonicPackedMotion(root, joint_names=tuple(reversed(joints)), body_names=bodies, clip_limit=1)


def test_sonic_packed_clip_limit_materializes_only_the_selected_prefix(tmp_path) -> None:
    joints = ("j0", "j1")
    bodies = ("b0",)

    def clip(value: float) -> dict[str, np.ndarray]:
        return {
            "joint_pos": np.full((10, 2), value, np.float32),
            "joint_vel": np.full((10, 2), value, np.float32),
            "body_pos_w": np.zeros((10, 1, 3), np.float32),
            "body_quat_w": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1, 1)),
            "body_lin_vel_w": np.zeros((10, 1, 3), np.float32),
            "body_ang_vel_w": np.zeros((10, 1, 3), np.float32),
            "smpl_joints": np.zeros((10, 24, 3), np.float32),
            "smpl_root_quat": np.tile(np.asarray([0, 0, 0, 1], np.float32), (10, 1)),
        }

    root = pack_sonic_store(
        tmp_path / "store",
        clips=[clip(1.0), clip(2.0)],
        joint_names=joints,
        body_names=bodies,
    )

    loaded = SonicPackedMotion(root, joint_names=joints, body_names=bodies, clip_limit=1)

    assert loaded.num_clips == 1
    assert loaded.num_frames == 10
    np.testing.assert_array_equal(loaded.joint_pos, np.ones((10, 2), dtype=np.float32))

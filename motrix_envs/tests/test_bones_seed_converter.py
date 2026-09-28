# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import joblib
import numpy as np

from motrix_envs.locomotion.sonic.mdp import G1_SONIC_JOINTS
from motrix_envs.motion.converters.bones_seed_converter import (
    BONES_SEED_FILTER_KEYWORDS,
    convert_bones_seed_dataset,
    filter_bones_seed_pairs,
)

_ROOT_Z_CM = 79.3  # G1 pelvis rest height in centimeters (CSV positions are cm)
_ROOT_VEL_MPS = 0.3
_ROOT_OMEGA_RPS = 0.5
_ROOT_YAW_DEG = 90.0
_CSV_FPS = 120.0


def _write_synthetic_clip(robot_dir, smpl_dir, name, num_frames=121):
    """Fabricate a BONES-SEED CSV/PKL pair with analytically known velocities.

    The root translates along world X at ``_ROOT_VEL_MPS`` while rotating at a
    constant rate about world Y (fixed 90° yaw composed with a linear roll
    ramp), so the pelvis linear/angular velocities are constants.
    """
    t = np.arange(num_frames) / _CSV_FPS
    rows = np.zeros((num_frames, 7 + len(G1_SONIC_JOINTS)))
    rows[:, 0] = np.arange(num_frames)
    rows[:, 1] = 100.0 * _ROOT_VEL_MPS * t
    rows[:, 3] = _ROOT_Z_CM
    rows[:, 4] = np.rad2deg(_ROOT_OMEGA_RPS) * t
    rows[:, 6] = _ROOT_YAW_DEG
    rows[:, 7:] = 0.1 * np.sin(2.0 * np.pi * t)[:, None]
    header = "Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ," + ",".join(
        f"{joint}_dof" for joint in G1_SONIC_JOINTS
    )
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(robot_dir / f"{name}.csv", rows, delimiter=",", header=header, comments="", fmt="%.9f")
    joblib.dump(
        {
            "fps": 50,
            "pose_aa": np.zeros((3 * num_frames, 24, 3), np.float32),
            "smpl_joints": np.zeros((3 * num_frames, 24, 3), np.float32),
        },
        smpl_dir / f"{name}.pkl",
    )


def test_bones_seed_velocity_contract(tmp_path) -> None:
    _write_synthetic_clip(tmp_path / "robot", tmp_path / "smpl", "analytic_walk")
    result = convert_bones_seed_dataset(tmp_path / "robot", tmp_path / "smpl", tmp_path / "out", filter_keywords=())
    assert result["converted"] == ["analytic_walk"] and not result["failed"]

    data = np.load(tmp_path / "out" / "analytic_walk.npz")
    dt = 1.0 / 50.0
    pelvis = int(np.flatnonzero(data["body_names"] == "pelvis")[0])

    # Constant-rate world motion: pelvis velocities must be the constants
    # (vx, 0, 0) and (0, omega, 0) in the WORLD frame. The engine free-joint
    # qvel frame bug this test pins produced root-orientation-rotated angular
    # velocities instead ((-omega, 0, 0) here, a ~sqrt(2) x error).
    np.testing.assert_allclose(
        data["body_lin_vel_w"][:, pelvis], np.tile((_ROOT_VEL_MPS, 0.0, 0.0), (data["num_frames"], 1)), atol=2e-2
    )
    np.testing.assert_allclose(
        data["body_ang_vel_w"][:, pelvis], np.tile((0.0, _ROOT_OMEGA_RPS, 0.0), (data["num_frames"], 1)), atol=2e-2
    )

    # Upstream dof-vel contract: forward differences with the final frame's
    # difference duplicated.
    joint_pos = data["joint_pos"].astype(np.float64)
    expected = np.empty_like(joint_pos)
    expected[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt
    expected[-1] = expected[-2]
    np.testing.assert_allclose(data["joint_vel"], expected, atol=1e-5)

    quat_norms = np.linalg.norm(data["body_quat_w"], axis=-1)
    np.testing.assert_allclose(quat_norms, 1.0, atol=1e-3)
    for field in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_vel"):
        assert np.all(np.isfinite(data[field]))


def test_bones_seed_parallel_workers_match_serial_and_resume(tmp_path) -> None:
    for name in ("par_a", "par_b"):
        _write_synthetic_clip(tmp_path / "robot", tmp_path / "smpl", name)
    serial = convert_bones_seed_dataset(tmp_path / "robot", tmp_path / "smpl", tmp_path / "serial", filter_keywords=())
    parallel = convert_bones_seed_dataset(
        tmp_path / "robot", tmp_path / "smpl", tmp_path / "parallel", filter_keywords=(), workers=2
    )
    assert parallel["converted"] == serial["converted"] == ["par_a", "par_b"]

    for name in ("par_a", "par_b"):
        reference = np.load(tmp_path / "serial" / f"{name}.npz")
        worker_output = np.load(tmp_path / "parallel" / f"{name}.npz")
        for field in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_array_equal(reference[field], worker_output[field])

    # Re-running over the same output converts nothing: existing NPZs are
    # skipped, so a parallel run resumes instead of redoing (or corrupting) work.
    resumed = convert_bones_seed_dataset(
        tmp_path / "robot", tmp_path / "smpl", tmp_path / "parallel", filter_keywords=(), workers=2
    )
    assert resumed["converted"] == []
    assert sorted(resumed["skipped"]) == ["par_a", "par_b"]


def _fake_path(stem):
    return Path(f"/corpus/{stem}.csv")


def test_bones_seed_filter_pairs_by_keyword() -> None:
    pairs = [(a, a) for a in map(_fake_path, ["handstand_hold", "jump_ff_180_R", "sit_down", "warm_up"])]
    kept, filtered = filter_bones_seed_pairs(pairs, BONES_SEED_FILTER_KEYWORDS)
    assert [p[0].stem for p in kept] == ["jump_ff_180_R", "warm_up"]
    assert filtered == ["handstand_hold", "sit_down"]

    # Case-insensitive substring match, mirroring the upstream filter.
    kept_ci, _ = filter_bones_seed_pairs(
        [(_fake_path("CartWheel_L"), _fake_path("CartWheel_L"))], BONES_SEED_FILTER_KEYWORDS
    )
    assert not kept_ci

    # Empty keyword list disables filtering entirely.
    kept_all, filtered_none = filter_bones_seed_pairs(pairs, ())
    assert len(kept_all) == len(pairs) and not filtered_none

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SonicMotionClip corpus assembly and the BONES-SEED converter contract."""

import numpy as np

from motrix_envs.motion import MotrixMotion
from motrix_envs.motion.converters.bones_seed_converter import (
    convert_bones_seed_dataset,
    resolve_bones_seed_pairs,
)
from motrix_envs.motion.converters.lafan_converter import G1_CSV_JOINT_ORDER
from motrix_envs.motion.sonic import SonicMotionClip

# Task-order selection with joints shuffled away from the file order.
_TASK_ORDER = {
    "joint_names": ("joint_2", "joint_0"),
    "tracked_body_names": ("body_1", "body_3"),
    "reference_body_name": "body_2",
    "root_body_name": "body_0",
}


def _write_sonic_npz(path, *, num_frames=6, fps=50, tag="clip", joints=("joint_0", "joint_1", "joint_2")):
    rng = np.random.default_rng(7)
    num_joints = len(joints)
    body_names = ("body_0", "body_1", "body_2", "body_3")
    num_bodies = len(body_names)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    body_quat_w[..., 3] = 1.0
    smpl_quat = np.zeros((num_frames, 4), dtype=np.float32)
    smpl_quat[:, 3] = 1.0
    np.savez(
        path,
        schema_version=np.int32(1),
        fps=np.int32(fps),
        num_frames=np.int32(num_frames),
        joint_names=np.asarray(joints),
        body_names=np.asarray(body_names),
        joint_pos=rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        joint_vel=rng.standard_normal((num_frames, num_joints)).astype(np.float32),
        body_pos_w=rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        body_ang_vel_w=rng.standard_normal((num_frames, num_bodies, 3)).astype(np.float32),
        ext_smpl_joints=rng.standard_normal((num_frames, 24, 3)).astype(np.float32),
        ext_smpl_root_quat=smpl_quat,
        clip_name=np.asarray(tag),
    )


def test_from_corpus_concatenates_clip_boundaries(tmp_path):
    _write_sonic_npz(tmp_path / "a.npz", num_frames=6)
    _write_sonic_npz(tmp_path / "b.npz", num_frames=10)
    clip = SonicMotionClip.from_corpus((tmp_path,), **_TASK_ORDER)

    assert clip.joint_pos.shape == (16, 2)  # task order selects two joints
    # Frames of clip "a" end at global 5, clip "b" frames end at global 15.
    assert clip.frame_clip_end.tolist() == [5] * 6 + [15] * 10
    assert clip.smpl_joints.shape == (16, 24, 3)
    assert clip.smpl_root_quat.shape == (16, 4)


# ---------------------------------------------------------------------------
# BONES-SEED converter
# ---------------------------------------------------------------------------

_ROOT_Z_CM = 79.3
_JOINT_VALUE_DEG = 10.0


def _write_bones_csv(path, num_rows_120hz):
    """Fabricate a constant-pose BONES-SEED csv (36 cols, 120 fps, header row)."""
    joints = list(G1_CSV_JOINT_ORDER)[::-1]  # reversed order exercises the remap
    header = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ] + [f"{name}_dof" for name in joints]
    rows = np.zeros((num_rows_120hz, 36), dtype=np.float64)
    rows[:, 0] = np.arange(num_rows_120hz)
    rows[:, 1] = 12.0
    rows[:, 2] = -3.0
    rows[:, 3] = _ROOT_Z_CM
    rows[:, 7:] = _JOINT_VALUE_DEG  # degrees; the converter applies deg2rad
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        np.savetxt(handle, rows, delimiter=",")
    return joints


def _write_smpl_pkl(path, num_frames):
    import joblib

    pose_aa = np.zeros((num_frames, 24, 3), dtype=np.float32)
    smpl_joints = np.zeros((num_frames, 24, 3), dtype=np.float32)
    smpl_joints[:, :, 0] = np.linspace(0.0, 1.0, num_frames)[:, None]
    joblib.dump({"fps": 50, "pose_aa": pose_aa, "smpl_joints": smpl_joints}, path)


def test_from_corpus_requires_smpl_channels(tmp_path):
    # A schema v1 file without the SMPL extension channels cannot train SONIC.
    _write_sonic_npz(tmp_path / "a.npz")
    with np.load(tmp_path / "a.npz") as data:
        payload = {key: data[key] for key in data.files if not key.startswith("ext_")}
    np.savez(tmp_path / "bare.npz", **payload)
    try:
        SonicMotionClip.from_corpus((tmp_path / "bare.npz",), **_TASK_ORDER)
    except ValueError as error:
        assert "ext_smpl" in str(error) or "smpl_joints" in str(error)
    else:
        raise AssertionError("from_corpus must reject clips without SMPL channels")


def test_convert_bones_seed_dataset_round_trip(tmp_path):
    num_rows = 4 * 61  # 120 fps rows -> stride 4 -> 61 rows @30Hz -> 100 frames @50Hz
    robot_root = tmp_path / "robot_filtered"
    smpl_root = tmp_path / "smpl_filtered"
    robot_root.mkdir()
    smpl_root.mkdir()
    _write_bones_csv(robot_root / "clip.csv", num_rows)
    _write_smpl_pkl(smpl_root / "clip.pkl", 100)
    result = convert_bones_seed_dataset(robot_root, smpl_root, tmp_path / "corpus")
    assert result["converted"] == ["clip"] and result["failed"] == {}

    motion = MotrixMotion(tmp_path / "corpus" / "clip.npz")
    assert motion.fps == 50
    assert motion.num_frames == 100
    assert set(motion.joint_names) == set(G1_CSV_JOINT_ORDER)
    assert motion.root_body_name == "pelvis"
    assert motion.reference_body_name == "pelvis"

    # Constant input survives stride/resample/fk: dof in radians, root in meters.
    np.testing.assert_allclose(motion.joint_pos, np.deg2rad(_JOINT_VALUE_DEG), atol=1e-5)
    root = motion.body_index("pelvis")
    np.testing.assert_allclose(motion.body_pos_w[:, root, 0], 0.12, atol=1e-4)
    np.testing.assert_allclose(motion.body_pos_w[:, root, 1], -0.03, atol=1e-4)
    np.testing.assert_allclose(motion.body_pos_w[:, root, 2], _ROOT_Z_CM / 100.0, atol=1e-4)

    # SMPL channels ride along as ext_ channels with unit root quats
    # (zero axis-angle root composed with the fixed SONIC rotations).
    smpl_joints = motion.extensions["smpl_joints"]
    smpl_root_quat = motion.extensions["smpl_root_quat"]
    assert smpl_joints.shape == (100, 24, 3)
    assert smpl_root_quat.shape == (100, 4)
    np.testing.assert_allclose(np.linalg.norm(smpl_root_quat, axis=-1), 1.0, atol=1e-5)
    # Constant input -> the fixed SONIC y-up→z-up root orientation on every frame.
    np.testing.assert_allclose(smpl_root_quat, np.broadcast_to(smpl_root_quat[0], smpl_root_quat.shape), atol=1e-6)
    np.testing.assert_allclose(smpl_root_quat[0], [0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)], atol=1e-5)


def test_resolve_pairs_and_dataset_skip(tmp_path):
    robot = tmp_path / "robot_filtered"
    smpl = tmp_path / "smpl_filtered"
    out = tmp_path / "corpus"
    robot.mkdir()
    smpl.mkdir()
    for name in ("a", "b", "c"):
        _write_bones_csv(robot / f"{name}.csv", 4 * 61)
        _write_smpl_pkl(smpl / f"{name}.pkl", 100)
    (robot / "unpaired.csv").write_text("x", encoding="utf-8")

    pairs = resolve_bones_seed_pairs(robot, smpl)
    assert [p[0].stem for p in pairs] == ["a", "b", "c"]

    first = convert_bones_seed_dataset(robot, smpl, out, max_clips=2)
    assert first["converted"] == ["a", "b"]
    second = convert_bones_seed_dataset(robot, smpl, out)
    assert second["skipped"] == ["a", "b"] and second["converted"] == ["c"]
    assert len(list(out.glob("*.npz"))) == 3

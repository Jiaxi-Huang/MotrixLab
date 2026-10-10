# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""lafan4sonic: paired LAFAN G1/BVH conversion and SONIC corpus consumption."""

import numpy as np
import torch

from motrix_envs.locomotion.sonic.mdp import G1_SONIC_BODY_NAMES, G1_SONIC_JOINTS
from motrix_envs.motion import MotrixMotion
from motrix_envs.motion.converters.lafan_converter import G1_CSV_JOINT_ORDER
from motrix_envs.motion.lafan4sonic.bvh import load_bvh
from motrix_envs.motion.lafan4sonic.converter import convert_lafan4sonic_dataset, resolve_lafan4sonic_pairs
from motrix_envs.motion.sonic import SonicMotionClip

_ROOT_POS = (0.05, -0.03, 0.8)
_JOINT_VALUE = 0.1

# SMPL 55-joint kinematic tree (public SMPL topology; also the layout baked
# into gear_sonic's human_joints_info.pkl).
_SMPL_PARENTS = np.asarray(
    (
        "-1 0 0 0 1 2 3 4 5 6 7 8 9 9 9 12 13 14 16 17 18 19 15 15 15 20 25 26 20 28 29 "
        "20 31 32 20 34 35 20 37 38 21 40 41 21 43 44 21 46 47 21 49 50 21 52 53"
    ).split(),
    dtype=np.intp,
)

# The synthetic LAFAN skeleton: every bone required by the SMPL mapping plus
# Hips, arranged in the LAFAN hierarchy.
_BVH_JOINTS = (
    ("Hips", -1),
    ("LeftUpLeg", 0),
    ("LeftLeg", 1),
    ("LeftFoot", 2),
    ("LeftToe", 3),
    ("RightUpLeg", 0),
    ("RightLeg", 5),
    ("RightFoot", 6),
    ("RightToe", 7),
    ("Spine", 0),
    ("Spine1", 9),
    ("Spine2", 10),
    ("Neck", 11),
    ("Head", 12),
    ("LeftShoulder", 11),
    ("LeftArm", 14),
    ("LeftForeArm", 15),
    ("LeftHand", 16),
    ("RightShoulder", 11),
    ("RightArm", 18),
    ("RightForeArm", 19),
    ("RightHand", 20),
)


def _write_g1_csv(path, num_rows_30hz):
    """Fabricate a constant-pose LAFAN retarget CSV (36 cols, 30 fps, no header)."""
    rows = np.zeros((num_rows_30hz, 36), dtype=np.float64)
    rows[:, 0:3] = _ROOT_POS
    rows[:, 6] = 1.0  # xyzw identity root quaternion
    rows[:, 7:] = _JOINT_VALUE
    np.savetxt(path, rows, delimiter=",")


def _write_lafan_bvh(path, num_frames_30hz, *, frame_time="0.033333"):
    """Fabricate a zero-rotation LAFAN BVH with the SMPL-mapped bone set."""
    lines = ["HIERARCHY"]
    for index, (name, parent) in enumerate(_BVH_JOINTS):
        keyword = "ROOT" if parent < 0 else "JOINT"
        indent = " " * (4 if parent < 0 else 4 * (index + 1))
        lines.append(f"{indent}{keyword} {name}")
        lines.append(f"{indent}{{")
        channels = (
            "6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
            if parent < 0
            else "3 Zrotation Yrotation Xrotation"
        )
        lines.append(f"{indent}\tOFFSET 0 0 0")
        lines.append(f"{indent}\tCHANNELS {channels}")
    # Close braces from the deepest joint up to the root.
    for depth in range(len(_BVH_JOINTS), 0, -1):
        indent = " " * (4 * depth)
        lines.append(f"{indent}}}")
    motion = np.zeros((num_frames_30hz, 6 + 3 * (len(_BVH_JOINTS) - 1)), dtype=np.float64)
    lines.append("MOTION")
    lines.append(f"Frames: {num_frames_30hz}")
    lines.append(f"Frame Time: {frame_time}")
    for row in motion:
        lines.append(" ".join(f"{value:.6f}" for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_human_info_pkl(path):
    """Fabricate a human_joints_info.pkl with the SMPL topology and rest offsets."""
    rest = np.zeros((55, 3), dtype=np.float32)
    rest[:, 0] = np.linspace(0.0, 0.02, 55, dtype=np.float32)
    torch.save({"J": torch.as_tensor(rest), "parents_list": _SMPL_PARENTS.tolist()}, path)


def _write_pair(robot_root, bvh_root, name, num_rows=61):
    _write_g1_csv(robot_root / f"{name}.csv", num_rows)
    _write_lafan_bvh(bvh_root / f"{name}.bvh", num_rows)


def test_convert_lafan4sonic_round_trip(tmp_path):
    # 61 rows @ 30 Hz -> 100 frames @ 50 Hz (same grid arithmetic as BONES-SEED).
    robot_root = tmp_path / "g1"
    bvh_root = tmp_path / "bvh"
    robot_root.mkdir()
    bvh_root.mkdir()
    _write_pair(robot_root, bvh_root, "clip", num_rows=61)
    _write_human_info_pkl(tmp_path / "human_joints_info.pkl")
    result = convert_lafan4sonic_dataset(
        robot_root, bvh_root, tmp_path / "corpus", human_joints_info=tmp_path / "human_joints_info.pkl"
    )
    assert result["converted"] == ["clip"] and result["failed"] == {}

    motion = MotrixMotion(tmp_path / "corpus" / "clip.npz")
    assert motion.fps == 50
    assert motion.num_frames == 100
    assert set(motion.joint_names) == set(G1_CSV_JOINT_ORDER)
    assert motion.root_body_name == "pelvis"
    assert motion.reference_body_name == "pelvis"

    # Constant input survives resample + FK: dof values pass through, the root
    # body lands at the CSV world position.
    np.testing.assert_allclose(motion.joint_pos, _JOINT_VALUE, atol=1e-5)
    root = motion.body_index("pelvis")
    np.testing.assert_allclose(
        motion.body_pos_w[:, root, :3], np.broadcast_to(np.asarray(_ROOT_POS), (100, 3)), atol=1e-4
    )
    assert np.all(np.isfinite(motion.joint_vel))
    assert np.all(np.isfinite(motion.body_lin_vel_w)) and np.all(np.isfinite(motion.body_ang_vel_w))

    # SMPL channels: 24 root-local joints and a constant unit root quat. For a
    # zero-rotation BVH the heading correction, y-up->z-up and Hips rest offset
    # cancel exactly, leaving the fixed canonical rotation on every frame.
    smpl_joints = motion.extensions["smpl_joints"]
    smpl_root_quat = motion.extensions["smpl_root_quat"]
    assert smpl_joints.shape == (100, 24, 3)
    assert smpl_root_quat.shape == (100, 4)
    np.testing.assert_allclose(np.abs(smpl_joints[:, 0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(smpl_root_quat, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(smpl_root_quat, np.broadcast_to(smpl_root_quat[0], smpl_root_quat.shape), atol=1e-6)
    np.testing.assert_allclose(smpl_root_quat[0], [-0.5, -0.5, -0.5, 0.5], atol=1e-5)


def test_resolve_pairs_and_dataset_skip(tmp_path):
    robot = tmp_path / "g1"
    bvh = tmp_path / "bvh"
    out = tmp_path / "corpus"
    robot.mkdir()
    bvh.mkdir()
    for name in ("a", "b", "c"):
        _write_pair(robot, bvh, name)
    (robot / "unpaired.csv").write_text("x", encoding="utf-8")

    pairs = resolve_lafan4sonic_pairs(robot, bvh)
    assert [p[0].stem for p in pairs] == ["a", "b", "c"]

    _write_human_info_pkl(tmp_path / "human_joints_info.pkl")
    first = convert_lafan4sonic_dataset(
        robot, bvh, out, human_joints_info=tmp_path / "human_joints_info.pkl", max_clips=2
    )
    assert first["converted"] == ["a", "b"]
    second = convert_lafan4sonic_dataset(robot, bvh, out, human_joints_info=tmp_path / "human_joints_info.pkl")
    assert second["skipped"] == ["a", "b"] and second["converted"] == ["c"]
    assert len(list(out.glob("*.npz"))) == 3


def test_converter_rejects_misaligned_sources(tmp_path):
    robot = tmp_path / "g1"
    bvh = tmp_path / "bvh"
    robot.mkdir()
    bvh.mkdir()
    _write_g1_csv(robot / "clip.csv", 61)
    _write_lafan_bvh(bvh / "clip.bvh", 60)
    _write_human_info_pkl(tmp_path / "human_joints_info.pkl")

    from motrix_envs.motion.lafan4sonic import retarget as retarget_path
    from motrix_envs.motion.lafan4sonic.converter import convert_lafan4sonic_pair
    from motrix_envs.motion.lafan4sonic.smpl import load_smpl_skeleton
    from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR

    model, fk_data = retarget_path.load_fk_model(UNITREE_G1_ASSET_DIR / "scene_g1_29dof.xml")
    skeleton = load_smpl_skeleton(tmp_path / "human_joints_info.pkl")
    try:
        convert_lafan4sonic_pair(
            model, fk_data, skeleton, robot / "clip.csv", bvh / "clip.bvh", tmp_path / "corpus" / "clip.npz"
        )
    except ValueError as error:
        assert "frame-count mismatch" in str(error)
    else:
        raise AssertionError("conversion must reject a G1/BVH frame-count mismatch")
    assert not (tmp_path / "corpus" / "clip.npz").exists()


def test_from_corpus_consumes_lafan_corpus(tmp_path):
    robot = tmp_path / "g1"
    bvh = tmp_path / "bvh"
    robot.mkdir()
    bvh.mkdir()
    _write_pair(robot, bvh, "a", num_rows=61)  # -> 100 frames
    _write_pair(robot, bvh, "b", num_rows=121)  # -> 200 frames
    _write_human_info_pkl(tmp_path / "human_joints_info.pkl")
    convert_lafan4sonic_dataset(robot, bvh, tmp_path / "corpus", human_joints_info=tmp_path / "human_joints_info.pkl")

    clip = SonicMotionClip.from_corpus(
        (tmp_path / "corpus",),
        joint_names=list(G1_SONIC_JOINTS),
        tracked_body_names=G1_SONIC_BODY_NAMES,
        reference_body_name="pelvis",
        root_body_name="pelvis",
    )
    assert clip.joint_pos.shape == (300, 29)
    assert clip.tracked_bodies_pos_w.shape == (300, 14, 3)
    assert clip.smpl_joints.shape == (300, 24, 3)
    assert clip.smpl_root_quat.shape == (300, 4)
    assert clip.frame_clip_end.tolist() == [99] * 100 + [299] * 200


def test_load_bvh_rejects_unsupported_rotation_channels(tmp_path):
    path = tmp_path / "bad.bvh"
    path.write_text(
        "HIERARCHY\n"
        "ROOT Hips\n"
        "{\n"
        "\tOFFSET 0 0 0\n"
        "\tCHANNELS 3 Xrotation Yrotation Zrotation\n"
        "MOTION\n"
        "Frames: 2\n"
        "Frame Time: 0.033333\n"
        "0 0 0 0 0 0\n"
        "0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    try:
        load_bvh(path)
    except ValueError as error:
        assert "rotation channels" in str(error)
    else:
        raise AssertionError("load_bvh must reject non-ZYX rotation channels")

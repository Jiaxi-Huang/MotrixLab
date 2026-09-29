# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"End-to-end contract tests over a synthetic single-clip SONIC corpus."

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_envs.locomotion.sonic import mdp

_NUM_FRAMES = 64


def _write_corpus_npz(corpus_dir, name: str = "synthetic_clip") -> None:
    """Write a minimal contract-valid SONIC clip (29 joints / 14 bodies)."""
    joints = mdp.G1_SONIC_JOINTS
    bodies = mdp.G1_SONIC_BODY_NAMES
    t = _NUM_FRAMES
    identity_bodies = np.zeros((t, len(bodies), 4), np.float32)
    identity_bodies[..., 3] = 1.0
    identity_root = np.zeros((t, 4), np.float32)
    identity_root[:, 3] = 1.0
    body_pos = np.zeros((t, len(bodies), 3), np.float32)
    body_pos[:, 0, 2] = 0.79  # pelvis rest height keeps the reference sane
    corpus_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        corpus_dir / f"{name}.npz",
        schema_version=np.int32(1),
        fps=np.int32(50),
        num_frames=np.int32(t),
        joint_names=np.asarray(joints),
        body_names=np.asarray(bodies),
        joint_pos=np.zeros((t, len(joints)), np.float32),
        joint_vel=np.zeros((t, len(joints)), np.float32),
        body_pos_w=body_pos,
        body_quat_w=identity_bodies,
        body_lin_vel_w=np.zeros((t, len(bodies), 3), np.float32),
        body_ang_vel_w=np.zeros((t, len(bodies), 3), np.float32),
        root_body_name=np.asarray("pelvis"),
        reference_body_name=np.asarray("pelvis"),
        clip_name=np.asarray(name),
        ext_smpl_joints=np.zeros((t, 24, 3), np.float32),
        ext_smpl_root_quat=identity_root,
    )


def test_sonic_environment_compiles_and_steps(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus_npz(corpus)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus),)
    motion = cfg.commands.motion
    joints = len(motion.joint_names)
    bodies = len(motion.tracked_body_names)
    future_frames = motion.num_future_frames
    history_frames = cfg.observations.policy.obs.num_history_frames
    critic_history_frames = cfg.observations.value.obs.num_history_frames
    smpl_joints = 24
    policy_width = (
        history_frames * mdp._sonic_actor_frame_dim(joints)
        + future_frames * mdp._g1_reference_frame_dim(joints)
        + future_frames * mdp._smpl_reference_frame_dim(smpl_joints)
        + mdp._hybrid_reference_width(future_frames)
        + mdp.SONIC_ENCODER_COUNT
    )
    value_width = (
        9 + 9 * bodies + future_frames * 2 * joints + critic_history_frames * mdp._sonic_actor_frame_dim(joints)
    )

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)

    state = env.init_state()
    assert state.obs.policy.shape == (2, policy_width)
    assert state.obs.value.shape == (2, value_width)

    next_state = env.step(np.zeros((2, joints), dtype=np.float32))
    assert np.isfinite(next_state.obs.policy).all()
    assert np.isfinite(next_state.obs.value).all()
    assert np.isfinite(next_state.reward).all()


def test_sonic_action_contract_matches_wbt_affine_control(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus_npz(corpus)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus),)

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)
    env.init_state()

    term = env.action_terms["joint_position"]
    space = env.action_space
    # Shared WBT action term: scales derive from the model gains in the
    # declared SONIC joint order as base_scale * effort_limit / kp.
    kps = dict(
        zip(
            (spec.name for spec in env.model.actuators),
            np.asarray(env.model.others["actuator_kp"], dtype=np.float32),
        )
    )
    efforts = {
        spec.name: float(np.max(np.abs(np.asarray(spec.force_range, dtype=np.float32)))) for spec in env.model.actuators
    }
    base_scale = cfg.actions.joint_position.control.action_scale
    expected_scales = np.asarray(
        [base_scale * efforts[name] / kps[name] for name in mdp.G1_SONIC_JOINTS],
        dtype=np.float32,
    )
    np.testing.assert_allclose(term.action_scales, expected_scales, rtol=1e-6)
    np.testing.assert_allclose(space.low, -space.high)
    ctrl_ranges = {spec.name: np.asarray(spec.ctrl_range, dtype=np.float32) for spec in env.model.actuators}
    ranges = np.stack([ctrl_ranges[name] for name in mdp.G1_SONIC_JOINTS])
    residual = np.maximum(
        np.abs(ranges[:, 0] - term.default_angles),
        np.abs(ranges[:, 1] - term.default_angles),
    )
    np.testing.assert_allclose(space.high * term.action_scales, residual, rtol=1e-6)

    actions = np.stack((space.low * 0.25, space.high * 0.5))
    targets = term.process(actions)
    expected = actions * term.action_scales + term.default_angles
    np.testing.assert_allclose(targets, expected, atol=2e-7)


def test_sonic_declared_joint_order_matches_model_actuator_order(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus_npz(corpus)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus),)

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)
    env.init_state()

    # The declared policy order IS the model actuator order: corpus arrays,
    # dof queries and the action term all follow this single declaration, so
    # no permutation is needed anywhere in the pipeline.
    assert tuple(spec.name for spec in env.model.actuators) == mdp.G1_SONIC_JOINTS


def test_sonic_reset_seeds_foot_velocity_from_teleport_frame(tmp_path) -> None:
    """Reset must seed the foot finite-difference from the teleported frame.

    Zeroing it instead would make the first post-reset update report
    (v - 0)/dt as a large spurious acceleration that pollutes early-episode
    rewards and Q targets.
    """
    corpus = tmp_path / "corpus"
    _write_corpus_npz(corpus)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus),)

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)
    env.init_state()

    motion = env.command_terms["motion"]
    foot_indices = np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], dtype=np.int64)
    seeded = motion.clip.joint_vel[np.ix_(motion.steps[:, 0], foot_indices)]
    np.testing.assert_allclose(motion.previous_foot_joint_velocity, seeded, atol=1e-6)
    np.testing.assert_array_equal(motion.foot_joint_acceleration, 0.0)


def test_sonic_sim_reset_switches_segment_within_episode(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write_corpus_npz(corpus)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus),)
    # Neutralize failure terminations: the forced final-frame state carries a
    # pose mismatch that must not end the episode before the switch fires.
    cfg.terminations.anchor_pos_z.threshold = 1e6
    cfg.terminations.anchor_pos_z.low_reference_threshold = 1e6
    cfg.terminations.anchor_ori.threshold = 1e6
    cfg.terminations.ee_body_pos_z.threshold = 1e6
    cfg.terminations.ee_body_pos_z.low_reference_threshold = 1e6
    cfg.terminations.feet_pos.threshold = 1e6

    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=3)
    env.init_state()
    motion = env.command_terms["motion"]
    last_frame = motion.clip.joint_pos.shape[0] - 1
    motion.steps[:] = last_frame

    state = env.step(np.zeros((2, len(cfg.commands.motion.joint_names)), dtype=np.float32))

    # Reaching the clip end switched to a freshly sampled segment inside the
    # same episode: no episode boundary was crossed, the lanes left the final
    # frame, and the sim-only rematerialization reseeded the finite-difference
    # foot velocity from the new frame.
    assert not state.done.any()
    assert (state.episode_steps == 1).all()
    assert (motion.steps[:, 0] < last_frame).all()
    foot_indices = np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], dtype=np.int64)
    seeded = motion.clip.joint_vel[np.ix_(motion.steps[:, 0], foot_indices)]
    np.testing.assert_allclose(motion.previous_foot_joint_velocity, seeded, atol=1e-6)

    # The follow-up step is a plain mid-clip advance: frames move by one,
    # nothing rematerializes, and the request flag is cleared before physics.
    before = motion.steps[:, 0].copy()
    state = env.step(np.zeros((2, len(cfg.commands.motion.joint_names)), dtype=np.float32))
    assert not state.done.any()
    np.testing.assert_array_equal(motion.steps[:, 0], before + 1)
    assert not env._sim_reset_requested.any()

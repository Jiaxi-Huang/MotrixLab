# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from motrix_env_core.mdp.action import JointPositionActionCfg
from motrix_envs.locomotion.sonic import mdp
from motrix_envs.locomotion.sonic.cfg import SonicActionsCfg


def _identity_quaternions(shape: tuple[int, ...]) -> np.ndarray:
    quaternions = np.zeros((*shape, 4), dtype=np.float32)
    quaternions[..., 3] = 1.0
    return quaternions


def test_sonic_config_defaults_match_upstream_base() -> None:
    motion = mdp.SonicMotionCommandCfg()

    # Upstream sonic_release tracking set: a torso virtual point 0.5 m up
    # plus both wrists, no offsets.
    assert motion.reward_point_body_names == (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert motion.reward_point_body_offsets == (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    # The sonic task reuses the shared joint position action term; the env wires
    # the SONIC policy joint order into it at post-init.
    assert isinstance(SonicActionsCfg().joint_position, JointPositionActionCfg)
    assert motion.reference_body_name == "pelvis"
    assert motion.encoder_sampling == "mixed"


def test_sonic_model_hip_pitch_release_gains() -> None:
    # Regression: the model file once carried the generic G1's 88 Nm hip-pitch
    # force clamp and smaller armature. The action term now derives its scales
    # from these very gains, so pin the release contract in g1_sonic.xml.
    from motrix_envs.locomotion.sonic.g1 import make_g1_sonic_cfg

    root = ElementTree.parse(make_g1_sonic_cfg().scene.objs.robot.model.file).getroot()
    joint = root.find(".//default[@class='hip_pitch']/joint")
    kp, kd, effort, armature = 99.098427777, 6.308801854, 139.0, 0.025101925
    for side in ("left", "right"):
        actuator = root.find(f"actuator/position[@name='{side}_hip_pitch_joint']")
        assert float(actuator.attrib["kp"]) == pytest.approx(kp)
        assert float(actuator.attrib["kv"]) == pytest.approx(kd)
        assert tuple(map(float, actuator.attrib["forcerange"].split())) == (-effort, effort)
        assert tuple(map(float, joint.attrib["actuatorfrcrange"].split())) == (-effort, effort)
        assert float(joint.attrib["armature"]) == pytest.approx(armature)


def test_sonic_reward_point_config_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        mdp.SonicMotionCommandCfg(
            reward_point_body_names=("torso_link",),
            reward_point_body_offsets=(),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        mdp.SonicTrackingRewardCfg(weight=2.0, std=0.0)
    with pytest.raises(ValueError, match="encoder_sampling"):
        mdp.SonicMotionCommandCfg(encoder_sampling="both")


def test_sonic_encoder_sampling_mode_selection() -> None:
    def sampled(mode: str, draws: int) -> set[int]:
        mode_id = mdp.ENCODER_SAMPLING_MODES.index(mode)
        return {
            int(mdp._sample_encoder_slot(np.asarray([seed], dtype=np.uint64), np.int64(mode_id)))
            for seed in range(draws)
        }

    # Slot order mirrors upstream encoder_sample_probs: [g1, teleop, smpl].
    assert sampled("g1", 8) == {0}
    assert sampled("teleop", 8) == {1}
    assert sampled("smpl", 8) == {2}
    # Mixed draws uniformly over the three reference streams.
    assert sampled("mixed", 64) == {0, 1, 2}


def test_sonic_reset_expands_smpl_native_rows_multi_hot() -> None:
    joints = len(mdp.G1_SONIC_JOINTS)
    clip = SimpleNamespace(
        joint_pos=np.zeros((4, joints), np.float32),
        joint_vel=np.zeros((4, joints), np.float32),
        frame_clip_end=np.full(4, 3, np.int64),
        reference_body_pos_w=np.zeros((4, 3), np.float32),
    )
    motion = SimpleNamespace(
        clip=clip,
        steps=np.zeros(1, np.int64),
        sampling_cdf=np.ones(1, np.float32),
        start_at_timestep_zero_prob=np.float32(1.0),
        foot_joint_policy_indices=np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], np.int64),
        previous_foot_joint_velocity=np.zeros(4, np.float32),
        foot_joint_acceleration=np.zeros(4, np.float32),
        encoder_index=np.zeros(mdp.SONIC_ENCODER_COUNT, np.float32),
        encoder_sampling_mode=np.int64(1),
        running_ref_height=np.zeros(1, np.float32),
        low_reference=np.zeros(1, np.bool_),
        low_reference_height_threshold=np.float32(0.5),
    )
    ctx = SimpleNamespace(rand=SimpleNamespace(state=np.zeros(1, np.uint64)))

    def observed(mode: str, draws: int) -> set[tuple[float, ...]]:
        motion.encoder_sampling_mode = np.int64(mdp.ENCODER_SAMPLING_MODES.index(mode))
        rows = set()
        for seed in range(draws):
            ctx.rand.state[0] = seed
            mdp.SonicMotionCommand.reset_env(motion, ctx)
            rows.add(tuple(float(value) for value in motion.encoder_index))
        return rows

    # Upstream legacy multi-hot: only the smpl-native row expands — g1 is
    # always co-activated and teleop with probability 0.5, producing the
    # tri-hot rows the g1-teleop / teleop-smpl alignment losses train on.
    assert observed("g1", 8) == {(1.0, 0.0, 0.0)}
    assert observed("teleop", 8) == {(0.0, 1.0, 0.0)}
    assert observed("smpl", 64) == {(1.0, 0.0, 1.0), (1.0, 1.0, 1.0)}


def test_sonic_adaptive_sampling_changes_probability_and_draws() -> None:
    from motrix_env_core.numba.manager.rand import initialize_rand_states, next_uniform

    baseline = mdp._adaptive_sampling_probabilities(
        np.zeros(4, dtype=np.float32),
        np.float32(0.2),
        np.int64(1),
        np.float32(0.8),
    )
    failed = np.zeros(4, dtype=np.float32)
    failed[2] = 10.0
    focused = mdp._adaptive_sampling_probabilities(
        failed,
        np.float32(0.2),
        np.int64(1),
        np.float32(0.8),
    )

    np.testing.assert_allclose(baseline.sum(), 1.0)
    np.testing.assert_allclose(focused.sum(), 1.0)
    assert np.all(focused > 0.0)
    assert np.argmax(focused) == 2
    assert focused[2] > baseline[2]

    cdf = np.cumsum(focused, dtype=np.float32)
    cdf[-1] = 1.0
    states = initialize_rand_states(256, 7)
    sampled_bins = []
    for state in states:
        unit = (next_uniform(state) + np.float32(1.0)) * np.float32(0.5)
        sampled_bins.append(min(int(np.searchsorted(cdf, unit, side="left")), 3))
    counts = np.bincount(sampled_bins, minlength=4)
    assert counts[2] > counts[[0, 1, 3]].max()


def test_sonic_cdf_sampling_draws_follow_probabilities() -> None:
    from motrix_env_core.numba.manager.rand import initialize_rand_states

    cdf = np.cumsum(np.asarray([0.1, 0.1, 0.6, 0.2], np.float32))
    cdf[-1] = 1.0
    states = initialize_rand_states(256, 11)
    sampled = [int(mdp._sample_motion_step_from_cdf(state, cdf, np.int64(200), np.float32(0.0))) for state in states]

    counts = np.bincount(np.asarray(sampled) * 4 // 200, minlength=4)
    assert counts[2] > counts[[0, 1, 3]].max()
    assert min(sampled) >= 0 and max(sampled) <= 198

    zero_states = initialize_rand_states(8, 12)
    assert all(
        mdp._sample_motion_step_from_cdf(state, cdf, np.int64(200), np.float32(1.0)) == 0 for state in zero_states
    )


def test_sonic_advance_switches_segment_at_clip_end() -> None:
    from motrix_env_core.numba.manager.rand import initialize_rand_states

    motion = SimpleNamespace(
        steps=np.asarray((5,), np.int64),
        start_at_timestep_zero_prob=np.float32(0.0),
        sampling_cdf=np.linspace(0.1, 1.0, 10).astype(np.float32),
        clip=SimpleNamespace(
            joint_pos=np.zeros((10, 2), np.float32),
            frame_clip_end=np.asarray([3, 3, 3, 3, 7, 7, 7, 7, 9, 9], np.int64),
        ),
    )
    ctx = SimpleNamespace(
        rand=SimpleNamespace(state=initialize_rand_states(1, 5)[0]),
        sim_reset_requested=np.zeros(1, dtype=bool),
    )

    # Mid-clip: a plain advance, no rematerialization request.
    mdp.SonicMotionCommand.advance(motion, ctx)
    assert motion.steps[0] == 6
    assert not ctx.sim_reset_requested[0]

    # From the clip's final frame the next advance switches segments and asks
    # for a sim-only rematerialization; the episode itself keeps running.
    motion.steps[0] = 7
    mdp.SonicMotionCommand.advance(motion, ctx)
    assert ctx.sim_reset_requested[0]
    assert 0 <= motion.steps[0] <= 8


def test_sonic_for_play_pins_g1_encoder_sampling() -> None:
    from motrix_envs.locomotion.sonic.g1 import make_g1_sonic_cfg

    assert make_g1_sonic_cfg().for_play().commands.motion.encoder_sampling == "g1"


def test_sonic_undesired_contacts_exclude_only_end_effectors() -> None:
    from motrix_env_core.sim import BodyLinkNetContactForceQuery
    from motrix_envs.locomotion.sonic.g1 import make_g1_sonic_cfg

    query = make_g1_sonic_cfg().queries.data["undesired_contact_forces"]

    assert isinstance(query, BodyLinkNetContactForceQuery)
    assert query.body == "pelvis"
    # Upstream undesired_contacts regex: every robot body except the six
    # end-effector links (ankle rolls, wrist yaws, elbows).
    assert set(query.exclude_links) == {
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    }


def test_sonic_low_reference_relaxes_z_terminations() -> None:
    bodies = len(mdp.G1_SONIC_BODY_NAMES)
    clip = SimpleNamespace(
        root_body_pos_w=np.asarray([[0.0, 0.0, 0.4]], np.float32),
        reference_body_pos_w=np.asarray([[0.0, 0.0, 0.4]], np.float32),
        reference_body_quat_w=_identity_quaternions((1,)),
        tracked_bodies_pos_w=np.zeros((1, bodies, 3), np.float32),
        tracked_bodies_quat_w=_identity_quaternions((1, bodies)),
        frame_clip_end=np.asarray((0,), np.int64),
    )
    clip.tracked_bodies_pos_w[0, 3, 2] = 0.4  # end-effector z error of 0.4 against the robot
    motion = SimpleNamespace(
        clip=clip,
        steps=np.asarray((0,), np.int64),
        reference_index=0,
        foot_joint_policy_indices=np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], np.int64),
        previous_foot_joint_velocity=np.zeros(4, np.float32),
        foot_joint_acceleration=np.zeros(4, np.float32),
        step_dt=np.float32(0.02),
        target_body_position_relative=np.zeros((bodies, 3), np.float32),
        target_body_orientation_relative=_identity_quaternions((bodies,)),
        running_ref_height=np.full((1,), 0.4, np.float32),
        low_reference=np.asarray((False,)),
        low_reference_height_threshold=np.float32(0.5),
    )
    ctx = SimpleNamespace(
        commands={"motion": motion},
        sim={
            "robot_dof_vel": np.zeros(len(mdp.G1_SONIC_JOINTS), np.float32),
            "tracked_body_pos": np.zeros((bodies, 3), np.float32),
            "tracked_body_quat": _identity_quaternions((bodies,)),
        },
    )
    mdp.SonicMotionCommand.update(motion, ctx)
    assert motion.low_reference[0]

    anchor_state = mdp.SonicAnchorPosTermination(np.float32(0.15), np.float32(0.75))
    ee_state = mdp.SonicEeBodyPosTermination(np.float32(0.15), np.float32(0.75))
    assert not mdp.SonicAnchorPosTermination.compute(ctx, anchor_state)
    assert not mdp.SonicEeBodyPosTermination.compute(ctx, ee_state)

    motion.low_reference[0] = False
    assert mdp.SonicAnchorPosTermination.compute(ctx, anchor_state)
    assert mdp.SonicEeBodyPosTermination.compute(ctx, ee_state)


def test_sonic_reference_advances_only_after_transition_evaluation() -> None:
    bodies = len(mdp.G1_SONIC_BODY_NAMES)
    clip = SimpleNamespace(
        frame_clip_end=np.asarray((1, 1), np.int64),
        reference_body_pos_w=np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 2.0)), np.float32),
        reference_body_quat_w=_identity_quaternions((2,)),
        tracked_bodies_pos_w=np.zeros((2, bodies, 3), np.float32),
        tracked_bodies_quat_w=_identity_quaternions((2, bodies)),
    )
    clip.tracked_bodies_pos_w[1, :, 2] = 1.0
    motion = SimpleNamespace(
        clip=clip,
        steps=np.asarray((0,), np.int64),
        reference_index=np.int64(0),
        foot_joint_policy_indices=np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], np.int64),
        previous_foot_joint_velocity=np.zeros(4, np.float32),
        foot_joint_acceleration=np.zeros(4, np.float32),
        step_dt=np.float32(0.02),
        target_body_position_relative=np.zeros((bodies, 3), np.float32),
        target_body_orientation_relative=_identity_quaternions((bodies,)),
        running_ref_height=np.asarray((1.0,), np.float32),
        low_reference=np.asarray((False,)),
        low_reference_height_threshold=np.float32(0.5),
    )
    ctx = SimpleNamespace(
        commands={"motion": motion},
        sim={
            "robot_dof_vel": np.zeros(len(mdp.G1_SONIC_JOINTS), np.float32),
            "tracked_body_pos": np.zeros((bodies, 3), np.float32),
            "tracked_body_quat": _identity_quaternions((bodies,)),
        },
    )

    mdp.SonicMotionCommand.update(motion, ctx)

    assert motion.steps[0] == 0
    np.testing.assert_array_equal(motion.target_body_position_relative[:, 2], 0.0)

    mdp.SonicMotionCommand.advance(motion, ctx)
    assert motion.steps[0] == 1


def test_sonic_actor_observation_uses_lane_local_action_width() -> None:
    joint_count = len(mdp.G1_SONIC_JOINTS)
    history_length = 2
    frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    term = mdp.SonicActorObservation(
        history=np.zeros((2, history_length, frame_dim), dtype=np.float32),
        seen_reset=np.full(2, -1, dtype=np.int64),
    )
    action = SimpleNamespace(
        action_queue=np.arange(joint_count, dtype=np.float32).reshape(1, joint_count),
        action_ptr=np.zeros(1, dtype=np.int64),
        action_scales=np.ones(joint_count, dtype=np.float32),
        default_angles=np.zeros(joint_count, dtype=np.float32),
    )
    tracked_body_quat = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    context = SimpleNamespace(
        env_id=np.int64(1),
        actions={"joint_position": action},
        commands={"motion": SimpleNamespace(reset_counter=np.asarray((0,), dtype=np.int64))},
        rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
        sim={
            "tracked_body_quat": tracked_body_quat,
            "tracked_body_angular_velocity": np.zeros((len(mdp.G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
            "robot_dof_pos": np.zeros(joint_count, dtype=np.float32),
            "robot_dof_vel": np.zeros(joint_count, dtype=np.float32),
        },
    )
    out = np.empty(history_length * frame_dim, dtype=np.float32)
    no_noise = np.zeros_like(out)

    mdp.SonicActorObservation.compute(context, out, term, no_noise)

    assert term.seen_reset.tolist() == [-1, 0]
    action_offset = history_length * (mdp.SONIC_VECTOR_DIM + 2 * joint_count)
    current = action.action_queue[action.action_ptr[0]]
    np.testing.assert_array_equal(out[action_offset : action_offset + joint_count], current)
    np.testing.assert_array_equal(out[action_offset + joint_count : action_offset + 2 * joint_count], current)

    previous_action = current.copy()
    action.action_queue[action.action_ptr[0]] += 100.0
    mdp.SonicActorObservation.compute(context, out, term, no_noise)
    np.testing.assert_array_equal(out[action_offset : action_offset + joint_count], previous_action)
    np.testing.assert_array_equal(
        out[action_offset + joint_count : action_offset + 2 * joint_count],
        action.action_queue[action.action_ptr[0]],
    )


def test_sonic_observation_noise_amplitudes_match_upstream() -> None:
    """Per-dim noise layouts pin the upstream corruption contract.

    Upstream noises (local_dir_hist + unitoken_all_noz): proprio av ±0.2 /
    joint pos ±0.01 / joint vel ±0.5 / gravity ±0.05 with the actions block
    clean; the g1 reference rotation tail ±0.05; the SMPL body blocks ±0.05
    with the wrist tail clean; the hybrid anchor orientation ±0.05 with the
    lower-body / vr blocks clean.
    """
    history, joints, frames = 2, 29, 5
    actor = mdp._actor_observation_noise(history, joints)
    assert actor.shape == (history * (3 * joints + 6),)
    np.testing.assert_array_equal(actor[: 3 * history], np.float32(mdp.SONIC_ANG_VEL_NOISE))
    np.testing.assert_array_equal(actor[3 * history : (3 + joints) * history], np.float32(mdp.SONIC_JOINT_POS_NOISE))
    np.testing.assert_array_equal(
        actor[(3 + joints) * history : (3 + 2 * joints) * history], np.float32(mdp.SONIC_JOINT_VEL_NOISE)
    )
    np.testing.assert_array_equal(actor[(3 + 2 * joints) * history : (3 + 3 * joints) * history], 0.0)
    np.testing.assert_array_equal(actor[(3 + 3 * joints) * history :], np.float32(mdp.SONIC_GRAVITY_NOISE))

    g1 = mdp._g1_reference_noise(frames, joints)
    assert g1.shape == (frames * (2 * joints + 6),)
    np.testing.assert_array_equal(g1[: 2 * frames * joints], 0.0)
    np.testing.assert_array_equal(g1[2 * frames * joints :], np.float32(mdp.SONIC_REFERENCE_NOISE))

    smpl = mdp._smpl_reference_noise(frames, 24)
    assert smpl.shape == (frames * (24 * 3 + 6 + 6),)
    np.testing.assert_array_equal(smpl[: frames * (24 * 3 + 6)], np.float32(mdp.SONIC_REFERENCE_NOISE))
    np.testing.assert_array_equal(smpl[frames * (24 * 3 + 6) :], 0.0)

    hybrid = mdp._hybrid_reference_noise(frames * 24 + 27)
    assert hybrid.shape == (frames * 24 + 27,)
    np.testing.assert_array_equal(hybrid[:-6], 0.0)
    np.testing.assert_array_equal(hybrid[-6:], np.float32(mdp.SONIC_REFERENCE_NOISE))


def test_sonic_actor_observation_applies_per_dim_uniform_noise() -> None:
    """The proprio kernel applies the per-dim amplitudes; zero-dim blocks stay exact."""
    joint_count = 3
    history_length = 2
    frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    term = mdp.SonicActorObservation(
        history=np.zeros((1, history_length, frame_dim), dtype=np.float32),
        seen_reset=np.full(1, -1, dtype=np.int64),
    )
    action = SimpleNamespace(
        action_queue=np.zeros((1, joint_count), dtype=np.float32),
        action_ptr=np.zeros(1, dtype=np.int64),
        action_scales=np.ones(joint_count, dtype=np.float32),
        default_angles=np.zeros(joint_count, dtype=np.float32),
    )
    context = SimpleNamespace(
        env_id=np.int64(0),
        actions={"joint_position": action},
        commands={"motion": SimpleNamespace(reset_counter=np.asarray((0,), dtype=np.int64))},
        rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
        sim={
            "tracked_body_quat": _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),)),
            "tracked_body_angular_velocity": np.zeros((len(mdp.G1_SONIC_BODY_NAMES), 3), dtype=np.float32),
            "robot_dof_pos": np.zeros(joint_count, dtype=np.float32),
            "robot_dof_vel": np.zeros(joint_count, dtype=np.float32),
        },
    )
    # Constant sim inputs: after the first call every history row holds the
    # same frame, so the clean output is stable across calls.
    clean = np.empty(history_length * frame_dim, dtype=np.float32)
    mdp.SonicActorObservation.compute(context, clean, term, np.zeros_like(clean))
    amplitudes = mdp._actor_observation_noise(history_length, joint_count)

    samples = 256
    deltas = np.empty((samples, clean.shape[0]), dtype=np.float64)
    for sample in range(samples):
        noised = np.empty_like(clean)
        mdp.SonicActorObservation.compute(context, noised, term, amplitudes)
        deltas[sample] = noised - clean

    # Uniform noise of amplitude a has std a/sqrt(3); 256 samples give a ~5%
    # estimator error, so a 25% band is a robust amplitude check.
    empirical = deltas.std(axis=0)
    expected = amplitudes.astype(np.float64) / np.sqrt(3.0)
    noisy = expected > 0.0
    np.testing.assert_allclose(empirical[noisy], expected[noisy], rtol=0.25)
    # Zero-amplitude dims (the actions block) stay bit-exact.
    np.testing.assert_array_equal(deltas[:, ~noisy], 0.0)


def test_sonic_actor_observation_blocks_match_release_field_order() -> None:
    """Proprio blocks follow the upstream ``PolicyCfg`` field order, gravity last.

    The release runtime order is IsaacLab's ``ObservationManager`` iterating the
    ``PolicyCfg`` dataclass fields — ``base_ang_vel, joint_pos, joint_vel,
    actions, gravity_dir`` (upstream's own ``PolicyAtmCfg`` states this) — not
    the yaml ``defaults:`` merge order, which lists gravity first. The verified
    release bridge in ``scripts/private/play_gear_sonic_release.py`` feeds
    upstream weights ``[av 3n | jp jn | jv jn | act jn | grav 3n]``; this test
    pins that exact feature-major block layout (per-term blocks, frames
    oldest-to-newest inside each block).
    """
    joint_count = 3
    history_length = 2
    frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    term = mdp.SonicActorObservation(
        history=np.zeros((1, history_length, frame_dim), dtype=np.float32),
        seen_reset=np.full(1, -1, dtype=np.int64),
    )
    body_count = len(mdp.G1_SONIC_BODY_NAMES)
    root_index = mdp.SONIC_ROOT_BODY_INDEX
    angular_velocity = np.zeros((body_count, 3), dtype=np.float32)
    angular_velocity[root_index] = (1.0, 2.0, 3.0)
    dof_pos = np.arange(joint_count, dtype=np.float32) * 1.0 + 10.0
    dof_vel = np.arange(joint_count, dtype=np.float32) * 1.0 + 20.0
    current_action = np.arange(joint_count, dtype=np.float32) * 1.0 + 30.0
    action = SimpleNamespace(
        action_queue=current_action.reshape(1, joint_count),
        action_ptr=np.zeros(1, dtype=np.int64),
        action_scales=np.ones(joint_count, dtype=np.float32),
        default_angles=np.zeros(joint_count, dtype=np.float32),
    )
    context = SimpleNamespace(
        env_id=np.int64(0),
        actions={"joint_position": action},
        commands={"motion": SimpleNamespace(reset_counter=np.asarray((0,), dtype=np.int64))},
        rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
        sim={
            "tracked_body_quat": _identity_quaternions((body_count,)),
            "tracked_body_angular_velocity": angular_velocity,
            "robot_dof_pos": dof_pos,
            "robot_dof_vel": dof_vel,
        },
    )
    out = np.empty(history_length * frame_dim, dtype=np.float32)

    mdp.SonicActorObservation.compute(context, out, term, np.zeros_like(out))

    def block(values: np.ndarray) -> np.ndarray:
        # Feature-major: the whole per-term block repeats across history rows.
        return np.tile(values, history_length)

    expected = np.concatenate(
        (
            block(angular_velocity[root_index]),
            block(dof_pos),
            block(dof_vel),
            block(current_action),
            block(np.asarray((0.0, 0.0, -1.0), dtype=np.float32)),
        )
    )
    np.testing.assert_allclose(out, expected, atol=1.0e-6)


def test_sonic_g1_reference_matches_release_term_layout_and_pelvis_frame() -> None:
    joint_count = 2
    num_future_frames = 2
    frame_count = 6
    positions = np.arange(frame_count * joint_count, dtype=np.float32).reshape(frame_count, joint_count)
    velocities = positions + 100.0
    target_quaternions = _identity_quaternions((frame_count, len(mdp.G1_SONIC_BODY_NAMES)))
    quarter_turn_z = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    target_quaternions[:, mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    tracked_quaternions = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    tracked_quaternions[mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    # Keep torso different so the test detects accidentally using the old reference body.
    torso = mdp.G1_SONIC_BODY_NAMES.index("torso_link")
    tracked_quaternions[torso] = (0.0, 0.0, 0.0, 1.0)
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(num_future_frames),
        clip=SimpleNamespace(
            joint_pos=positions,
            frame_clip_end=np.full(frame_count, frame_count - 1, np.int64),
            joint_vel=velocities,
            tracked_bodies_quat_w=target_quaternions,
        ),
    )
    output = np.empty(num_future_frames * (2 * joint_count + 6), dtype=np.float32)

    mdp.SonicG1ReferenceObservation.compute(
        SimpleNamespace(
            commands={"motion": motion},
            rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
            sim={"tracked_body_quat": tracked_quaternions},
        ),
        output,
        np.zeros_like(output),
    )

    selected = np.asarray((0, 5), dtype=np.int64)
    identity_rotation6 = np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    expected = np.concatenate(
        (
            positions[selected].reshape(-1),
            velocities[selected].reshape(-1),
            np.tile(identity_rotation6, num_future_frames),
        )
    )
    np.testing.assert_allclose(output, expected, atol=1.0e-6)


def test_sonic_smpl_reference_matches_release_local_frame_and_wrist_tail() -> None:
    num_future_frames = 2
    smpl_joint_count = 24
    policy_joint_count = len(mdp.G1_SONIC_JOINTS)
    quarter_turn_z = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    tracked_quaternions = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    tracked_quaternions[mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    human_quaternions = np.tile(quarter_turn_z, (num_future_frames, 1))
    smpl_joints = np.zeros((num_future_frames, smpl_joint_count, 3), dtype=np.float32)
    smpl_joints[..., 0] = 1.0
    joint_positions = np.arange(num_future_frames * policy_joint_count, dtype=np.float32).reshape(
        num_future_frames, policy_joint_count
    )
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(num_future_frames),
        clip=SimpleNamespace(
            joint_pos=joint_positions,
            frame_clip_end=np.full(num_future_frames, num_future_frames - 1, np.int64),
            smpl_joints=smpl_joints,
            smpl_root_quat=human_quaternions,
        ),
    )
    output = np.empty(num_future_frames * (smpl_joint_count * 3 + 6 + 6), dtype=np.float32)

    mdp.SonicSmplReferenceObservation.compute(
        SimpleNamespace(
            commands={"motion": motion},
            rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
            sim={"tracked_body_quat": tracked_quaternions},
        ),
        output,
        np.zeros_like(output),
    )

    local_joints = np.zeros_like(smpl_joints)
    local_joints[..., 1] = -1.0
    identity_rotation6 = np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    human_features = np.concatenate(
        (
            local_joints.reshape(num_future_frames, -1),
            np.tile(identity_rotation6, (num_future_frames, 1)),
        ),
        axis=-1,
    )
    wrists = joint_positions[:, mdp.SONIC_WRIST_POLICY_INDICES]
    expected = np.concatenate((human_features.reshape(-1), wrists.reshape(-1)))
    np.testing.assert_allclose(output, expected, atol=1.0e-6)


def test_sonic_hybrid_reference_matches_release_layout() -> None:
    num_future_frames = 2
    frame_count = 6
    joints = len(mdp.G1_SONIC_JOINTS)
    bodies = len(mdp.G1_SONIC_BODY_NAMES)
    positions = np.arange(frame_count * joints, dtype=np.float32).reshape(frame_count, joints)
    velocities = positions + 100.0
    body_pos = np.zeros((frame_count, bodies, 3), np.float32)
    body_pos[:, mdp.SONIC_ROOT_BODY_INDEX] = (0.5, -0.25, 0.8)
    vr_body_pos = ((1.2, 0.3, 0.9), (-0.4, 0.6, 1.1), (0.1, 0.2, 1.25))
    for body, xyz in zip(mdp.HYBRID_VR_BODY_INDICES, vr_body_pos):
        body_pos[:, body] = xyz
    body_quat = _identity_quaternions((frame_count, bodies))
    quarter_turn_z = np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), dtype=np.float32)
    body_quat[:, mdp.SONIC_ROOT_BODY_INDEX] = quarter_turn_z
    tracked_quaternions = _identity_quaternions((bodies,))
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(num_future_frames),
        clip=SimpleNamespace(
            joint_pos=positions,
            joint_vel=velocities,
            frame_clip_end=np.full(frame_count, frame_count - 1, np.int64),
            tracked_bodies_pos_w=body_pos,
            tracked_bodies_quat_w=body_quat,
        ),
    )
    output = np.empty(num_future_frames * 24 + 27, dtype=np.float32)

    mdp.SonicHybridReferenceObservation.compute(
        SimpleNamespace(
            commands={"motion": motion},
            rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
            sim={"tracked_body_quat": tracked_quaternions},
        ),
        output,
        np.zeros_like(output),
    )

    # Lower-body joint order: the six left-leg joints first, then the six
    # right-leg ones, sampled on the G1 stride grid; positions of all frames
    # first, then velocities. Columns resolve by name from the declared order.
    selected = np.asarray((0, mdp.G1_FUTURE_STRIDE), dtype=np.int64)
    leg_columns = np.asarray([mdp.G1_SONIC_JOINTS.index(name) for name in mdp.HYBRID_LEG_JOINTS], dtype=np.int64)
    expected_lower = np.concatenate(
        (
            positions[selected][:, leg_columns].reshape(-1),
            velocities[selected][:, leg_columns].reshape(-1),
        )
    )
    # Current-frame vr targets in the reference-anchor local frame: the anchor
    # is a quarter turn about z, so its inverse maps (x, y, z) -> (y, -x, z).
    anchor = body_pos[0, mdp.SONIC_ROOT_BODY_INDEX]
    vr_positions = []
    for body, offset in zip(mdp.HYBRID_VR_BODY_INDICES, mdp.HYBRID_VR_BODY_OFFSETS):
        target = body_pos[0, body] + np.asarray(offset, np.float32) - anchor
        vr_positions.extend((target[1], -target[0], target[2]))
    inverse_anchor_quat = np.asarray((0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)), np.float32)
    # First-two-columns 6D of Rz(90 deg), flattened row-major: (R00, R01,
    # R10, R11, R20, R21).
    anchor_rotation6 = np.asarray((0.0, -1.0, 1.0, 0.0, 0.0, 0.0), np.float32)
    expected = np.concatenate(
        (
            expected_lower,
            np.asarray(vr_positions, np.float32),
            np.tile(inverse_anchor_quat, 3),
            anchor_rotation6,
        )
    )
    np.testing.assert_allclose(output, expected, atol=1.0e-6)


def test_sonic_observation_sizes_are_derived_from_runtime_shapes() -> None:
    num_future_frames = np.int64(2)
    joint_count = 7
    body_count = 5
    smpl_joint_count = 24
    motion = SimpleNamespace(
        num_future_frames=num_future_frames,
        clip=SimpleNamespace(
            joint_pos=np.empty((3, joint_count), dtype=np.float32),
            tracked_bodies_pos_w=np.empty((3, body_count, mdp.SONIC_VECTOR_DIM), dtype=np.float32),
            smpl_joints=np.empty((3, smpl_joint_count, mdp.SONIC_VECTOR_DIM), dtype=np.float32),
        ),
    )
    env = SimpleNamespace(
        num_envs=4,
        model=SimpleNamespace(
            bodies={"robot": SimpleNamespace(joint_names=tuple(f"joint_{i}" for i in range(joint_count)))}
        ),
        command_terms={"motion": motion},
    )

    actor_frame_dim = mdp.SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * mdp.SONIC_VECTOR_DIM
    num_history_frames = 3
    assert mdp.SonicActorObservationCfg(num_history_frames=num_history_frames)(env).size == (
        num_history_frames * actor_frame_dim
    )
    assert mdp.SonicG1ReferenceObservationCfg()(env).size == num_future_frames * (
        2 * joint_count + mdp.SONIC_ROTATION_REPRESENTATION_DIM
    )
    assert mdp.SonicSmplReferenceObservationCfg()(env).size == num_future_frames * (
        smpl_joint_count * mdp.SONIC_VECTOR_DIM
        + mdp.SONIC_ROTATION_REPRESENTATION_DIM
        + len(mdp.SONIC_WRIST_POLICY_INDICES)
    )
    # Upstream teleop layout: lower-body block per future frame plus the
    # static current-frame tail (3 positions + 3 quaternions + anchor 6D).
    assert mdp.SonicHybridReferenceObservationCfg()(env).size == num_future_frames * 24 + 27
    # Merged gear_sonic/MimicLite contract: per-future-frame joints plus the
    # ref-root channel (3 + 6), body-diff terms (15 per body per diff step),
    # then the proprioception history frames.
    critic_history_frames = 4
    assert mdp.SonicCriticObservationCfg(num_history_frames=critic_history_frames)(env).size == (
        num_future_frames * (2 * joint_count + 9)
        + len(mdp.SONIC_CRITIC_DIFF_STEPS) * body_count * 15
        + critic_history_frames * actor_frame_dim
    )


def test_sonic_anti_shake_uses_three_dimensional_thresholded_norm() -> None:
    velocity = np.zeros((3, 3), dtype=np.float32)
    velocity[0] = (1.2, 0.9, 0.0)  # norm is exactly the threshold
    velocity[1] = (2.0, 0.0, 0.0)
    velocity[2] = (0.0, 0.0, 2.0)
    term = mdp.SonicAntiShakeReward(np.asarray((0, 1, 2), dtype=np.int64))

    actual = mdp.SonicAntiShakeReward.compute(
        SimpleNamespace(sim={"tracked_body_angular_velocity": velocity}),
        term,
    )

    assert actual == pytest.approx((0.0 + 0.25 + 0.25) / 3.0)


def test_sonic_anti_shake_cfg_tracks_only_wrists() -> None:
    tracked = mdp.G1_SONIC_BODY_NAMES
    motion = SimpleNamespace(tracked_body_names=tracked)
    env = SimpleNamespace(cfg=SimpleNamespace(commands=SimpleNamespace(motion=motion)))

    term = mdp.SonicAntiShakeRewardCfg(weight=-0.005)(env)

    expected = np.asarray(
        [tracked.index("left_wrist_yaw_link"), tracked.index("right_wrist_yaw_link")],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(term.args[0].body_indices, expected)


def test_sonic_anchor_orientation_threshold_uses_squared_angle_error() -> None:
    target = _identity_quaternions((1,))
    tracked = _identity_quaternions((len(mdp.G1_SONIC_BODY_NAMES),))
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        reference_index=np.int64(mdp.SONIC_ROOT_BODY_INDEX),
        clip=SimpleNamespace(root_body_quat_w=target),
    )
    context = SimpleNamespace(
        commands={"motion": motion},
        sim={"tracked_body_quat": tracked},
    )
    term = mdp.SonicAnchorOriTermination(np.float32(0.2))

    tracked[mdp.SONIC_ROOT_BODY_INDEX] = (np.sin(0.15), 0.0, 0.0, np.cos(0.15))
    assert not mdp.SonicAnchorOriTermination.compute(context, term)

    tracked[mdp.SONIC_ROOT_BODY_INDEX] = (np.sin(0.25), 0.0, 0.0, np.cos(0.25))
    assert mdp.SonicAnchorOriTermination.compute(context, term)


def test_sonic_feet_acc_matches_upstream_sum_of_squared_acceleration() -> None:
    motion = SimpleNamespace(foot_joint_acceleration=np.asarray((1.0, 2.0, 3.0, 4.0), dtype=np.float32))
    context = SimpleNamespace(
        commands={"motion": motion},
    )

    actual = mdp.SonicFeetAccelerationReward.compute(context)

    assert actual == pytest.approx(30.0)


def test_sonic_foot_acceleration_history_is_task_local() -> None:
    foot_indices = np.asarray([mdp.G1_SONIC_JOINTS.index(n) for n in mdp.SONIC_FOOT_JOINTS], dtype=np.int64)
    velocity = np.zeros(len(mdp.G1_SONIC_JOINTS), dtype=np.float32)
    velocity[foot_indices] = (0.02, 0.04, 0.06, 0.08)
    previous = np.zeros(4, dtype=np.float32)
    acceleration = np.zeros(4, dtype=np.float32)

    mdp._update_foot_joint_acceleration(velocity, foot_indices, previous, acceleration, np.float32(0.02))

    np.testing.assert_allclose(acceleration, (1.0, 2.0, 3.0, 4.0))
    np.testing.assert_array_equal(previous, velocity[foot_indices])


def test_sonic_tracking_reward_uses_configured_anchor_local_points() -> None:
    body_count = len(mdp.G1_SONIC_BODY_NAMES)
    target_position = np.zeros((1, body_count, 3), dtype=np.float32)
    target_quaternion = _identity_quaternions((1, body_count))
    robot_position = np.zeros((body_count, 3), dtype=np.float32)
    robot_quaternion = _identity_quaternions((body_count,))
    torso = mdp.G1_SONIC_BODY_NAMES.index("torso_link")
    left_wrist = mdp.G1_SONIC_BODY_NAMES.index("left_wrist_yaw_link")
    right_wrist = mdp.G1_SONIC_BODY_NAMES.index("right_wrist_yaw_link")
    robot_position[left_wrist, 0] = 0.1
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        reference_index=np.int64(torso),
        clip=SimpleNamespace(
            tracked_bodies_pos_w=target_position,
            tracked_bodies_quat_w=target_quaternion,
        ),
    )
    scratch = np.zeros((1, 3), dtype=np.float32)
    term = mdp.SonicTrackingReward(
        np.asarray((torso, left_wrist, right_wrist), dtype=np.int64),
        np.asarray(((0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float32),
        np.float32(0.1),
        scratch.copy(),
        scratch.copy(),
        scratch.copy(),
        scratch.copy(),
    )
    context = SimpleNamespace(
        env_id=np.int64(0),
        commands={"motion": motion},
        sim={"tracked_body_pos": robot_position, "tracked_body_quat": robot_quaternion},
    )

    actual = mdp.SonicTrackingReward.compute(context, term)

    assert actual == pytest.approx(np.exp(-1.0 / 3.0), rel=1.0e-6)


def test_sonic_clip_boundary_clamps_reference_frames() -> None:
    motion = SimpleNamespace(
        steps=np.asarray((1,), np.int64),
        num_future_frames=np.int64(2),
        clip=SimpleNamespace(
            frame_clip_end=np.asarray([1, 1, 3, 3]),
            joint_pos=np.asarray([[0.0], [1.0], [100.0], [101.0]], np.float32),
            joint_vel=np.zeros((4, 1), np.float32),
            tracked_bodies_quat_w=_identity_quaternions((4, 1)),
        ),
    )
    ctx = SimpleNamespace(
        commands={"motion": motion},
        rand=SimpleNamespace(state=np.zeros(1, np.uint64)),
        sim={"tracked_body_quat": _identity_quaternions((1,))},
    )
    out = np.empty(16, np.float32)
    mdp.SonicG1ReferenceObservation.compute(ctx, out, np.zeros_like(out))
    # Future reference frames clamp at the clip boundary instead of leaking
    # into the next clip.
    np.testing.assert_array_equal(out[:2], [1.0, 1.0])


def test_sonic_critic_observes_robot_state_and_resets_history() -> None:
    n, joints, bodies = 2, 2, 1
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(n),
        reference_index=np.int64(0),
        reset_counter=np.asarray([0]),
        clip=SimpleNamespace(
            frame_clip_end=np.asarray([1, 1]),
            joint_pos=np.zeros((2, joints), np.float32),
            joint_vel=np.zeros((2, joints), np.float32),
            reference_body_pos_w=np.zeros((2, 3), np.float32),
            reference_body_quat_w=_identity_quaternions((2,)),
            tracked_bodies_pos_w=np.zeros((2, bodies, 3), np.float32),
            tracked_bodies_quat_w=_identity_quaternions((2, bodies)),
            tracked_bodies_lin_vel_w=np.zeros((2, bodies, 3), np.float32),
            tracked_bodies_ang_vel_w=np.zeros((2, bodies, 3), np.float32),
        ),
    )
    sim = {
        "tracked_body_pos": np.zeros((bodies, 3), np.float32),
        "tracked_body_quat": _identity_quaternions((bodies,)),
        "tracked_body_linear_velocity": np.asarray([[1, 2, 3]], np.float32),
        "tracked_body_angular_velocity": np.zeros((bodies, 3), np.float32),
        "robot_dof_pos": np.zeros(joints, np.float32),
        "robot_dof_vel": np.zeros(joints, np.float32),
    }
    ctx = SimpleNamespace(
        env_id=0,
        commands={"motion": motion},
        sim=sim,
        actions={
            "joint_position": SimpleNamespace(
                action_queue=np.zeros((1, joints), np.float32),
                action_ptr=np.zeros(1, dtype=np.int64),
                action_scales=np.ones(joints, np.float32),
                default_angles=np.zeros(joints, np.float32),
            )
        },
    )
    term = mdp.SonicCriticObservation(np.zeros((2, n, 6 + 3 * joints), np.float32), np.full(2, -1, np.int64))
    out = np.full(n * (2 * joints + 9) + 2 * 15 * bodies + n * (6 + 3 * joints), np.nan, np.float32)
    mdp.SonicCriticObservation.compute(ctx, out, term)
    assert np.isfinite(out).all()
    history_start = n * (2 * joints + 9) + 2 * 15 * bodies
    np.testing.assert_array_equal(out[history_start : history_start + 6], [1, 2, 3, 1, 2, 3])
    sim["tracked_body_linear_velocity"][:] = 4
    mdp.SonicCriticObservation.compute(ctx, out, term)
    np.testing.assert_array_equal(out[history_start : history_start + 6], [1, 2, 3, 4, 4, 4])
    motion.reset_counter[0] += 1
    mdp.SonicCriticObservation.compute(ctx, out, term)
    np.testing.assert_array_equal(out[history_start : history_start + 6], [4, 4, 4, 4, 4, 4])
    np.testing.assert_array_equal(term.history[1], 0)


def _run_critic_observation(
    clip_body_pos,
    clip_body_quat,
    robot_body_pos,
    robot_body_quat,
    *,
    anchor_quat=None,
    root_pos=None,
    ref_root_pos=None,
):
    """Run ``SonicCriticObservation.compute`` on a one-env, two-body fixture.

    Body 0 is the pelvis anchor (static at the origin unless ``root_pos``
    moves the robot's), body 1 carries the caller's per-frame reference pose
    (leading axis: [frame, component]) and per-frame robot pose. Anchors take
    ``anchor_quat`` on both sides so the projected-yaw canonicalization is
    exercised. Returns the output vector plus context and term for in-place
    mutation and recompute.
    """
    joints, bodies, frames = 1, 2, 2
    identity = _identity_quaternions((frames, bodies))
    clip_bodies_quat = identity.copy()
    clip_bodies_quat[:, 1] = clip_body_quat
    robot_bodies_quat = _identity_quaternions((bodies,))
    robot_bodies_quat[1] = robot_body_quat
    clip_bodies_pos = np.zeros((frames, bodies, 3), np.float32)
    clip_bodies_pos[:, 1] = clip_body_pos
    robot_bodies_pos = np.zeros((bodies, 3), np.float32)
    robot_bodies_pos[1] = robot_body_pos
    if anchor_quat is not None:
        clip_bodies_quat[:, 0] = anchor_quat
        robot_bodies_quat[0] = anchor_quat
    motion = SimpleNamespace(
        steps=np.asarray((0,), np.int64),
        num_future_frames=np.int64(1),
        reference_index=np.int64(0),
        reset_counter=np.asarray([0]),
        clip=SimpleNamespace(
            frame_clip_end=np.asarray([1, 1]),
            joint_pos=np.zeros((frames, joints), np.float32),
            joint_vel=np.zeros((frames, joints), np.float32),
            reference_body_pos_w=np.zeros((frames, 3), np.float32),
            reference_body_quat_w=_identity_quaternions((frames,)),
            tracked_bodies_pos_w=clip_bodies_pos,
            tracked_bodies_quat_w=clip_bodies_quat,
            tracked_bodies_lin_vel_w=np.zeros((frames, bodies, 3), np.float32),
            tracked_bodies_ang_vel_w=np.zeros((frames, bodies, 3), np.float32),
        ),
    )
    if ref_root_pos is not None:
        motion.clip.reference_body_pos_w[:] = ref_root_pos
    sim = {
        "tracked_body_pos": robot_bodies_pos,
        "tracked_body_quat": robot_bodies_quat,
        "tracked_body_linear_velocity": np.zeros((bodies, 3), np.float32),
        "tracked_body_angular_velocity": np.zeros((bodies, 3), np.float32),
        "robot_dof_pos": np.zeros(joints, np.float32),
        "robot_dof_vel": np.zeros(joints, np.float32),
    }
    if root_pos is not None:
        sim["tracked_body_pos"][mdp.SONIC_ROOT_BODY_INDEX] = root_pos
    ctx = SimpleNamespace(
        env_id=0,
        commands={"motion": motion},
        sim=sim,
        actions={
            "joint_position": SimpleNamespace(
                action_queue=np.zeros((1, joints), np.float32),
                action_ptr=np.zeros(1, dtype=np.int64),
                action_scales=np.ones(joints, np.float32),
                default_angles=np.zeros(joints, np.float32),
            )
        },
    )
    term = mdp.SonicCriticObservation(np.zeros((1, 1, 6 + 3 * joints), np.float32), np.full(1, -1, np.int64))
    out = np.full(1 * (2 * joints + 9) + 2 * 15 * bodies + 1 * (6 + 3 * joints), np.nan, np.float32)
    mdp.SonicCriticObservation.compute(ctx, out, term)
    return out, ctx, term


def test_sonic_critic_diff_and_root_channels_match_mimiclite_semantics() -> None:
    identity6 = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], np.float32)
    yaw90 = np.zeros(4, np.float32)
    yaw90[2] = np.float32(np.sin(np.pi / 4))
    yaw90[3] = np.float32(np.cos(np.pi / 4))
    identity_body = _identity_quaternions((2,))
    # Feature-grouped diff layout offsets; body 1 at diff step s lives at
    # flat index s * 2 + 1 (step-major, two bodies).
    bodies = 2
    future_dim = 1 * (2 * 1 + 9)
    diff_stride = len(mdp.SONIC_CRITIC_DIFF_STEPS) * bodies
    pos_base = future_dim
    ori_base = pos_base + 3 * diff_stride
    lin_base = ori_base + 6 * diff_stride
    ang_base = lin_base + 3 * diff_stride

    def diff_index(step, body):
        return step * bodies + body

    # Aligned reference and robot: zero position and velocity diffs, identity
    # orientation diffs (the 6D of a null rotation, not zeros), and a
    # zero/identity root channel.
    out, _, _ = _run_critic_observation(
        np.zeros((2, 3), np.float32), identity_body, np.zeros(3, np.float32), identity_body[0]
    )
    np.testing.assert_allclose(out[pos_base:ori_base], 0.0, atol=1e-7)
    np.testing.assert_allclose(out[ori_base:lin_base], np.tile(identity6, diff_stride), atol=1e-6)
    np.testing.assert_allclose(out[lin_base : ang_base + 3 * diff_stride], 0.0, atol=1e-7)
    np.testing.assert_allclose(out[future_dim - 9 : future_dim - 6], 0.0, atol=1e-7)
    np.testing.assert_allclose(out[future_dim - 6 : future_dim], identity6, atol=1e-6)

    # Reference offset: diff pos is ref-local minus robot-local in the
    # projected-yaw anchor frames, and the velocity diff is the plain
    # world-frame difference. Both diff steps agree on a static clip.
    ref_offset = np.tile(np.asarray((0.3, 0.0, 0.1), np.float32), (2, 1))
    out, ctx, term = _run_critic_observation(ref_offset, identity_body, np.zeros(3, np.float32), identity_body[0])
    ctx.commands["motion"].clip.tracked_bodies_lin_vel_w[:, 1] = (0.5, 0.0, 0.0)
    ctx.sim["tracked_body_linear_velocity"][1] = (0.2, 0.0, 0.0)
    mdp.SonicCriticObservation.compute(ctx, out, term)
    for step in range(len(mdp.SONIC_CRITIC_DIFF_STEPS)):
        index = diff_index(step, 1)
        np.testing.assert_allclose(out[pos_base + 3 * index : pos_base + 3 * index + 3], (0.3, 0.0, 0.1), atol=1e-6)
        np.testing.assert_allclose(out[lin_base + 3 * index : lin_base + 3 * index + 3], (0.3, 0.0, 0.0), atol=1e-6)

    # A clip whose second frame keeps moving advances only the second diff
    # step; a yaw-90 reference orientation yields the relative rotation 6D in
    # this repo's first-two-columns layout.
    moving = ref_offset.copy()
    moving[1] += (0.1, 0.0, 0.0)
    yaw_body = np.tile(yaw90, (2, 1))
    out, _, _ = _run_critic_observation(moving, yaw_body, np.zeros(3, np.float32), identity_body[0])
    for step in range(len(mdp.SONIC_CRITIC_DIFF_STEPS)):
        index = diff_index(step, 1)
        np.testing.assert_allclose(
            out[ori_base + 6 * index : ori_base + 6 * index + 6], (0.0, -1.0, 1.0, 0.0, 0.0, 0.0), atol=1e-6
        )
    np.testing.assert_allclose(
        out[pos_base + 3 * diff_index(0, 1) : pos_base + 3 * diff_index(0, 1) + 3], (0.3, 0.0, 0.1), atol=1e-6
    )
    np.testing.assert_allclose(
        out[pos_base + 3 * diff_index(1, 1) : pos_base + 3 * diff_index(1, 1) + 3], (0.4, 0.0, 0.1), atol=1e-6
    )

    # The diff terms stay invariant under a common yaw rotation and xy
    # translation of both sides: each anchor's projected-yaw frame absorbs it.
    def rotate_xy(vector, translation):
        vector = np.asarray(vector, np.float32)
        translation = np.asarray(translation, np.float32)
        return np.asarray(
            (-vector[1] + translation[0], vector[0] + translation[1], vector[2] + translation[2]), np.float32
        )

    translation = (1.0, 2.0, 0.0)
    transformed = np.asarray([rotate_xy(row, translation) for row in ref_offset], np.float32)
    out, _, _ = _run_critic_observation(
        transformed,
        yaw_body,
        rotate_xy((0.0, 0.0, 0.0), translation),
        identity_body[0],
        anchor_quat=yaw90,
    )
    index = diff_index(0, 1)
    np.testing.assert_allclose(out[pos_base + 3 * index : pos_base + 3 * index + 3], (0.3, 0.0, 0.1), atol=1e-6)
    np.testing.assert_allclose(
        out[ori_base + 6 * index : ori_base + 6 * index + 6], (0.0, -1.0, 1.0, 0.0, 0.0, 0.0), atol=1e-6
    )

    # Root channel: reference root relative to the robot root in the robot
    # root frame, on the future-frame slots of the G1 grid.
    out, _, _ = _run_critic_observation(
        np.zeros((2, 3), np.float32),
        identity_body,
        np.zeros(3, np.float32),
        identity_body[0],
        root_pos=(1.0, 0.0, 0.0),
        ref_root_pos=(1.5, 0.0, 0.25),
    )
    np.testing.assert_allclose(out[future_dim - 9 : future_dim - 6], (0.5, 0.0, 0.25), atol=1e-6)
    np.testing.assert_allclose(out[future_dim - 6 : future_dim], identity6, atol=1e-6)


def test_sonic_adaptive_kernel_matches_reference_smoothing() -> None:
    """Vectorized kernel smoothing keeps the per-bin probabilities of the loop form."""
    rng = np.random.default_rng(3)
    for num_bins, kernel_size in ((1, 3), (4, 3), (30, 1), (30, 2), (30, 7), (500, 3)):
        for failed in (rng.random(num_bins).astype(np.float32) * 10.0, np.zeros(num_bins, np.float32)):
            got = mdp._adaptive_sampling_probabilities(
                failed,
                np.float32(0.2),
                np.int64(kernel_size),
                np.float32(0.8),
            )
            kernel = np.asarray([0.8**i for i in range(kernel_size)], np.float32)
            kernel /= np.sum(kernel)
            base = failed + np.float32(0.2) / num_bins
            padded = np.pad(base, (0, kernel_size - 1), mode="edge")
            want = np.asarray([np.sum(padded[i : i + kernel_size] * kernel) for i in range(num_bins)], np.float32)
            want = (want / np.sum(want)).astype(np.float32)
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)


def test_sonic_cdf_sampling_matches_linear_walk_reference() -> None:
    """Binary-search bin selection reproduces the linear walk draw for draw."""
    from motrix_env_core.numba.manager.rand import initialize_rand_states, next_uniform

    def reference(state, cdf, num_frames, zero_prob):
        unit = (next_uniform(state) + np.float32(1.0)) * np.float32(0.5)
        if cdf.size == 0:
            step_value = unit * np.float32(num_frames - 1)
        else:
            bin_id = 0
            while bin_id + 1 < cdf.size and unit > cdf[bin_id]:
                bin_id += 1
            bin_unit = (next_uniform(state) + np.float32(1.0)) * np.float32(0.5)
            step_value = (np.float32(bin_id) + bin_unit) / np.float32(cdf.size) * np.float32(num_frames)
        step_value = min(max(step_value, np.float32(0.0)), np.float32(num_frames - 2))
        step = np.int64(step_value)
        if (next_uniform(state) + np.float32(1.0)) * np.float32(0.5) < zero_prob:
            step = np.int64(0)
        return int(step)

    rng = np.random.default_rng(5)
    cases = [
        np.cumsum(np.asarray([0.1, 0.1, 0.6, 0.2], np.float32)),
        np.ones(1, np.float32),
        np.full(64, np.float32(1.0 / 64)).cumsum(),  # plateau boundaries
        rng.dirichlet(np.ones(100_000)).astype(np.float32).cumsum(),
        np.empty(0, np.float32),  # adaptive sampling disabled
    ]
    for cdf in cases:
        cdf = cdf.astype(np.float32)
        if cdf.size:
            cdf[-1] = 1.0
        for seed in range(300):
            got = mdp._sample_motion_step_from_cdf(
                initialize_rand_states(1, seed)[0], cdf, np.int64(2000), np.float32(0.1)
            )
            want = reference(initialize_rand_states(1, seed)[0], cdf, 2000, np.float32(0.1))
            assert got == want, f"cdf size {cdf.size} seed {seed}: {got} != {want}"

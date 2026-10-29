# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-side APG parsing-head contracts for g1-supermimic.

Covers the head built from the environment's NumPy surfaces: dimensions, the
analytic torch heads against the manager kernels (goal / proprio / reward
parity, terminal-endpoint semantics, imagination-start validity), and the
differentiable forward kinematics against MotrixSim. These tests own the
task package's parity contract with its own manager reward group.
"""

import numpy as np
import pytest
import torch

from motrix_env_core.torch.geometry import quat_xyzw_to_mat, unflatten_state
from motrix_envs.locomotion.kinematics import ModelKinematics
from motrix_envs.locomotion.supermimic.apg_head import build_head
from motrix_envs.locomotion.supermimic.g1 import SuperMimicManagerEnv, make_g1_supermimic_cfg
from motrix_envs.locomotion.supermimic.mdp import SuperMimicMotionCommand
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR

_DANCE_MOTION = MOTION_DIR / "dance" / "dance1_subject2.npz"


def _corpus(tmp_path, lengths=(160, 140)) -> tuple[str, ...]:
    with np.load(_DANCE_MOTION, allow_pickle=False) as data:
        total = int(np.asarray(data["num_frames"]).reshape(-1)[0])
        fields = {key: data[key] for key in data.files}
    paths = []
    start = 0
    for index, length in enumerate(lengths):
        part = {"num_frames": np.int32(length)}
        for name, value in fields.items():
            if name == "num_frames":
                continue
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                part[name] = value[start : start + length]
            else:
                part[name] = value
        path = tmp_path / f"corpus{index}.npz"
        np.savez(path, **part)
        paths.append(str(path))
        start += length
    return tuple(paths)


@pytest.mark.numba
def test_head_dimensions_and_env_buffers(tmp_path):
    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(_corpus(tmp_path)), num_envs=2, seed=1)
    motion = env.command_terms["motion"]
    assert isinstance(motion, SuperMimicMotionCommand)
    joints = env.action_space.shape[0]
    assert motion.state.shape == (2, 13 + 2 * joints)
    # The context holds friction + per-reset joint offsets + CoM offset.
    assert motion.context.shape == (2, 1 + joints + 3)
    action_term = env._action_terms["joint_position"]
    zero_action = np.zeros((2, joints), dtype=np.float32)
    np.testing.assert_allclose(
        action_term._process(zero_action),
        action_term.state.default_angles + motion.context[:, 1:30],
        atol=1e-6,
    )
    env.init_state()
    frames = motion.steps[:, 0]
    expected_q = np.clip(
        motion.clip.joint_pos[frames] + motion.context[:, 1 : 1 + joints],
        action_term.state.joint_lower,
        action_term.state.joint_upper,
    )
    np.testing.assert_allclose(motion.state[:, 13 : 13 + joints], expected_q, atol=1e-6)

    head = build_head(env)
    joints = head.num_joints
    assert joints == env.action_space.shape[0]
    # The dims follow the env surfaces: the context is friction + joint
    # offsets + CoM, proprio is 3N + 12, each goal frame is 15 + 2N wide.
    assert head.context_dim == 1 + joints + 3
    assert head.obs_dim == 3 * joints + 12
    assert head.goal_frame_dim == 15 + 2 * joints
    assert head.goal_dim == head.goal_frame_dim * head.goal_frames
    assert head.state_dim == 13 + 2 * joints
    # The proprio + goal widths exactly cover the env's policy group.
    assert head.obs_dim + head.goal_dim == env.observation_space.policy.shape[0]
    assert head.max_step_reward > 0.0
    assert head.action_low.shape == (joints,) and np.all(head.action_low < head.action_high)
    assert head.current_transition()[0].shape == (env.num_envs, head.state_dim)
    assert head.endpoint_state().shape == (env.num_envs, head.state_dim)
    assert head.sampling_state() is None or isinstance(head.sampling_state(), dict)


@pytest.mark.numba
def test_head_matches_env_outputs(tmp_path):
    """Goal, proprio, and reward features match the manager kernels."""
    cfg = make_g1_supermimic_cfg(_corpus(tmp_path, lengths=(40, 35)))
    cfg.commands.motion.num_goal_frames = 4
    env = SuperMimicManagerEnv(cfg, num_envs=3, seed=5)
    env.init_state()
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (3, env.action_space.shape[0])).astype(np.float32)
    previous = np.zeros_like(actions)
    env.step(previous)
    result = env.step(actions)
    # The head's termination classification only reads keys the env publishes.
    assert {"bad_ref_z", "bad_ref_ori"} <= set(result.metrics)

    motion = env.command_terms["motion"]
    head = build_head(env)
    frames = torch.tensor(motion.steps[:, 0], dtype=torch.long)
    unflattened = unflatten_state(torch.tensor(motion.state))

    # Done lanes reset before the returned observation; compare live episodes.
    live = ~(np.asarray(result.terminated) | np.asarray(result.truncated))
    assert bool(live.any())

    def slice_of(group: str, name: str):
        layout = env.manager_layout.observations[group]
        return next(term for term in layout.terms if term.name == name).output_slice

    # Goal features equal the manager goal observation term exactly.
    policy_goal = next(
        term for term in env.manager_layout.observations["policy"].terms if term.name == "goal"
    ).output_slice
    torch_goal = head.goal(frames, torch.tensor(motion.context)).detach().numpy()
    for lane in np.flatnonzero(live):
        np.testing.assert_allclose(torch_goal[lane], result.obs.policy[lane][policy_goal], atol=2e-4)

    # Proprio blocks equal the noise-free value-group slices.
    context = torch.tensor(motion.context)
    proprio = head.obs(unflattened, torch.tensor(actions), context, frames).detach().numpy()
    for name in ("dof_pos", "dof_vel", "base_ang_vel", "gravity", "actions"):
        obs_slice = slice_of("value", name)
        np.testing.assert_allclose(
            proprio[live][:, obs_slice], result.obs.value[live][:, obs_slice], atol=2e-3, err_msg=name
        )

    # Analytic reward tracks the manager reward on the same state.
    reward = head.reward(unflattened, frames, torch.tensor(actions), torch.tensor(previous), context)
    np.testing.assert_allclose(reward.detach().numpy()[live], result.reward[live], atol=0.02)

    # Imagination-start validity follows the owning clip's end.
    valid = head.valid_imagination_start(frames, 32)
    ends = motion.clip.frame_clip_end[frames.numpy()]
    np.testing.assert_array_equal(valid.numpy(), (frames.numpy() + 32) <= ends)

    # The head splits the policy group as [proprio | goal].
    obs, goal = head.split_observation(torch.tensor(result.obs.policy))
    assert obs.shape == (3, head.obs_dim)
    assert goal.shape == (3, head.goal_dim)
    assert head.goal_dim == head.goal_frame_dim * 4


@pytest.mark.numba
def test_head_terminal_transition_keeps_physical_endpoint(tmp_path):
    corpus = _corpus(tmp_path, lengths=(40,))
    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(corpus).for_eval(), num_envs=1, seed=3)
    env.init_state()
    motion = env.command_terms["motion"]
    final_frame = int(motion.clip.frame_clip_end[0])
    motion.requested_frames[0, 0] = final_frame - 1
    env.reset(np.array([0], dtype=np.int64))
    motion.requested_frames[0, 0] = 0  # Distinct auto-reset pose after the terminal transition.
    action = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
    result = env.step(action)

    assert bool(result.metrics["motion_clip_end"][0])
    assert bool(result.terminated[0])
    assert motion.steps[0, 0] == 0
    assert not np.allclose(motion.endpoint_state, motion.state)
    head = build_head(env)
    endpoint = unflatten_state(torch.tensor(head.endpoint_state()))
    terminal_context = torch.tensor(np.asarray(motion.context)[:1])
    reward = head.reward(
        endpoint, torch.tensor([final_frame]), torch.tensor(action), torch.tensor(action), terminal_context
    )
    np.testing.assert_allclose(reward.detach().numpy(), result.reward, atol=0.02)


def test_head_kinematics_matches_motrixsim():
    """The differentiable FK reproduces MotrixSim link poses on random states."""
    import motrixsim as mtx
    from motrix_robots.unitree import UNITREE_G1_ASSET_DIR
    from scipy.spatial.transform import Rotation as R

    from motrix_envs.locomotion.wbt.g1.common import G1_TRACKED_BODY_NAMES

    model_file = UNITREE_G1_ASSET_DIR / "g1_29dof.xml"
    model = mtx.load_model(str(model_file))
    joint_names = [str(n) for n in model.joint_names]
    kin = ModelKinematics(str(model_file), joint_names, G1_TRACKED_BODY_NAMES)

    rng = np.random.default_rng(3)
    batch = 6
    pos = rng.uniform(-1, 1, (batch, 3)).astype(np.float32)
    pos[:, 2] += 1.0
    quats = R.random(batch, random_state=11).as_quat().astype(np.float32)
    q = rng.uniform(-0.4, 0.4, (batch, len(joint_names))).astype(np.float32)
    qd = (rng.standard_normal((batch, len(joint_names))) * 0.5).astype(np.float32)
    v = rng.standard_normal((batch, 3)).astype(np.float32)
    w = rng.standard_normal((batch, 3)).astype(np.float32)

    body_p, body_R, _, _ = kin.forward(
        torch.tensor(pos),
        quat_xyzw_to_mat(torch.tensor(quats)),
        torch.tensor(v),
        torch.tensor(w),
        torch.tensor(q),
        torch.tensor(qd),
    )

    data = mtx.SceneData(model, batch=[batch])
    data.set_dof_pos(np.concatenate([pos, quats, q], 1).astype(np.float32), model)
    data.set_dof_vel(np.concatenate([v, w, qd], 1).astype(np.float32))
    model.forward_kinematic(data)
    poses = np.asarray(model.get_link_poses(data))
    link_index = {str(n): i for i, n in enumerate(model.link_names)}
    tracked = [link_index[name] for name in G1_TRACKED_BODY_NAMES]

    assert float(np.abs(body_p.detach().numpy() - poses[:, tracked, :3]).max()) < 1e-4
    reference_R = quat_xyzw_to_mat(torch.tensor(poses[:, tracked, 3:7])).numpy()
    cosine = np.clip((np.einsum("tkij,tkij->tk", reference_R, body_R.detach().numpy()) - 1) / 2, -1, 1)
    assert float(np.arccos(cosine).max()) < 2e-3

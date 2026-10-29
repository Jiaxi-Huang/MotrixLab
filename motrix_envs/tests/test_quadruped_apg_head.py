# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-side APG parsing-head contracts for the quadruped velocity-walk task.

Covers the command-driven head over the real ``go2-walk-flat`` direct
environment: dimensions, the observation split against the env's policy
group, world-root state reads, command goals, the imagination reward, and
the tilt hazard.
"""

import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.torch.geometry import unflatten_state
from motrix_envs.locomotion.quadruped.apg_head import build_head


def _make_env(num_envs: int = 2):
    return registry.resolve("go2-walk-flat").make(num_envs=num_envs, mode="train", seed=1)


def _make_head(num_envs: int = 2):
    env = _make_env(num_envs)
    return env, build_head(env).to(torch.device("cpu"))


def test_quadruped_head_dimensions():
    env, head = _make_head()
    assert head.num_joints == env.num_actuators
    assert head.num_feet == env.num_feet
    # [command 3 | friction 1 | mass 1 | CoM x/y 2 | delay 1]
    assert head.context_dim == 3 + 5
    assert head.obs_dim == 3 * head.num_joints + 10
    assert head.goal_dim == env.velocity_commands.shape[1]
    assert head.state_dim == 13 + 2 * head.num_joints
    assert head.action_low.shape == (head.num_joints,)
    assert np.all(head.action_low < head.action_high)


def test_quadruped_head_split_matches_env_policy_group():
    env, head = _make_head(3)
    result = env.init_state()
    full = torch.tensor(np.asarray(result.obs.policy))
    assert full.shape == (3, head.obs_dim + head.goal_dim)
    obs, goal = head.split_observation(full)
    assert obs.shape == (3, head.obs_dim)
    assert goal.shape == (3, head.goal_dim)
    # The command block sits between the action block and the gait phase.
    np.testing.assert_allclose(goal.numpy(), env._commands, atol=1e-6)
    command_start = 6 + 3 * head.num_joints
    recomposed = torch.zeros_like(full)
    recomposed[:, :command_start] = obs[:, :command_start]
    recomposed[:, command_start : command_start + head.goal_dim] = goal
    recomposed[:, command_start + head.goal_dim :] = obs[:, command_start:]
    torch.testing.assert_close(recomposed, full)


def test_quadruped_head_reads_world_state_and_context():
    env, head = _make_head(2)
    action = np.zeros((2, head.num_joints), np.float32)
    env.init_state()
    env.step(action)
    state, frames, context = head.current_transition()
    assert state.shape == (2, head.state_dim)
    assert frames.shape == (2,)
    assert context.shape == (2, head.context_dim)
    # The joint block of the world state is the env's own joint readout.
    np.testing.assert_allclose(state[:, 13 : 13 + head.num_joints], env.get_dof_pos(), atol=1e-6)
    np.testing.assert_allclose(state[:, -head.num_joints :], env.get_dof_vel(), atol=1e-6)
    endpoint = head.endpoint_state()
    assert endpoint.shape == (2, head.state_dim)
    # The command head of the context matches the env's command buffer.
    np.testing.assert_allclose(context[:, :3], env._commands, atol=1e-6)


def test_quadruped_head_imagination_heads_are_finite_and_command_driven():
    env, head = _make_head(2)
    action = np.zeros((2, head.num_joints), np.float32)
    env.init_state()
    env.step(action)
    state_np, frames_np, context_np = head.current_transition()
    state = unflatten_state(torch.tensor(state_np))
    frames = torch.tensor(frames_np)
    context = torch.tensor(context_np)
    zeros = torch.zeros((2, head.num_joints))

    obs = head.obs(state, zeros, context, frames)
    assert obs.shape == (2, head.obs_dim)
    assert bool(torch.isfinite(obs).all())
    goal = head.goal(frames, context)
    np.testing.assert_allclose(goal.numpy(), context[:, :3].numpy(), atol=1e-6)

    reward = head.reward(state, frames, zeros, zeros, context)
    assert reward.shape == (2,)
    assert bool(torch.isfinite(reward).all())
    assert float(reward.abs().max()) <= head.max_step_reward * head.ctrl_dt + 1e-6

    assert bool(head.valid_imagination_start(frames, 64).all())
    features = head.critic_features(state, frames, head.normalize_context(context))
    assert features.shape[-1] == 13 + 2 * head.num_joints + head.context_dim


def test_quadruped_head_soft_termination_separates_upright_from_tilted():
    env, head = _make_head(1)
    frames = torch.zeros((1,), dtype=torch.long)
    upright = head.ref_state(frames)
    assert float(head.soft_termination(upright, frames)) < 0.01
    # A 90-degree roll about x puts the up-axis z at zero, well past the threshold.
    tilted = list(upright)
    roll = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).expand(1, 3, 3)
    tilted[1] = roll
    assert float(head.soft_termination(tuple(tilted), frames)) > 0.99

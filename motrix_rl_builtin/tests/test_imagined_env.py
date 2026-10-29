# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""ImaginedEnv contracts: shapes, autograd boundaries, and parity.

Parity is asserted against a hand-rolled copy of the pre-refactor
``policy_update`` imagination math (initial history under no_grad, the raw
``_imag_step`` loop, then batched analytic reward / termination / critic
features) so the env's per-step evaluation cannot drift from it.
"""

import torch
from motrix_rl_builtin.apg.imagined_env import ImaginedEnv, imagine_rollout, imagine_step
from motrix_rl_builtin.apg.networks import Policy, WorldModel
from motrix_rl_builtin.apg.replay import ReplayBuffer
from test_apg import _NUM_JOINTS, _OBS_DIM, _STATE_DIM, _FakeHead

from motrix_env_core.torch.geometry import unflatten_state

_HIST = 2
_HORIZON = 4
_BATCH = 8


def _make_modules():
    torch.manual_seed(3)
    head = _FakeHead(None)
    wm = WorldModel([16, 16], _NUM_JOINTS, head.context_dim)
    torch.nn.init.normal_(wm.net[-1].weight, std=0.05)
    pi = Policy(_OBS_DIM, head.goal_dim, _HIST, 8, _NUM_JOINTS, [16, 16])
    return head, wm, pi


def _sample_start_and_history(head):
    buffer = ReplayBuffer(64, 2, "cpu", num_joints=_NUM_JOINTS, context_dim=head.context_dim)
    torch.manual_seed(11)
    for step in range(16):
        buffer.add(
            {
                "s": torch.randn(2, _STATE_DIM),
                "s1": torch.randn(2, _STATE_DIM),
                "a": torch.randn(2, _NUM_JOINTS),
                "a_prev": torch.randn(2, _NUM_JOINTS),
                "xi": torch.randn(2, head.context_dim),
                "r": torch.zeros(2),
                "f": torch.full((2,), step, dtype=torch.long),
                "ep_step": torch.zeros(2, dtype=torch.int32),
                "done": torch.zeros(2, dtype=torch.bool),
            }
        )
    torch.manual_seed(13)
    start, history = buffer.sample_starts(_BATCH, _HIST)
    return start, history


def _handrolled_reference(wm, pi, head, start, history):
    """The pre-refactor policy_update imagination math, verbatim."""
    state = unflatten_state(start["s"])
    context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
    normalized_context = head.normalize_context(context)
    with torch.no_grad():
        obs_hist = head.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"])
    obs_hist = head.add_noise(obs_hist)
    initial_goal = head.goal(frames, context)
    goal = initial_goal
    initial_state = state
    states, actions, previous_actions, all_frames, goals = [], [], [], [], []
    for _ in range(_HORIZON):
        previous_actions.append(previous_action)
        action = pi(obs_hist, goal)
        next_state = wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = head.add_noise(head.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = head.goal(next_frames, context)
        state, previous_action, obs_hist, frames, goal = next_state, action, next_hist, next_frames, next_goal
        states.append(state)
        actions.append(previous_action)
        all_frames.append(frames)
        goals.append(goal)

    def flatten(parts):
        return torch.stack(parts).flatten(0, 1)

    all_states = tuple(flatten([item[i] for item in states]) for i in range(6))
    flat_frames = flatten(all_frames)
    flat_context = context.unsqueeze(0).expand(_HORIZON, -1, -1).flatten(0, 1)
    rewards = head.reward(all_states, flat_frames, flatten(actions), flatten(previous_actions), flat_context)
    dones = head.soft_termination(all_states, flat_frames)
    with torch.no_grad():
        critic_states = tuple(
            torch.cat((initial_state[i].unsqueeze(0), torch.stack([item[i] for item in states])), 0).flatten(0, 1)
            for i in range(6)
        )
        critic_frames = torch.cat((start["f"].unsqueeze(0), torch.stack(all_frames)), 0).flatten()
        critic_goals = torch.cat((initial_goal.unsqueeze(0), torch.stack(goals)), 0).flatten(0, 1)
        critic_context = normalized_context.unsqueeze(0).expand(_HORIZON + 1, -1, -1).flatten(0, 1)
        critic_input = torch.cat(
            (head.critic_features(critic_states, critic_frames, critic_context), critic_goals), -1
        ).view(_HORIZON + 1, _BATCH, -1)
    return rewards, dones, critic_input


def test_rollout_shapes_and_start_indexing():
    head, wm, pi = _make_modules()
    start, history = _sample_start_and_history(head)
    env = ImaginedEnv(wm, pi, head)
    env.reset(start, history)
    traj = imagine_rollout(env, _HORIZON)
    goal_dim = head.goal_dim
    critic_dim = 3 + head.context_dim + goal_dim
    assert len(traj.states) == _HORIZON + 1
    assert all(len(state) == 6 for state in traj.states)
    assert traj.frames.shape == (_HORIZON + 1, _BATCH)
    assert traj.goals.shape == (_HORIZON + 1, _BATCH, goal_dim)
    assert traj.actions.shape == (_HORIZON, _BATCH, _NUM_JOINTS)
    assert traj.previous_actions.shape == (_HORIZON, _BATCH, _NUM_JOINTS)
    assert traj.rewards.shape == (_HORIZON, _BATCH)
    assert traj.dones.shape == (_HORIZON, _BATCH)
    assert traj.critic_input.shape == (_HORIZON + 1, _BATCH, critic_dim)
    # The first previous action is the start's a_prev, not a rollout action.
    torch.testing.assert_close(traj.previous_actions[0], start["a_prev"])


def test_rollout_matches_handrolled_policy_update_math():
    head, wm, pi = _make_modules()
    start, history = _sample_start_and_history(head)
    env = ImaginedEnv(wm, pi, head)
    env.reset(start, history)
    traj = imagine_rollout(env, _HORIZON)
    rewards, dones, critic_input = _handrolled_reference(wm, pi, head, start, history)
    torch.testing.assert_close(traj.rewards, rewards.view(_HORIZON, _BATCH))
    torch.testing.assert_close(traj.dones, dones.view(_HORIZON, _BATCH))
    torch.testing.assert_close(traj.critic_input, critic_input)


def test_grad_flow_survives_frozen_modules_and_detach_boundary():
    head, wm, pi = _make_modules()
    start, history = _sample_start_and_history(head)

    for parameter in wm.parameters():
        parameter.requires_grad_(False)

    env = ImaginedEnv(wm, pi, head)
    env.reset(start, history)
    # Replay-side initial critic input carries no graph (before stacking).
    assert not env.initial_critic_input.requires_grad
    traj = imagine_rollout(env, _HORIZON)
    # Live-graph fields feed the policy objective.
    assert traj.actions.requires_grad
    assert traj.rewards.requires_grad
    assert traj.critic_input.requires_grad

    loss = traj.rewards.sum() + traj.critic_input[-1].sum()
    loss.backward()
    # Gradients flow through the frozen world model into the policy...
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in pi.parameters())
    # ...without accumulating into the frozen parameters.
    assert all(parameter.grad is None for parameter in wm.parameters())


def test_soft_termination_stays_a_unit_interval_hazard():
    class HazardHead(_FakeHead):
        def soft_termination(self, state, frames):
            return torch.sigmoid(10.0 * state[0][..., 2])

    head, wm, pi = _make_modules()
    head = HazardHead(None)
    start, history = _sample_start_and_history(head)
    env = ImaginedEnv(wm, pi, head)
    env.reset(start, history)
    traj = imagine_rollout(env, _HORIZON)
    assert traj.dones.dtype == torch.float32
    assert torch.all(traj.dones >= 0.0) and torch.all(traj.dones <= 1.0)
    transition = env.step()
    assert transition.terminated.dtype == torch.float32
    assert transition.truncated.dtype == torch.bool and not transition.truncated.any()


def test_noise_is_applied_once_at_reset_then_once_per_step():
    shapes = []

    class NoiseRecordingHead(_FakeHead):
        def add_noise(self, obs):
            shapes.append(tuple(obs.shape))
            return obs

    head, wm, pi = _make_modules()
    head = NoiseRecordingHead(None)
    start, history = _sample_start_and_history(head)
    env = ImaginedEnv(wm, pi, head)
    env.reset(start, history)
    assert shapes == [(_BATCH, _HIST, head.obs_dim)]
    imagine_rollout(env, _HORIZON)
    assert shapes == [(_BATCH, _HIST, head.obs_dim)] + [(_BATCH, head.obs_dim)] * _HORIZON


def test_imagine_step_is_a_pure_function_of_its_inputs():
    head, wm, pi = _make_modules()
    start, history = _sample_start_and_history(head)
    state = unflatten_state(start["s"])
    goal = head.goal(start["f"], start["xi"])
    with torch.no_grad():
        obs_hist = head.add_noise(
            head.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"])
        )
    first, next_hist = imagine_step(
        pi,
        wm,
        head,
        state=state,
        obs_hist=obs_hist,
        previous_action=start["a_prev"],
        frames=start["f"],
        context=start["xi"],
        normalized_context=head.normalize_context(start["xi"]),
        goal=goal,
    )
    second, _ = imagine_step(
        pi,
        wm,
        head,
        state=state,
        obs_hist=obs_hist,
        previous_action=start["a_prev"],
        frames=start["f"],
        context=start["xi"],
        normalized_context=head.normalize_context(start["xi"]),
        goal=goal,
    )
    torch.testing.assert_close(first.reward, second.reward)
    torch.testing.assert_close(first.critic_input, second.critic_input)
    torch.testing.assert_close(next_hist, torch.cat((obs_hist[:, 1:], first.obs.unsqueeze(1)), 1))

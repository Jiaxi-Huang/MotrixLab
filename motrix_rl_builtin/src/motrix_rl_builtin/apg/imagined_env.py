# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The imagination environment: world model plus parsing head as an env.

Model-as-Env: the world model predicts the next state and the task's analytic
parsing head turns it into observations, reward, and a soft termination
hazard, so imagination becomes a rollout over an environment whose ``step``
produces the ``ArrayEnvState`` field shapes. Two semantic differences from a
real environment are deliberate and load-bearing:

- the fields are autograd-carrying tensors, not detached arrays — the policy
  objective differentiates through ``obs -> next action`` and through the
  terminal critic bootstrap;
- ``terminated`` is the differentiable soft hazard in ``[0, 1]``, not a bool
  — consumers discount with ``cumprod(1 - done)``.

Collection-side mechanics (exploration noise, action clamping, fresh-episode
history resets) live in ``rollout.py`` and are intentionally NOT used here:
imagination runs the raw policy output without noise or clamping, and the
observation history shifts unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motrix_env_core.torch.geometry import State, unflatten_state

from .contracts import ApgTask
from .networks import Policy, WorldModel


@dataclass
class ImaginedTransition:
    """One imagined transition; the shapes match the ``ArrayEnvState`` fields.

    ``action`` is the raw policy output that produced the step;
    ``previous_action`` is the action before it (``a_prev`` of the start on
    the first step). ``critic_input`` is the concatenated critic features and
    goal of the post-step state and carries the graph — the terminal
    bootstrap differentiates through it, while the critic regression detaches
    its copy.
    """

    state: State
    frames: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    obs: torch.Tensor
    goal: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    critic_input: torch.Tensor


@dataclass
class ImaginedTrajectory:
    """A full imagined rollout: a state axis of ``H + 1`` and a transition
    axis of ``H`` over the imagination batch ``B``.

    ``actions[t]`` caused ``states[t] -> states[t + 1]``;
    ``previous_actions[0]`` is the start's ``a_prev``;
    ``rewards[t] = reward(s_{t+1}, f_{t+1}, a_t, a_{t-1}, context)``. The
    rewards, dones, and final critic input keep the policy gradient; the
    critic regression consumes detached copies.
    """

    states: list[State]
    frames: torch.Tensor
    goals: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    critic_input: torch.Tensor


def imagine_step(
    pi: Policy,
    wm: WorldModel,
    head: ApgTask,
    *,
    state: State,
    obs_hist: torch.Tensor,
    previous_action: torch.Tensor,
    frames: torch.Tensor,
    context: torch.Tensor,
    normalized_context: torch.Tensor,
    goal: torch.Tensor,
) -> tuple[ImaginedTransition, torch.Tensor]:
    """One imagination step as a pure function of explicit inputs.

    Returns the transition and the next observation history; ``torch.compile``
    friendly (no hidden state). The caller freezes the world model and critic
    around the rollout; freezing only stops gradient accumulation into their
    parameters, not the flow through them.
    """
    action = pi(obs_hist, goal)
    next_state = wm.step(state, action, normalized_context)
    next_frames = frames + 1
    obs = head.add_noise(head.obs(next_state, action, context, next_frames))
    next_obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
    next_goal = head.goal(next_frames, context)
    reward = head.reward(next_state, next_frames, action, previous_action, context)
    terminated = head.soft_termination(next_state, next_frames)
    critic_input = torch.cat((head.critic_features(next_state, next_frames, normalized_context), next_goal), -1)
    transition = ImaginedTransition(
        state=next_state,
        frames=next_frames,
        action=action,
        previous_action=previous_action,
        obs=obs,
        goal=next_goal,
        reward=reward,
        terminated=terminated,
        truncated=torch.zeros_like(next_frames, dtype=torch.bool),
        critic_input=critic_input,
    )
    return transition, next_obs_hist


class ImaginedEnv:
    """A differentiable environment over ``WorldModel`` and an ``ApgTask``.

    ``reset`` installs a replay-sampled start and its boundary-padded
    observation history (the only ``no_grad`` region: the initial history is
    replay data); every ``step`` then selects the raw policy action, advances
    the world model, and evaluates the analytic heads on the imagined state.
    """

    def __init__(
        self,
        wm: WorldModel,
        pi: Policy,
        head: ApgTask,
        *,
        step_fn=None,
    ) -> None:
        self._wm = wm
        self._pi = pi
        self._head = head
        self._step_fn = step_fn if step_fn is not None else imagine_step

    def reset(self, start: dict[str, torch.Tensor], history: dict[str, torch.Tensor]) -> None:
        """Install a replay-sampled start (``sample_starts`` output) and history."""
        self._state = unflatten_state(start["s"])
        self._frames = start["f"]
        self._previous_action = start["a_prev"]
        self._context = start["xi"]
        self._normalized_context = self._head.normalize_context(self._context)
        self._goal = self._head.goal(self._frames, self._context)
        with torch.no_grad():
            self._obs_hist = self._head.add_noise(
                self._head.obs(
                    unflatten_state(history["s"]),
                    history["a_prev"],
                    history["xi"],
                    history["f"],
                )
            )
        # Replay-side data carries no graph, so the initial critic input is a
        # leaf-free tensor either way; compute it alongside the step inputs.
        self.initial_state = self._state
        self.initial_frames = self._frames
        self.initial_goal = self._goal
        self.initial_critic_input = self._critic_input(self._state, self._frames, self._goal)

    def step(self) -> ImaginedTransition:
        """Select the raw policy action and advance one imagined step."""
        transition, next_obs_hist = self._step_fn(
            self._pi,
            self._wm,
            self._head,
            state=self._state,
            obs_hist=self._obs_hist,
            previous_action=self._previous_action,
            frames=self._frames,
            context=self._context,
            normalized_context=self._normalized_context,
            goal=self._goal,
        )
        self._state = transition.state
        self._frames = transition.frames
        self._previous_action = transition.action
        self._goal = transition.goal
        self._obs_hist = next_obs_hist
        return transition

    def _critic_input(self, state: State, frames: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.cat((self._head.critic_features(state, frames, self._normalized_context), goal), -1)


def imagine_rollout(env: ImaginedEnv, horizon: int) -> ImaginedTrajectory:
    """Unroll ``horizon`` imagined steps into a single cached trajectory.

    The rollout builds the autograd graph: do not wrap it in ``no_grad`` and
    do not detach its fields — the policy objective consumes them live and
    the critic regression detaches its own copies.
    """
    states = [env.initial_state]
    frames = [env.initial_frames]
    goals = [env.initial_goal]
    critic_inputs = [env.initial_critic_input]
    actions: list[torch.Tensor] = []
    previous_actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    for _ in range(horizon):
        transition = env.step()
        actions.append(transition.action)
        previous_actions.append(transition.previous_action)
        rewards.append(transition.reward)
        dones.append(transition.terminated)
        states.append(transition.state)
        frames.append(transition.frames)
        goals.append(transition.goal)
        critic_inputs.append(transition.critic_input)
    return ImaginedTrajectory(
        states=states,
        frames=torch.stack(frames),
        goals=torch.stack(goals),
        actions=torch.stack(actions),
        previous_actions=torch.stack(previous_actions),
        rewards=torch.stack(rewards),
        dones=torch.stack(dones),
        critic_input=torch.stack(critic_inputs),
    )


__all__ = [
    "ImaginedEnv",
    "ImaginedTrajectory",
    "ImaginedTransition",
    "imagine_rollout",
    "imagine_step",
]

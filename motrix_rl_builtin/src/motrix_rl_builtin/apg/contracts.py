# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-facing contracts of the APG world-model learner.

APG is task-agnostic: the learner owns the world model, the goal-conditioned
policy, the critic, and the imagination rollout, while every piece of task
knowledge lives in a *parsing head* (``ApgTask``) implemented on the task
side. Tasks declare the head factory in their task yaml (``algo.head``) as a
``"package.module:attr"`` dotted path; :func:`build_head` imports it, calls
``factory(env)``, and validates the returned head against the protocol
surface. Heads are duck-typed: a task implementation never imports this
package, it just matches the method and attribute shapes below.

The head owns every dimension (joints, observation, goal, context), every
differentiable head evaluated in imagination (observation, goal, reward, soft
termination, privileged critic features, context normalization), the action
bounds, the observation split of the env's policy group, and the env-surface
reads collection needs (per-step state/index/context buffers, endpoint state,
termination classification, sampling-state snapshot). The ``frames`` index a
head consumes is opaque to the learner: corpus-frame tasks advance it through
a clip, command tasks may treat it as an episode step.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import numpy as np

from motrix_env_core.torch.geometry import State


@runtime_checkable
class SupportsApg(Protocol):
    """Generic environment surface the APG trainers consume.

    Both manager and direct workflow envs expose this through the
    ``ArrayEnv`` base; task-specific buffers are reached through the head.
    """

    @property
    def num_envs(self) -> int: ...

    @property
    def cfg(self) -> Any: ...

    def init_state(self) -> Any: ...

    def step(self, actions: Any) -> Any: ...


class ApgTask(Protocol):
    """The parsing-head protocol between a task and the APG learner.

    Dimensions and scalars (attributes):

    - ``num_joints``: task joint count ``N``; actions and joint blocks are
      ``N`` wide.
    - ``context_dim``: width of the per-step context vector stored in replay
      and consumed by the world model and critic.
    - ``obs_dim``: proprioceptive observation width.
    - ``goal_dim``: total goal width the policy consumes. The goal structure
      (how many future reference frames, or a quasi-static command) is task
      knowledge; the learner only sees the width.
    - ``max_step_reward`` / ``ctrl_dt``: per-step reward ceiling and control
      rate used to normalize reported rewards.
    - ``action_low`` / ``action_high``: NumPy arrays of shape ``(N,)``.

    Differentiable heads (torch, batched over the leading dim):

    - ``obs(state, previous_action, context, frames) -> Tensor`` of ``obs_dim``
    - ``add_noise(obs) -> Tensor``: observation noise applied in imagination
    - ``goal(frames, context) -> Tensor`` of ``goal_dim``; the context
      carries whatever the task needs to reconstruct goals along an
      imagined rollout (commands).
    - ``reward(state, frames, action, previous_action, context) -> Tensor``;
      like goals, command-driven rewards read the command from the context
    - ``soft_termination(state, frames) -> Tensor``: termination hazard
    - ``critic_features(state, frames, normalized_context) -> Tensor``
    - ``normalize_context(context) -> Tensor``
    - ``ref_state(frames) -> State``: a reference state used for shape probing
    - ``valid_imagination_start(frames, horizon) -> BoolTensor``
    - ``to(device) -> ApgTask``: move every tensor to the device

    Collection hooks (bound to the env at construction, NumPy in/out):

    - ``current_transition() -> (state, frames, context)``: the buffers of the
      step about to be taken, as ``(num_envs, state_dim)``, ``(num_envs,)``
      int64 and ``(num_envs, context_dim)`` arrays.
    - ``endpoint_state() -> (num_envs, state_dim)``: the physical endpoint
      state of the step just taken (terminal transitions keep it).
    - ``split_observation(full) -> (obs, goal)`` torch split of the env's
      policy observation into proprioceptive and goal tensors.
    - ``termination_counts(metrics) -> float``: per-step count of task
      failures, classified by the head from the env step metrics it owns.
    - ``sampling_state()`` / ``load_sampling_state(state)`` /
      ``sampling_metrics() -> Mapping[str, float]``: task curriculum snapshot
      channels checkpointed with the learner.
    """


def require_apg_env(env: object) -> SupportsApg:
    """Narrow an environment to the APG surface, failing closed."""
    if not isinstance(env, SupportsApg):
        raise TypeError(
            f"motrix.apg requires an environment exposing num_envs, cfg, init_state and step; got {type(env).__name__}."
        )
    return env


def resolve_head(spec: str):
    """Import an ``ApgTask`` factory from a ``"package.module:attr"`` path."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr or ":" in attr:
        raise ValueError(f"motrix.apg head must be a 'package.module:factory' dotted path, got {spec!r}.")
    factory = getattr(importlib.import_module(module_name), attr, None)
    if not callable(factory):
        raise TypeError(f"motrix.apg head {spec!r} does not resolve to a callable factory.")
    return factory


# Every attribute and method the learner, trainers, and evaluator consume;
# checked by name so heads stay duck-typed (no isinstance against ApgTask).
_HEAD_MEMBERS = (
    "num_joints",
    "context_dim",
    "obs_dim",
    "goal_dim",
    "max_step_reward",
    "ctrl_dt",
    "action_low",
    "action_high",
    "obs",
    "add_noise",
    "goal",
    "reward",
    "soft_termination",
    "critic_features",
    "normalize_context",
    "ref_state",
    "valid_imagination_start",
    "to",
    "current_transition",
    "endpoint_state",
    "split_observation",
    "termination_counts",
    "sampling_state",
    "load_sampling_state",
    "sampling_metrics",
)


def require_apg_head(head) -> ApgTask:
    """Narrow a constructed head to the ``ApgTask`` surface, failing closed."""
    missing = [name for name in _HEAD_MEMBERS if not hasattr(head, name)]
    if missing:
        raise TypeError(f"The motrix.apg head {type(head).__name__} is missing ApgTask members: {', '.join(missing)}.")
    return head


def build_head(spec: str, env) -> ApgTask:
    """Resolve a head factory, build it against ``env``, and validate it."""
    return require_apg_head(resolve_head(spec)(env))


def head_transition_arrays(head: ApgTask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the pending step buffers through the head as NumPy arrays."""
    state, frames, context = head.current_transition()
    return (
        np.asarray(state, dtype=np.float32),
        np.asarray(frames, dtype=np.int64),
        np.asarray(context, dtype=np.float32),
    )


__all__ = [
    "ApgTask",
    "State",
    "SupportsApg",
    "build_head",
    "head_transition_arrays",
    "require_apg_env",
    "require_apg_head",
    "resolve_head",
]

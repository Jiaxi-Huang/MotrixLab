# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""World-model contracts between the SuperMimic trainer and the task env.

MotrixLab environments are NumPy-frontends; the torch side (differentiable
MDP, networks, learner) lives in this plugin. The contract below is the
NumPy surface the plugin consumes: a manager environment whose ``motion``
command term publishes the corpus clip and the per-environment state, frame,
and simulation-context buffers. :func:`build_supermimic_mdp` in
:mod:`motrix_rl_builtin.supermimic.mdp` turns that surface into the torch
MDP; the plugin never imports task packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

State = tuple[Any, Any, Any, Any, Any, Any]


@runtime_checkable
class SuperMimicMotionTermProtocol(Protocol):
    """Buffers the motion command publishes every control step."""

    @property
    def clip(self) -> Any: ...

    @property
    def steps(self) -> Any: ...

    @property
    def state(self) -> Any: ...

    @property
    def context(self) -> Any: ...


@runtime_checkable
class SupportsSuperMimic(Protocol):
    """NumPy environment surface required by the SuperMimic trainer."""

    @property
    def command_terms(self) -> Mapping[str, Any]: ...

    @property
    def num_envs(self) -> int: ...

    @property
    def cfg(self) -> Any: ...

    def init_state(self) -> Any: ...

    def step(self, actions: Any) -> Any: ...


def require_supermimic_env(env: object) -> SupportsSuperMimic:
    """Narrow an environment to the contract, failing closed."""
    if not isinstance(env, SupportsSuperMimic):
        raise TypeError(
            f"motrix.supermimic requires a manager environment exposing the SuperMimic contract "
            f"(command_terms, num_envs, cfg, init_state, step); got {type(env).__name__}."
        )
    motion = env.command_terms.get("motion")
    if motion is None or not hasattr(motion, "state") or not hasattr(motion, "context"):
        raise TypeError(
            "motrix.supermimic requires a 'motion' command term publishing the state, frame, "
            "and simulation-context buffers."
        )
    return env


__all__ = ["State", "SuperMimicMotionTermProtocol", "SupportsSuperMimic", "require_supermimic_env"]

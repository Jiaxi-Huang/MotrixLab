# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG input semantics resolved from the Manager's named observation layout."""

from dataclasses import dataclass

import torch

from motrix_env_core.manager.env import ManagerEnv


@dataclass(frozen=True)
class ApgObservationLayout:
    proprio_slice: slice
    goal_slice: slice
    value_dim: int
    action_dim: int

    @classmethod
    def from_env(cls, env: ManagerEnv) -> "ApgObservationLayout":
        spec = env.observation_spec
        for name, group in spec.groups.items():
            if group.history_length != 1:
                raise ValueError(f"APG observation group {name!r} requires history_length=1")
        if spec.value is None:
            raise ValueError("APG requires a privileged 'value' observation group")
        terms = {term.name: term.output_slice for term in env.manager_layout.observations["policy"].terms}
        if terms.keys() != {"proprio", "goal"}:
            raise ValueError(f"APG policy requires exactly 'proprio' and 'goal' terms; got {sorted(terms)}")
        return cls(terms["proprio"], terms["goal"], spec.value.frame_space.shape[0], env.action_space.shape[0])

    @property
    def proprio_dim(self) -> int:
        return self.proprio_slice.stop - self.proprio_slice.start

    @property
    def goal_dim(self) -> int:
        return self.goal_slice.stop - self.goal_slice.start

    def proprio(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[..., self.proprio_slice]

    def goal(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[..., self.goal_slice]

    def validate_policy_dimensions(self, proprio_dim: int, goal_dim: int, action_dim: int) -> None:
        expected = (self.proprio_dim, self.goal_dim, self.action_dim)
        actual = (proprio_dim, goal_dim, action_dim)
        if actual != expected:
            raise ValueError(
                f"APG policy dimensions (proprio, goal, action) {actual} do not match environment {expected}"
            )

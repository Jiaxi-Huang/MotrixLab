# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Differentiable forward kinematics shared by the locomotion task heads.

The kinematic tree (parent, static local pose, hinge axis, joint column) is
read once from a MotrixSim model file; ``forward`` then evaluates tracked
body poses and link-origin velocities batched in torch, so reward and critic
heads can backpropagate through it during imagined rollouts. The rotation
and state-layout math it builds on is the single implementation in
``motrix_env_core.torch.geometry``. Requires torch (the ``apg`` optional
dependency).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from motrix_env_core.torch.geometry import axis_angle_to_mat, quat_xyzw_to_mat


class ModelKinematics:
    """Differentiable forward kinematics of a robot's tracked bodies."""

    def __init__(
        self,
        model_file: str | Path,
        joint_names: Sequence[str],
        tracked_body_names: Sequence[str],
        device: torch.device | str = "cpu",
    ) -> None:
        import motrixsim as mtx

        model = mtx.load_model(str(model_file))
        self.device = torch.device(device)
        joint_index = {name: index for index, name in enumerate(joint_names)}
        link_names = [str(name) for name in model.link_names]
        parent: list[int] = []
        joint_idx: list[int] = []
        axes: list[list[float]] = []
        offsets_p: list[list[float]] = []
        offsets_r: list[list[float]] = []
        link_to_node: dict[str, int] = {}
        for link_name in link_names:
            link = model.get_link(link_name)
            parent_link = link.parent
            local_pose = np.asarray(link.local_pose, dtype=np.float64)
            link_to_node[link_name] = len(parent)
            parent.append(-1 if parent_link is None else link_to_node[str(parent_link.name)])
            pose = local_pose.astype(np.float32)
            offsets_p.append(pose[:3])
            offsets_r.append(pose[3:])  # xyzw, like every MotrixSim quaternion
            if link.num_joints == 1:
                joint = model.get_joint(str(model.joint_names[int(link.joint_indices[0])]))
                joint_idx.append(joint_index[str(joint.name)])
                axes.append(np.asarray(joint.axis, dtype=np.float32).tolist())
            else:
                if link.num_joints > 1:
                    raise ValueError(f"Model kinematics cannot handle multi-joint link {link_name!r}")
                joint_idx.append(-1)
                axes.append([0.0, 0.0, 0.0])
        if parent[0] != -1:
            raise ValueError("Model kinematics requires the first link to be the floating base")
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.as_tensor(np.asarray(offsets_p, dtype=np.float32), device=self.device)
        self.off_R = quat_xyzw_to_mat(torch.as_tensor(np.asarray(offsets_r, dtype=np.float32), device=self.device))
        tracked = []
        for name in tracked_body_names:
            try:
                tracked.append(link_names.index(name))
            except ValueError:
                raise ValueError(f"Tracked body {name!r} is not a model link") from None
        self.tracked_idx = tracked
        self.num_bodies = len(parent)

        lower = np.zeros((len(joint_names),), dtype=np.float64)
        upper = np.zeros((len(joint_names),), dtype=np.float64)
        for index in range(model.num_joints):
            name = str(model.joint_names[index])
            if name in joint_index:
                column = joint_index[name]
                limits = np.asarray(model.get_joint(name).range, dtype=np.float64).reshape(-1)
                lower[column] = limits[0]
                upper[column] = limits[1]
        self.q_lo = torch.tensor(lower, dtype=torch.float32, device=self.device)
        self.q_hi = torch.tensor(upper, dtype=torch.float32, device=self.device)

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracked-body world poses and link-origin velocities, differentiable."""
        positions = [p]
        rotations = [R]
        linear_velocities = [v]
        angular_velocities = [w]
        for index in range(1, self.num_bodies):
            parent_index = self.parent[index]
            parent_rotation = rotations[parent_index]
            position = positions[parent_index] + (parent_rotation @ self.off_p[index].unsqueeze(-1)).squeeze(-1)
            offset_rotation = parent_rotation @ self.off_R[index]
            joint_index = self.joint_idx[index]
            if joint_index >= 0:
                angle = q[..., joint_index]
                rotation = offset_rotation @ axis_angle_to_mat(self.axes[index], angle)
                axis_world = (offset_rotation @ self.axes[index].unsqueeze(-1)).squeeze(-1)
                angular_velocity = (
                    angular_velocities[parent_index] + axis_world * qd[..., joint_index : joint_index + 1]
                )
            else:
                rotation = offset_rotation
                angular_velocity = angular_velocities[parent_index]
            linear_velocity = linear_velocities[parent_index] + torch.cross(
                angular_velocities[parent_index],
                position - positions[parent_index],
                dim=-1,
            )
            positions.append(position)
            rotations.append(rotation)
            linear_velocities.append(linear_velocity)
            angular_velocities.append(angular_velocity)
        return (
            torch.stack([positions[index] for index in self.tracked_idx], -2),
            torch.stack([rotations[index] for index in self.tracked_idx], -3),
            torch.stack([linear_velocities[index] for index in self.tracked_idx], -2),
            torch.stack([angular_velocities[index] for index in self.tracked_idx], -2),
        )


__all__ = ["ModelKinematics"]

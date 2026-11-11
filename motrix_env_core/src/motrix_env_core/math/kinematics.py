# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Differentiable forward kinematics over a serializable kinematic tree.

The tree is pure data (parent links, joint axes, fixed offsets); loading it
from a simulator model lives in the backend packages. The recursion here is
plain torch, so gradients flow from body poses back to the root state and
joint positions — the property world-model imagination relies on: the same
FK evaluates tracked-body poses for reference motions, logged simulator
states, and world-model predictions.

Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
Chong Zhang. MIT License.
"""

from __future__ import annotations

import torch

from motrix_env_core.math.quaternion import to_matrix
from motrix_env_core.math.rotation import axis_angle_to_mat

__all__ = ["KinematicTree"]


class KinematicTree:
    """Moving-body tree topology for differentiable FK.

    Bodies are listed in tree order (parents before children). Each moving
    body hosts exactly one joint: either a free/root joint (``joint_idx`` is
    ``None``, the body takes the root state directly) or a 1-DoF revolute
    joint (``joint_idx`` indexes into the joint-position vector ``q``).
    ``off_p`` / ``off_R`` are the fixed pose of the body in its parent frame.

    Args:
        parent: Parent body index per body; ``-1`` for the root body.
        joint_idx: q-vector index per body; ``-1`` for the root body.
        axes: Unit revolute axes in each body's parent frame, ``(N, 3)``.
        off_p: Body origins in the parent frame, ``(N, 3)``.
        off_R: Body orientations in the parent frame, as ``xyzw`` quaternions
            ``(N, 4)`` (the repo-wide convention of :mod:`motrix_env_core.math.quaternion`)
            or rotation matrices ``(N, 3, 3)``.
        device: Device for the constant buffers.
    """

    def __init__(
        self,
        parent: list[int],
        joint_idx: list[int],
        axes: torch.Tensor,
        off_p: torch.Tensor,
        off_R: torch.Tensor,
        link_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        *,
        device: str | torch.device = "cpu",
    ):
        self.parent = list(parent)
        self.joint_idx = list(joint_idx)
        self.link_names = list(link_names) if link_names is not None else [str(i) for i in range(len(parent))]
        self.joint_names = (
            list(joint_names) if joint_names is not None else [f"joint_{i}" for i in range(max(self.joint_idx) + 1)]
        )
        self.axes = torch.as_tensor(axes, dtype=torch.float32, device=device)
        self.off_p = torch.as_tensor(off_p, dtype=torch.float32, device=device)
        if off_R.shape[-1] == 4:  # xyzw quaternion (repo-wide convention)
            self.off_R = to_matrix(torch.as_tensor(off_R, dtype=torch.float32, device=device))
        else:
            self.off_R = torch.as_tensor(off_R, dtype=torch.float32, device=device)
        self.num_bodies = len(self.parent)
        # Leaf rotations cannot affect any link origin, including all-body output.
        self._position_rotation_needed = frozenset(par for par in self.parent if par >= 0)
        # Depth-grouped index tensors for the batched FK pass: one vectorized
        # step per tree level (every parent is complete a level above). The
        # eager per-body loop this replaces spent ~8 kernel launches per body
        # — the dominant dispatch cost of imagination rollouts.
        depth = [0] * self.num_bodies
        for i in range(self.num_bodies):
            if self.parent[i] >= 0:
                depth[i] = depth[self.parent[i]] + 1
        self._levels = tuple(
            torch.tensor(
                [i for i in range(self.num_bodies) if depth[i] == level], dtype=torch.long, device=device
            )
            for level in range(1, max(depth) + 1)
        )
        self._parent_of = torch.tensor(self.parent, dtype=torch.long, device=device)
        self._joint_of = torch.tensor(self.joint_idx, dtype=torch.long, device=device)
        self._roots = torch.tensor(
            [i for i in range(self.num_bodies) if self.parent[i] < 0], dtype=torch.long, device=device
        )

    def positions(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        q: torch.Tensor,
        bodies: list[int] | None = None,
    ) -> torch.Tensor:
        """World origins of the requested bodies without computing velocities.

        Gradients flow to the root state ``p``/``R`` and joint positions ``q``.

        Args:
            p, R: Root position ``(..., 3)`` and rotation ``(..., 3, 3)``.
            q: Joint positions ``(..., num_joints)``.
            bodies: Body indices to return; all bodies when ``None``.

        Returns:
            Body origins stacked on a new second-to-last axis, ``(..., M, 3)``.
        """
        positions: list[torch.Tensor | None] = [None] * self.num_bodies
        rotations: list[torch.Tensor | None] = [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                positions[i], rotations[i] = p, R
                continue
            parent_rotation = rotations[par]
            positions[i] = positions[par] + (parent_rotation @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            if i in self._position_rotation_needed:
                rotations[i] = (parent_rotation @ self.off_R[i]) @ axis_angle_to_mat(
                    self.axes[i], q[..., self.joint_idx[i]]
                )
        idx = range(self.num_bodies) if bodies is None else bodies
        return torch.stack([positions[i] for i in idx], -2)

    def forward(
        self,
        p: torch.Tensor,
        R: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        bodies: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World poses and velocities of the requested bodies.

        Gradients flow to the root state ``p``/``R``/``v``/``w`` and joint
        states ``q``/``qd``.

        Args:
            p, R, v, w: Root position ``(..., 3)``, rotation ``(..., 3, 3)``,
                and world-frame linear / angular velocity ``(..., 3)``.
            q, qd: Joint positions / velocities ``(..., num_joints)``.
            bodies: Body indices to return; all bodies when ``None``.

        Returns:
            Positions ``(..., M, 3)``, rotations ``(..., M, 3, 3)``, link-origin
            linear velocities ``(..., M, 3)`` and angular velocities ``(..., M, 3)``.
        """
        batch_shape = p.shape[:-1]
        P_all = p.new_zeros(batch_shape + (self.num_bodies, 3))
        R_all = R.new_zeros(batch_shape + (self.num_bodies, 3, 3))
        V_all = v.new_zeros(batch_shape + (self.num_bodies, 3))
        W_all = w.new_zeros(batch_shape + (self.num_bodies, 3))
        # Every root body carries the root state itself.
        P_all[..., self._roots, :] = p.unsqueeze(-2)
        R_all[..., self._roots, :, :] = R.unsqueeze(-3)
        V_all[..., self._roots, :] = v.unsqueeze(-2)
        W_all[..., self._roots, :] = w.unsqueeze(-2)
        # One batched recursion step per depth level — same per-body math as
        # the eager loop it replaces (a fixed link's joint index -1 wraps
        # exactly like the original q[..., -1] read), parents strictly
        # shallower so the in-place level writes never alias the reads.
        for level in self._levels:
            par = self._parent_of[level]
            joint = self._joint_of[level]
            Rp = R_all[..., par, :, :]
            Pp = P_all[..., par, :]
            Vp = V_all[..., par, :]
            Wp = W_all[..., par, :]
            pi = Pp + (Rp @ self.off_p[level].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[level]
            Ri = R_off @ axis_angle_to_mat(self.axes[level], q[..., joint])
            axis_w = (R_off @ self.axes[level].unsqueeze(-1)).squeeze(-1)
            wi = Wp + axis_w * qd[..., joint].unsqueeze(-1)
            vi = Vp + torch.cross(Wp, pi - Pp, dim=-1)
            P_all[..., level, :] = pi
            R_all[..., level, :, :] = Ri
            V_all[..., level, :] = vi
            W_all[..., level, :] = wi
        idx = (
            torch.arange(self.num_bodies, device=p.device)
            if bodies is None
            else torch.as_tensor(bodies, device=p.device)
        )
        return (
            P_all[..., idx, :],
            R_all[..., idx, :, :],
            V_all[..., idx, :],
            W_all[..., idx, :],
        )

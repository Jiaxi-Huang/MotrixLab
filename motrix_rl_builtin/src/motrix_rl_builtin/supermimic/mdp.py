# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-facing differentiable MDP for the SuperMimic world-model learner.

MotrixLab environments are NumPy-frontends; the torch side of the contract
lives here, on the RL plugin that owns the torch dependency. The module
mirrors the manager-kernel semantics of the g1-supermimic task (same
anchoring, same feature layouts): differentiable forward kinematics for the
tracked bodies, the analytic reward / soft-termination heads evaluated in
imagination, the goal and proprioceptive features the policy consumes, and
the privileged critic features. :func:`build_supermimic_mdp` assembles the
MDP from the environment's NumPy surfaces (the motion command's corpus clip
and buffers, the robot config, and the control rate) without importing task
packages.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

# State layout shared with the manager command's ``state`` buffer:
# [root position 3 | root quaternion xyzw 4 | linear velocity 3 |
#  angular velocity 3 | joint positions N | joint velocities N].
STATE_HEAD_DIM = 13


def quat_xyzw_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices of xyzw quaternions, ``(..., 3, 3)``."""
    q = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def mat_to_quat_xyzw(mat: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions of rotation matrices (largest-component branch)."""
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(((m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw), qw), -1),
            torch.stack((qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx), (m21 - m12) / (4 * qx)), -1),
            torch.stack(((m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy), (m02 - m20) / (4 * qy)), -1),
            torch.stack(((m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz, (m10 - m01) / (4 * qz)), -1),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return q * torch.where(q[..., 3:4] < 0, -1.0, 1.0)


def mat_to_6d_rows(rotation: torch.Tensor) -> torch.Tensor:
    """First two matrix rows, row-major — the manager kernel's 6D layout."""
    return rotation[..., :2, :].reshape(rotation.shape[:-2] + (6,))


def yaw_mat(rotation: torch.Tensor) -> torch.Tensor:
    """Yaw-only rotation matrices (column-normalized forward projection)."""
    forward_x, forward_y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(forward_x * forward_x + forward_y * forward_y + 1e-9)
    cos, sin = forward_x / norm, forward_y / norm
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    return torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1).reshape(rotation.shape[:-2] + (3, 3))


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices of axis-angle pairs (axis is unit in the last dim)."""
    x, y, z = axis.unbind(-1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    complement = 1 - cos
    return torch.stack(
        (
            cos + x * x * complement,
            x * y * complement - z * sin,
            x * z * complement + y * sin,
            y * x * complement + z * sin,
            cos + y * y * complement,
            y * z * complement - x * sin,
            z * x * complement - y * sin,
            z * y * complement + x * sin,
            cos + z * z * complement,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))


def rot_angle_sq(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between two rotation matrices."""
    trace = (first * second).sum((-1, -2))
    cosine = torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine) ** 2


class SuperMimicKinematics:
    """Differentiable forward kinematics of the tracked G1 bodies.

    The kinematic tree (parent, static local pose, hinge axis, joint column)
    is read once from the MotrixSim model file; ``forward`` then evaluates
    body poses and link-origin velocities batched in torch, mirroring the
    upstream G1 kinematics.
    """

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
                    raise ValueError(f"SuperMimic kinematics cannot handle multi-joint link {link_name!r}")
                joint_idx.append(-1)
                axes.append([0.0, 0.0, 0.0])
        if parent[0] != -1:
            raise ValueError("SuperMimic kinematics requires the first link to be the floating base")
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(axes, dtype=torch.float32, device=self.device)
        self.off_p = torch.tensor(offsets_p, dtype=torch.float32, device=self.device)
        self.off_R = quat_xyzw_to_mat(torch.tensor(offsets_r, dtype=torch.float32, device=self.device))
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


class SuperMimicTorchMDP:
    """Analytic heads over the motion corpus for imagination and collection.

    Attributes:
        num_joints: Task joint count ``N``.
        context_dim: Simulation-context width (friction scale + torso CoM).
        obs_dim: Proprioceptive observation width ``3N + 12``.
        goal_frame_dim: Goal width per future frame ``15 + 2N``.
        max_step_reward: Dt-free per-step reward ceiling (5.0).
        state_dim: Flattened state width ``13 + 2N`` (xyzw quaternion).
    """

    def __init__(
        self,
        *,
        clip,
        kinematics: SuperMimicKinematics,
        joint_names: Sequence[str],
        tracked_body_names: Sequence[str],
        default_joint_pos: Sequence[float],
        obs_noise: Sequence[float],
        ctrl_dt: float,
        reward_cfg: RewardCfg | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.kin = kinematics
        self.num_joints = clip.joint_pos.shape[1]
        self.num_tracked = len(tracked_body_names)
        self.context_dim = 4
        self.obs_dim = 3 * self.num_joints + 12
        self.goal_frame_dim = 15 + 2 * self.num_joints
        self.state_dim = STATE_HEAD_DIM + 2 * self.num_joints
        self.max_step_reward = 5.0
        self.ctrl_dt = float(ctrl_dt)
        self.cfg = reward_cfg if reward_cfg is not None else RewardCfg()
        self.default_q = torch.tensor(default_joint_pos, dtype=torch.float32, device=self.device)
        self.obs_noise = torch.tensor(obs_noise, dtype=torch.float32, device=self.device)

        root_R = quat_xyzw_to_mat(torch.tensor(clip.root_body_quat_w, dtype=torch.float32))
        self.root_p = torch.tensor(clip.root_body_pos_w, dtype=torch.float32)
        self.root_R = root_R
        self.root_q = torch.tensor(clip.root_body_quat_w, dtype=torch.float32)
        self.root_v = torch.tensor(clip.root_body_lin_vel_w, dtype=torch.float32)
        self.root_w = torch.tensor(clip.root_body_ang_vel_w, dtype=torch.float32)
        self.q = torch.tensor(clip.joint_pos, dtype=torch.float32)
        self.qd = torch.tensor(clip.joint_vel, dtype=torch.float32)
        self.body_p = torch.tensor(clip.tracked_bodies_pos_w, dtype=torch.float32)
        self.body_R = quat_xyzw_to_mat(torch.tensor(clip.tracked_bodies_quat_w, dtype=torch.float32))
        self.body_v = torch.tensor(clip.tracked_bodies_lin_vel_w, dtype=torch.float32)
        self.body_w = torch.tensor(clip.tracked_bodies_ang_vel_w, dtype=torch.float32)
        self.clip_end = torch.tensor(clip.frame_clip_end, dtype=torch.int64)
        middle = (self.kin.q_lo + self.kin.q_hi) / 2
        half_range = (self.kin.q_hi - self.kin.q_lo) / 2
        self.soft_lo = middle - half_range * self.cfg.soft_limit_factor
        self.soft_hi = middle + half_range * self.cfg.soft_limit_factor

    def to(self, device: torch.device | str) -> SuperMimicTorchMDP:
        self.device = torch.device(device)
        tensors = (
            "default_q",
            "obs_noise",
            "root_p",
            "root_R",
            "root_q",
            "root_v",
            "root_w",
            "q",
            "qd",
            "body_p",
            "body_R",
            "body_v",
            "body_w",
            "clip_end",
            "soft_lo",
            "soft_hi",
        )
        for name in tensors:
            setattr(self, name, getattr(self, name).to(self.device))
        self.kin.axes = self.kin.axes.to(self.device)
        self.kin.off_p = self.kin.off_p.to(self.device)
        self.kin.off_R = self.kin.off_R.to(self.device)
        self.kin.q_lo = self.kin.q_lo.to(self.device)
        self.kin.q_hi = self.kin.q_hi.to(self.device)
        return self

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def unflatten_state(self, value: torch.Tensor):
        num_joints = (value.shape[-1] - STATE_HEAD_DIM) // 2
        joint_end = STATE_HEAD_DIM + num_joints
        return (
            value[..., 0:3],
            quat_xyzw_to_mat(value[..., 3:7]),
            value[..., 7:10],
            value[..., 10:13],
            value[..., 13:joint_end],
            value[..., joint_end:],
        )

    def flatten_state(self, state) -> torch.Tensor:
        position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        return torch.cat((position, mat_to_quat_xyzw(rotation), lin_vel, ang_vel, joint_pos, joint_vel), -1)

    def ref_state(self, frames: torch.Tensor):
        """Reference ``(p, R, v, w, q, qd)`` at the given corpus frames."""
        return (
            self.root_p[frames],
            self.root_R[frames],
            self.root_v[frames],
            self.root_w[frames],
            self.q[frames],
            self.qd[frames],
        )

    def valid_imagination_start(self, frames: torch.Tensor, horizon: int) -> torch.Tensor:
        """Frames whose owning clip still has ``horizon`` frames ahead."""
        return frames + horizon <= self.clip_end[frames]

    # ------------------------------------------------------------------
    # Observation / goal features
    # ------------------------------------------------------------------

    def obs(self, state, previous_action: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """The 99-D proprioceptive observation for the 29-DoF G1.

        Layout matches the manager policy group: joint positions relative to
        the default pose, joint velocities, root-local angular velocity,
        gravity, previous action, and the relative rotation to the current
        reference frame (first-two-rows 6D).
        """
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        gravity = -rotation[..., 2, :]
        return torch.cat(
            (
                joint_pos - self.default_q,
                joint_vel,
                (rotation_t @ angular_velocity.unsqueeze(-1)).squeeze(-1),
                gravity,
                previous_action,
                mat_to_6d_rows(rotation_t @ self.root_R[frames]),
            ),
            -1,
        )

    def add_noise(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Future reference features in the current reference heading frame.

        Per future frame (clamped at the owning clip's end): root position
        delta, relative root rotation 6D (rows), heading-local root velocities,
        reference joint positions and scaled velocities — the exact layout of
        the manager goal observation term.
        """
        end = self.clip_end[frames]
        future = frames.unsqueeze(-1) + torch.arange(1, num_frames + 1, device=frames.device)
        future = torch.minimum(future, end.unsqueeze(-1))
        heading = yaw_mat(self.root_R[frames])
        heading_t = heading.transpose(-1, -2).unsqueeze(-3)
        position_delta = (heading_t @ (self.root_p[future] - self.root_p[frames].unsqueeze(1)).unsqueeze(-1)).squeeze(
            -1
        )
        rotation = mat_to_6d_rows(heading_t @ self.root_R[future])
        linear = (heading_t @ self.root_v[future].unsqueeze(-1)).squeeze(-1)
        angular = (heading_t @ self.root_w[future].unsqueeze(-1)).squeeze(-1)
        features = torch.cat((position_delta, rotation, linear, angular, self.q[future], self.qd[future] * 0.1), -1)
        return features.reshape(frames.shape[0], -1)

    # ------------------------------------------------------------------
    # Analytic reward / termination heads
    # ------------------------------------------------------------------

    def reward(self, state, frames: torch.Tensor, action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
        """The dt-weighted tracking reward, matching the manager reward group."""
        cfg = self.cfg
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        body_p, body_R, body_v, body_w = self.kin.forward(
            position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel
        )
        ref_p, ref_R = self.root_p[frames], self.root_R[frames]
        delta_rotation = yaw_mat(rotation @ ref_R.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + (
            delta_rotation.unsqueeze(-3) @ (self.body_p[frames] - ref_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        ref_body_R = delta_rotation.unsqueeze(-3) @ self.body_R[frames]
        error_pos = ((ref_body_p - body_p) ** 2).sum(-1).mean(-1)
        error_ori = rot_angle_sq(ref_body_R, body_R).mean(-1)
        error_linvel = ((self.body_v[frames] - body_v) ** 2).sum(-1).mean(-1)
        error_angvel = ((self.body_w[frames] - body_w) ** 2).sum(-1).mean(-1)
        error_root_pos = ((ref_p - position) ** 2).sum(-1)
        error_root_ori = rot_angle_sq(ref_R, rotation)
        tracking = (
            torch.exp(-error_pos / cfg.body_pos_std**2)
            + torch.exp(-error_ori / cfg.body_ori_std**2)
            + torch.exp(-error_linvel / cfg.body_linvel_std**2)
            + torch.exp(-error_angvel / cfg.body_angvel_std**2)
            + 0.5 * torch.exp(-error_root_pos / cfg.root_pos_std**2)
            + 0.5 * torch.exp(-error_root_ori / cfg.root_ori_std**2)
        )
        action_rate = ((action - previous_action) ** 2).sum(-1)
        joint_limit = (torch.relu(joint_pos - self.soft_hi) + torch.relu(self.soft_lo - joint_pos)).sum(-1)
        return (tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit) * self.ctrl_dt

    def soft_termination(self, state, frames: torch.Tensor) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        position, rotation = state[0], state[1]
        ref_p, ref_R = self.root_p[frames], self.root_R[frames]
        error_height = (ref_p[..., 2] - position[..., 2]).abs()
        error_tilt = (ref_R[..., 2, 2] - rotation[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((error_height - self.cfg.term_height) / 0.02)) * (
            1 - torch.sigmoid((error_tilt - self.cfg.term_tilt) / 0.04)
        )
        return 1 - alive

    # ------------------------------------------------------------------
    # Critic features / context
    # ------------------------------------------------------------------

    def critic_features(
        self,
        state,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
    ) -> torch.Tensor:
        """Privileged value features: invariant state plus reference errors."""
        position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel = state
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        ref_p, ref_R = self.root_p[frames], self.root_R[frames]
        body_p = self.kin.forward(position, rotation, linear_velocity, angular_velocity, joint_pos, joint_vel)[0]
        delta_rotation = yaw_mat(rotation @ ref_R.transpose(-1, -2))
        anchor = torch.cat((position[..., :2], ref_p[..., 2:3]), -1)
        ref_body_p = anchor.unsqueeze(-2) + (
            delta_rotation.unsqueeze(-3) @ (self.body_p[frames] - ref_p.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        body_error = (heading_t.unsqueeze(-3) @ (ref_body_p - body_p).unsqueeze(-1)).squeeze(-1).flatten(-2)
        root_error = (heading_t @ (ref_p - position).unsqueeze(-1)).squeeze(-1)
        relative_rotation = mat_to_6d_rows(rotation.transpose(-1, -2) @ ref_R)
        return torch.cat(
            (
                self.invariant_features(state),
                self.q[frames] - joint_pos,
                body_error,
                root_error,
                relative_rotation,
                normalized_context,
            ),
            -1,
        )

    def invariant_features(self, state) -> torch.Tensor:
        """Yaw-invariant state summary used by the world-model inputs."""
        _, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
        heading = yaw_mat(rotation)
        rotation_t = rotation.transpose(-1, -2)
        return torch.cat(
            (
                state[0][..., 2:3],
                mat_to_6d_rows(heading.transpose(-1, -2) @ rotation),
                (rotation_t @ lin_vel.unsqueeze(-1)).squeeze(-1),
                (rotation_t @ ang_vel.unsqueeze(-1)).squeeze(-1),
                joint_pos,
                joint_vel,
            ),
            -1,
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        """Standardize the friction scale and CoM offset to ~unit range."""
        return torch.cat(
            (
                (context[..., 0:1] - 0.75) / 0.45,
                context[..., 1:] / 0.05,
            ),
            -1,
        )


class RewardCfg:
    """Kernel widths and penalties of the analytic reward (upstream values)."""

    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def obs_noise_table(num_joints: int) -> np.ndarray:
    """Per-dim uniform proprio noise amplitudes (the upstream OBS_NOISE table).

    Blocks: joint position, joint velocity, local angular velocity, gravity,
    previous action (exact), relative rotation 6D.
    """
    return np.concatenate(
        (
            np.full(num_joints, 0.01),
            np.full(num_joints, 0.5),
            np.full(3, 0.2),
            np.full(3, 0.05),
            np.zeros(num_joints),
            np.full(6, 0.05),
        )
    ).astype(np.float32)


def build_supermimic_mdp(env) -> SuperMimicTorchMDP:
    """Assemble the differentiable MDP from the environment's NumPy surfaces.

    Reads the motion command's corpus clip, the robot's model file and
    default key pose, the task joint / tracked-body orders, and the control
    rate — all plain NumPy / config data the manager environment already
    publishes. Requires ``motrixsim`` for the one-time kinematic-tree load
    (always present behind a simulated environment).
    """
    motion_cfg = env.cfg.commands.motion
    robot = env.cfg.scene.objs.robot
    joint_names = tuple(motion_cfg.joint_names)
    tracked_body_names = tuple(motion_cfg.tracked_body_names)
    resolved = (robot.resolve_name(name) for name in robot.key_pose.joint_names)
    default_by_name = dict(zip(resolved, robot.key_pose.poses["default"], strict=True))
    missing = sorted(set(joint_names).difference(default_by_name))
    if missing:
        raise ValueError(f"Robot key pose is missing task joints: {missing}")
    default_joint_pos = np.asarray([default_by_name[name] for name in joint_names], dtype=np.float32)
    kinematics = SuperMimicKinematics(robot.model.file, joint_names, tracked_body_names)
    return SuperMimicTorchMDP(
        clip=env.command_terms["motion"].clip,
        kinematics=kinematics,
        joint_names=joint_names,
        tracked_body_names=tracked_body_names,
        default_joint_pos=default_joint_pos,
        obs_noise=obs_noise_table(len(joint_names)),
        ctrl_dt=env.cfg.ctrl_dt,
    )


__all__ = [
    "RewardCfg",
    "STATE_HEAD_DIM",
    "SuperMimicKinematics",
    "SuperMimicTorchMDP",
    "build_supermimic_mdp",
    "obs_noise_table",
]

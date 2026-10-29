# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-side APG parsing head for the g1-supermimic tracking task.

Implements the APG ``ApgTask`` protocol (see the provider's
``motrix_rl_builtin.apg.contracts``) duck-typed: no provider types or
training logic are imported — only the shared torch rotation/state-layout
math from ``motrix_env_core.torch.geometry``. It mirrors the manager-kernel
semantics of the task: the
differentiable forward kinematics of the tracked bodies, the analytic
reward / soft-termination / observation / goal / critic heads evaluated in
imagination, the context layout of the reset randomization, and the
env-surface reads collection needs (the motion command's state, frame,
context and endpoint buffers, the failure metrics, and the curriculum
sampling state). Cross-package parity with the manager kernels is covered
numerically by the tests in ``motrix_envs/tests``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.torch.geometry import (
    flat_state_dim,
    invariant_features,
    mat_to_6d_rows,
    quat_xyzw_to_mat,
    rot_angle_sq,
    yaw_mat,
)
from motrix_envs.locomotion.kinematics import ModelKinematics

# Upstream action convention: a wide clamp that never binds, so saturated
# policy outputs keep their gradients in imagination (a ±1 box would zero
# the gradient of every saturated dim).
ACTION_LIMIT = 20.0

# Simulation-context layout mirrored from the task's reset terms: friction
# scale, per-reset joint offsets (N), and torso CoM offset (3).
CONTEXT_HEAD_DIM = 1
CONTEXT_TAIL_DIM = 3


@dataclass(frozen=True)
class RewardCfg:
    """Kernel widths and penalties of the analytic reward.

    The values mirror the ``g1-supermimic`` manager reward group (upstream
    SuperMimic); cross-package parity is covered numerically by the
    integration test that compares this reward against the manager reward.
    """

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


class SuperMimicApgHead:
    """The g1-supermimic parsing head: analytic torch heads plus env reads.

    Attributes:
        num_joints: Task joint count ``N``.
        context_dim: Simulation-context width (friction + joint offsets + CoM).
        obs_dim: Proprioceptive observation width ``3N + 12``.
        goal_frame_dim: Goal width per future frame ``15 + 2N``.
        max_step_reward: Dt-free per-step reward ceiling (5.0).
        state_dim: Flattened state width ``13 + 2N`` (xyzw quaternion).
        motion: The manager motion command term (public for the task-side
            per-clip evaluator).
        clip_end: Per-frame corpus clip end indices.
    """

    def __init__(
        self,
        env,
        *,
        reward_cfg: RewardCfg | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        motion = getattr(env.command_terms, "get", lambda _name: None)("motion")
        if motion is None or not all(hasattr(motion, field) for field in ("state", "endpoint_state", "context")):
            raise TypeError(
                "The g1-supermimic APG head requires a 'motion' command term publishing the state, "
                "endpoint, frame, and simulation-context buffers."
            )
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
        context_dim = int(np.asarray(motion.context).shape[-1])
        expected = CONTEXT_HEAD_DIM + len(joint_names) + CONTEXT_TAIL_DIM
        if context_dim != expected:
            raise ValueError(
                f"The simulation context must hold friction + {len(joint_names)} joint offsets "
                f"+ CoM offset ({expected} dims), got {context_dim}."
            )

        self._env = env
        self.motion = motion
        self.device = torch.device(device)
        self.kin = ModelKinematics(robot.model.file, joint_names, tracked_body_names, device=device)
        clip = motion.clip
        self.num_joints = clip.joint_pos.shape[1]
        self.context_dim = int(context_dim)
        self.obs_dim = 3 * self.num_joints + 12
        self.goal_frame_dim = 15 + 2 * self.num_joints
        # The goal width follows the env's own goal observation term: the
        # motion command publishes ``num_goal_frames`` future frames.
        self.goal_frames = int(motion_cfg.num_goal_frames)
        self.goal_dim = self.goal_frame_dim * self.goal_frames
        self.state_dim = flat_state_dim(self.num_joints)
        self.max_step_reward = 5.0
        self.ctrl_dt = float(env.cfg.ctrl_dt)
        self.cfg = reward_cfg if reward_cfg is not None else RewardCfg()
        self.default_q = torch.tensor(default_joint_pos, dtype=torch.float32, device=self.device)
        self.obs_noise = torch.tensor(obs_noise_table(len(joint_names)), dtype=torch.float32, device=self.device)
        # Upstream's action convention: a wide clamp that never binds.
        self.action_low = np.full(self.num_joints, -ACTION_LIMIT, dtype=np.float32)
        self.action_high = np.full(self.num_joints, ACTION_LIMIT, dtype=np.float32)

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

    def to(self, device: torch.device | str) -> SuperMimicApgHead:
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

    def obs(self, state, previous_action: torch.Tensor, context: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """The proprioceptive observation, layout matching the manager policy group.

        Joint positions relative to the default pose, joint velocities,
        root-local angular velocity, gravity, previous action, and the
        relative rotation to the current reference frame (first-two-rows 6D).
        """
        _, rotation, _, angular_velocity, joint_pos, joint_vel = state
        rotation_t = rotation.transpose(-1, -2)
        gravity = -rotation[..., 2, :]
        return torch.cat(
            (
                joint_pos - self.default_q - context[..., 1 : 1 + self.num_joints],
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

    def goal(self, frames: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Future reference features in the current reference heading frame.

        The command rides in ``context`` for command-driven tasks; corpus
        tracking reconstructs goals from the reference alone, using the
        ``goal_frames`` width declared by the env's goal observation term.
        Per future frame (clamped at the owning clip's end): root position
        delta, relative root rotation 6D (rows), heading-local root
        velocities, reference joint positions and scaled velocities — the
        exact layout of the manager goal observation term.
        """
        num_frames = self.goal_frames
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

    def reward(
        self,
        state,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
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
                invariant_features(state),
                self.q[frames] - joint_pos,
                body_error,
                root_error,
                relative_rotation,
                normalized_context,
            ),
            -1,
        )

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        """Standardize the friction scale and CoM offset to ~unit range."""
        joints = self.num_joints
        return torch.cat(
            (
                (context[..., 0:1] - 0.75) / 0.45,
                context[..., 1 : 1 + joints] / 0.01,
                context[..., 1 + joints : 1 + joints + CONTEXT_TAIL_DIM] / 0.05,
            ),
            -1,
        )

    # ------------------------------------------------------------------
    # Collection hooks over the env's motion command
    # ------------------------------------------------------------------

    def current_transition(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The motion buffers of the step about to be taken."""
        return self.motion.state, self.motion.steps[:, 0], self.motion.context

    def endpoint_state(self) -> np.ndarray:
        """The physical endpoint state of the step just taken."""
        return self.motion.endpoint_state

    def split_observation(self, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy group is [proprioceptive head | goal tail]."""
        return full[..., : self.obs_dim], full[..., self.obs_dim :]

    def termination_counts(self, metrics: Mapping[str, np.ndarray]) -> float:
        """Reference failures: the union of the height and tilt checks."""
        hard = np.asarray(metrics["bad_ref_z"], dtype=bool) | np.asarray(metrics["bad_ref_ori"], dtype=bool)
        return float(hard.sum())

    def sampling_state(self):
        return self._env.supermimic_sampling_state()

    def load_sampling_state(self, state) -> None:
        self._env.load_supermimic_sampling_state(state)

    def sampling_metrics(self) -> Mapping[str, float]:
        return self._env.supermimic_sampling_metrics()


def build_head(env) -> SuperMimicApgHead:
    """APG head factory: assemble the parsing head from the environment."""
    return SuperMimicApgHead(env)


__all__ = [
    "ACTION_LIMIT",
    "CONTEXT_HEAD_DIM",
    "CONTEXT_TAIL_DIM",
    "RewardCfg",
    "SuperMimicApgHead",
    "build_head",
    "obs_noise_table",
]

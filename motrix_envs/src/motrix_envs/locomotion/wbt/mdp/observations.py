# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed observation terms."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.math import quaternion
from motrix_env_core.math.quaternion import relative_orientation, rotate_inverse
from motrix_env_core.mdp.observations import (
    UniformNoiseCfg,
    add_uniform_noise,
)
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


def motion_joint_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    joint_count = motion.clip.joint_pos.shape[1]
    out[:, :joint_count] = motion.clip.joint_pos[steps]
    out[:, joint_count:] = motion.clip.joint_vel[steps]


@configclass(kw_only=True)
class MotionJointObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        motion = ctx.command_terms["motion"]
        if not isinstance(motion, WbtMotionCommand):
            raise TypeError(f"WBT motion command must be WbtMotionCommand, got {type(motion).__name__}.")
        return ObsTerm(2 * motion.clip.joint_pos.shape[1], motion_joint_obs)


def motion_reference_position_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    reference_index = motion.reference_index
    reference_pos = ctx.sim["tracked_body_pos"][:, reference_index]
    reference_quat = ctx.sim["tracked_body_quat"][:, reference_index]
    local = rotate_inverse(
        reference_quat,
        motion.clip.reference_body_pos_w[motion.steps[:, 0]] - reference_pos,
    )
    out.copy_(local)


@configclass(kw_only=True)
class MotionReferencePositionObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, motion_reference_position_obs)


def motion_reference_orientation_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    noise_amplitude: float,
) -> None:
    motion = ctx.commands["motion"]
    reference_quat = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    local = relative_orientation(reference_quat, motion.clip.reference_body_quat_w[motion.steps[:, 0]])
    out.copy_(local)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class MotionReferenceOrientationObsCfg(ObservationTermCfg):
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6, motion_reference_orientation_obs, float(self.noise.amplitude))


def robot_body_position_in_reference_frame_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    reference_index = motion.reference_index
    reference_pos = ctx.sim["tracked_body_pos"][:, reference_index]
    local = rotate_inverse(
        ctx.sim["tracked_body_quat"][:, reference_index][:, None, :],
        ctx.sim["tracked_body_pos"] - reference_pos[:, None, :],
    )
    out.copy_(local.reshape(local.shape[0], -1))


@configclass(kw_only=True)
class RobotBodyPositionInReferenceFrameObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(3 * num_tracked, robot_body_position_in_reference_frame_obs)


def robot_body_orientation_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    # Bug-compatible with the historical per-lane oracle: body zero writes its
    # 6D matrix over the shared out[:4] inverse-quaternion scratch, and every
    # later body reads those four matrix entries as its reference quaternion.
    # Do not silently repair this alias; migrate the observation contract
    # explicitly.
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    reference_quat = tracked_body_quat[:, ctx.commands["motion"].reference_index]
    first = quaternion.matrix6d(quaternion.mul(quaternion.inverse(reference_quat), tracked_body_quat[:, 0, :]))
    rest = quaternion.matrix6d(quaternion.mul(first[:, None, :4], tracked_body_quat[:, 1:, :]))
    out.copy_(torch.cat((first[:, None, :], rest), dim=1).reshape(tracked_body_quat.shape[0], -1))


@configclass(kw_only=True)
class RobotBodyOrientationObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(6 * num_tracked, robot_body_orientation_obs)


@dataclass(frozen=True, slots=True)
class RelativePositionParams:
    reference: torch.Tensor
    noise_amplitude: float


def dof_pos_rel_obs(ctx: ManagerContext, out: torch.Tensor, params: RelativePositionParams) -> None:
    out.copy_(ctx.sim["robot_dof_pos"] - params.reference)
    add_uniform_noise(ctx, out, params.noise_amplitude)


@configclass(kw_only=True)
class DofPosRelObsCfg(ObservationTermCfg):
    """Joint positions relative to the scene robot's default key pose.

    Reads the WBT-owned ``robot_dof_pos`` query, which is declared in motion
    joint order; the reference pose is reordered to the same joint order from
    the robot's ``RobotCfg.key_pose``.
    """

    reference_key_pose: str = "default"
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        # The motion command's __call__() already validates that
        # ``robot_dof_pos`` is a JointPositionQuery in motion joint order.
        queried_joints = ctx.model.bodies["robot"].joint_names
        robot = ctx.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}.")
        robot_name = robot.resolved_base_link_name
        try:
            key_pose = np.asarray(robot.key_pose.poses[self.reference_key_pose], dtype=np.float32)
        except KeyError as error:
            raise ValueError(
                f"WBT key pose {self.reference_key_pose!r} is not defined on RobotCfg {robot_name!r}; "
                f"available poses: {sorted(robot.key_pose.poses)}."
            ) from error
        resolved_names = (robot.resolve_name(name) for name in robot.key_pose.joint_names)
        positions = dict(zip(resolved_names, key_pose, strict=True))
        missing = sorted(set(queried_joints).difference(positions))
        if missing:
            raise ValueError(f"RobotCfg {robot_name!r} key pose is missing queried joints: {missing}.")
        reference = np.asarray([positions[name] for name in queried_joints], dtype=np.float32)
        params = RelativePositionParams(torch.from_numpy(reference), float(self.noise.amplitude))
        return ObsTerm(reference.shape[0], dof_pos_rel_obs, params)


def dof_vel_obs(ctx: ManagerContext, out: torch.Tensor, noise_amplitude: float) -> None:
    out.copy_(ctx.sim["robot_dof_vel"])
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class DofVelObsCfg(ObservationTermCfg):
    """Joint velocities from the WBT-owned ``robot_dof_vel`` query."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        num_joints = len(ctx.model.bodies["robot"].joint_names)
        return ObsTerm(num_joints, dof_vel_obs, float(self.noise.amplitude))


@lru_cache(maxsize=16)
def _device_offsets(offsets: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Window offsets materialized once per (offsets, device) — a per-call
    ``torch.as_tensor(..., device=...)`` is a CPU->GPU copy every term
    invocation (and illegal inside CUDA-graph capture)."""
    return torch.tensor(offsets, device=device)


def motion_future_joint_obs(ctx: ManagerContext, out: torch.Tensor, offsets: tuple[int, ...]) -> None:
    """Future joint-target window: ``[qj, vj]`` per frame, clamped at the clip end."""
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    end = motion.clip.frame_clip_end[steps]
    frames = torch.minimum(steps[:, None] + _device_offsets(tuple(offsets), steps.device), end[:, None])
    stacked = torch.cat((motion.clip.joint_pos[frames], motion.clip.joint_vel[frames]), dim=-1)
    out.copy_(stacked.reshape(stacked.shape[0], -1))


@configclass(kw_only=True)
class MotionFutureJointObsCfg(ObservationTermCfg):
    """Upstream supermimic goal window: ``window`` future frames at ``stride``.

    Offsets run ``0, stride, ..., (window-1)*stride`` frames (stride 5 at
    50 Hz ≈ 0.1 s spacing, window 10 ≈ 0.9 s lookahead), each clamped at the
    current clip's end frame. Reference data carries no observation noise.
    """

    window: int = 10
    stride: int = 5

    def __call__(self, ctx) -> ObsTerm:
        motion = ctx.command_terms["motion"]
        if not isinstance(motion, WbtMotionCommand):
            raise TypeError(f"WBT motion command must be WbtMotionCommand, got {type(motion).__name__}.")
        offsets = tuple(self.stride * index for index in range(self.window))
        return ObsTerm(self.window * 2 * motion.clip.joint_pos.shape[1], motion_future_joint_obs, offsets)


def motion_future_reference_orientation_obs(ctx: ManagerContext, out: torch.Tensor, offsets: tuple[int, ...]) -> None:
    """Anchor-relative reference orientation window as 6D rotations."""
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    end = motion.clip.frame_clip_end[steps]
    frames = torch.minimum(steps[:, None] + _device_offsets(tuple(offsets), steps.device), end[:, None])
    anchor_quat = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    local = relative_orientation(anchor_quat[:, None, :], motion.clip.reference_body_quat_w[frames])
    out.copy_(local.reshape(local.shape[0], -1))


@configclass(kw_only=True)
class MotionFutureReferenceOrientationObsCfg(ObservationTermCfg):
    window: int = 10
    stride: int = 5

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(
            self.window * 6,
            motion_future_reference_orientation_obs,
            tuple(self.stride * index for index in range(self.window)),
        )


def motion_future_reference_position_obs(ctx: ManagerContext, out: torch.Tensor, offsets: tuple[int, ...]) -> None:
    """Anchor-relative reference position window (privileged value input)."""
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    end = motion.clip.frame_clip_end[steps]
    frames = torch.minimum(steps[:, None] + _device_offsets(tuple(offsets), steps.device), end[:, None])
    anchor_pos = ctx.sim["tracked_body_pos"][:, motion.reference_index]
    anchor_quat = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    local = rotate_inverse(
        anchor_quat[:, None, :],
        motion.clip.reference_body_pos_w[frames] - anchor_pos[:, None, :],
    )
    out.copy_(local.reshape(local.shape[0], -1))


@configclass(kw_only=True)
class MotionFutureReferencePositionObsCfg(ObservationTermCfg):
    window: int = 10
    stride: int = 5

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(
            self.window * 3,
            motion_future_reference_position_obs,
            tuple(self.stride * index for index in range(self.window)),
        )


def projected_gravity_obs(ctx: ManagerContext, out: torch.Tensor, noise_amplitude: float) -> None:
    """Gravity direction in the anchor body frame."""
    anchor_quat = ctx.sim["tracked_body_quat"][:, ctx.commands["motion"].reference_index]
    gravity = out.new_zeros(out.shape)
    gravity[:, 2] = -1.0
    out.copy_(rotate_inverse(anchor_quat, gravity))
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class ProjectedGravityObsCfg(ObservationTermCfg):
    noise: UniformNoiseCfg = UniformNoiseCfg(amplitude=0.05)

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, projected_gravity_obs, float(self.noise.amplitude))


def soft_termination_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    z_threshold: float,
    ori_threshold: float,
    softness: float,
) -> None:
    """Soft termination probability of the tracking terminations.

    Upstream supermimic's world-model termination: the anchor z error and
    the tilt error (both the same quantities the hard ``bad_ref_z`` /
    ``bad_ref_orientation`` terms threshold) are normalized by their
    thresholds, and the max feeds one sigmoid. The trainer consumes this
    value-group channel as the imagination survival gate.
    """
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    reference_pos = motion.clip.reference_body_pos_w[steps]
    reference_quat = motion.clip.reference_body_quat_w[steps]
    anchor_pos = ctx.sim["tracked_body_pos"][:, motion.reference_index]
    anchor_quat = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    z_error = (reference_pos[..., 2] - anchor_pos[..., 2]).abs()
    # Gravity's z in a body frame is 2(x^2 + y^2) - 1 for a unit quaternion.
    tilt_error = (
        2.0 * (reference_quat[..., 0] ** 2 + reference_quat[..., 1] ** 2)
        - 2.0 * (anchor_quat[..., 0] ** 2 + anchor_quat[..., 1] ** 2)
    ).abs()
    normalized = torch.maximum(z_error / z_threshold, tilt_error / ori_threshold)
    out.copy_(torch.sigmoid((normalized - 1.0) / softness).unsqueeze(-1))


@configclass(kw_only=True)
class SoftTerminationObsCfg(ObservationTermCfg):
    """Soft tracking-termination probability as a one-dim value-group term.

    Defaults mirror the supermimic variant's hard termination thresholds
    (reference z 0.25, orientation 0.8) and upstream's termination softness
    (0.05).
    """

    z_threshold: float = 0.25
    ori_threshold: float = 0.8
    softness: float = 0.05

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(
            1,
            soft_termination_obs,
            float(self.z_threshold),
            float(self.ori_threshold),
            float(self.softness),
        )

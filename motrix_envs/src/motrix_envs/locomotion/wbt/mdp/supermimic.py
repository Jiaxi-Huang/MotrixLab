# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic motion-only observations and episode boundary semantics."""

from dataclasses import dataclass, fields

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.manager.terminations import TerminationTerm, TerminationTermCfg
from motrix_env_core.manager.truncations import TruncationTerm, TruncationTermCfg
from motrix_env_core.math import quaternion as quat
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand, WbtMotionCommandCfg


@dataclass(frozen=True, slots=True)
class MotionCommand(WbtMotionCommand):
    # These are state, not diagnostic gauges: replay and FinalState own them.
    steps: torch.Tensor
    clip_ended: torch.Tensor
    frame_end: torch.Tensor

    def training_state_dict(self) -> dict[str, torch.Tensor]:
        # Sampler owns bin/global statistics. Open lane accumulators are
        # deliberately restarted because simulator episodes are not restored.
        return {
            field.name: getattr(self.sampler, field.name).clone()
            for field in fields(self.sampler)
            if isinstance(getattr(self.sampler, field.name), torch.Tensor) and not field.name.startswith("lane_")
        }

    def load_training_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = self.training_state_dict()
        if state.keys() != expected.keys():
            raise ValueError("Motion sampler checkpoint does not match configured sampler")
        for name, value in state.items():
            target = getattr(self.sampler, name)
            if target.shape != value.shape:
                raise ValueError(f"Motion sampler corpus changed: {name}")
            target.copy_(value)

    def update(self, ctx: ManagerContext) -> None:
        indices = torch.minimum(self.steps, self.frame_end)[:, 0]
        pos, rot = quat.align_targets(
            ctx.sim["tracked_body_pos"][:, self.reference_index],
            ctx.sim["tracked_body_quat"][:, self.reference_index],
            self.clip.reference_body_pos_w[indices],
            self.clip.reference_body_quat_w[indices],
            self.clip.tracked_bodies_pos_w[indices],
            self.clip.tracked_bodies_quat_w[indices],
        )
        self.target_body_position_relative.copy_(pos)
        self.target_body_orientation_relative.copy_(rot)

    def advance(self, ctx: ManagerContext) -> None:
        self.sampler.fold_step(ctx)
        # The final frame is evaluated before this increment. Truncation owns
        # the reset; no inline clip resampling can destroy the true successor.
        self.steps.copy_(torch.minimum(self.steps + 1, self.frame_end))

    def reset(self, ctx: ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        WbtMotionCommand.reset(self, ctx, env_ids, episode_reset)
        self.frame_end[env_ids, 0] = self.clip.frame_clip_end[self.steps[env_ids, 0]]


@configclass(kw_only=True)
class MotionCommandCfg(WbtMotionCommandCfg):
    def __call__(self, env) -> MotionCommand:
        base = super().__call__(env)
        return MotionCommand(
            **{field.name: getattr(base, field.name) for field in fields(base)},
            frame_end=torch.zeros_like(base.steps),
        )


def clip_end(ctx: ManagerContext) -> torch.Tensor:
    motion = ctx.commands["motion"]
    return (motion.steps >= motion.frame_end)[:, 0]


@configclass(kw_only=True)
class ClipEndCfg(TruncationTermCfg):
    def __call__(self, ctx) -> TruncationTerm:
        return TruncationTerm(clip_end)


def proprio(ctx: ManagerContext, out: torch.Tensor, noise: torch.Tensor) -> None:
    rotation = ctx.sim["root_quat"]
    down = torch.zeros_like(ctx.sim["root_pos"])
    down[:, 2] = -1
    action = ctx.actions["joint_position"]
    out.copy_(
        torch.cat(
            [
                quat.rotate_inverse(rotation, down),
                quat.rotate_inverse(rotation, ctx.sim["root_ang_vel"]),
                ctx.sim["robot_dof_pos"] - action.default_angles,
                ctx.sim["robot_dof_vel"],
                action.current(),
            ],
            dim=-1,
        )
    )
    out.add_((2 * ctx.rand.uniform_like(out) - 1) * noise)


@configclass(kw_only=True)
class ProprioCfg(ObservationTermCfg):
    noisy: bool = True

    def __call__(self, ctx) -> ObsTerm:
        n = ctx.actions["joint_position"].action_space.shape[0]
        noise = torch.tensor([0.05] * 3 + [0.2] * 3 + [0.01] * n + [0.5] * n + [0.0] * n)
        if not self.noisy:
            noise.zero_()
        return ObsTerm(6 + 3 * n, proprio, noise)


def motion_goal(ctx: ManagerContext, out: torch.Tensor, offsets: torch.Tensor, privileged: bool) -> None:
    motion = ctx.commands["motion"]
    indices = torch.minimum(motion.steps + offsets, motion.frame_end)
    clip = motion.clip
    content = torch.cat([clip.joint_pos[indices], clip.joint_vel[indices]], dim=-1).flatten(1)
    root_quat = ctx.sim["root_quat"][:, None]
    relative = quat.mul(quat.inverse(root_quat), clip.root_body_quat_w[indices])
    # SuperMimic uses first two COLUMNS, row-major interleaving, unlike the
    # historical WBT matrix6d helper (first two rows).
    orientation = quat.to_matrix(relative)[..., :2].flatten(1)
    parts = [content]
    if privileged:
        parts.append(
            quat.rotate_inverse(root_quat, clip.root_body_pos_w[indices] - ctx.sim["root_pos"][:, None]).flatten(1)
        )
    parts.append(orientation)
    out.copy_(torch.cat(parts, dim=-1))


@configclass(kw_only=True)
class MotionGoalCfg(ObservationTermCfg):
    frames: int = 10
    stride: int = 5
    privileged: bool = False

    def __call__(self, ctx) -> ObsTerm:
        width = self.frames * (
            2 * ctx.actions["joint_position"].action_space.shape[0] + 6 + (3 if self.privileged else 0)
        )
        return ObsTerm(width, motion_goal, torch.arange(self.frames) * self.stride, self.privileged)


# The real and modeled boundary share the same task-owned error and threshold.
def height_error(ctx: ManagerContext) -> torch.Tensor:
    motion = ctx.commands["motion"]
    return (motion.clip.reference_body_pos_w[motion.steps[:, 0], 2] - ctx.sim["root_pos"][:, 2]).abs()


def tilt_error(ctx: ManagerContext) -> torch.Tensor:
    motion = ctx.commands["motion"]
    target = motion.clip.reference_body_quat_w[motion.steps[:, 0]]
    actual = ctx.sim["root_quat"]
    return (2 * (target[:, 0].square() + target[:, 1].square() - actual[:, 0].square() - actual[:, 1].square())).abs()


def height_done(ctx: ManagerContext, threshold: float, temperature: float) -> torch.Tensor:
    error = height_error(ctx)
    ctx.metrics["ref_z_abs_err"][:, 0] = error
    return error > threshold


def height_probability(ctx: ManagerContext, threshold: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((height_error(ctx) - threshold) / temperature)


def tilt_done(ctx: ManagerContext, threshold: float, temperature: float) -> torch.Tensor:
    error = tilt_error(ctx)
    ctx.metrics["ref_ori_abs_err"][:, 0] = error
    return error > threshold


def tilt_probability(ctx: ManagerContext, threshold: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((tilt_error(ctx) - threshold) / temperature)


@configclass(kw_only=True)
class HeightTerminationCfg(TerminationTermCfg):
    threshold: float = 0.25
    temperature: float = 0.02

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            height_done,
            self.threshold,
            self.temperature,
            probability_dispatch=height_probability,
            metric_names=("ref_z_abs_err",),
        )


@configclass(kw_only=True)
class TiltTerminationCfg(TerminationTermCfg):
    threshold: float = 0.8
    temperature: float = 0.04

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            tilt_done,
            self.threshold,
            self.temperature,
            probability_dispatch=tilt_probability,
            metric_names=("ref_ori_abs_err",),
        )

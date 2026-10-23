# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dynamics-randomization reset terms for the SuperMimic task.

Mirrors upstream SuperMimic's simulation context: every lane draws one robot
collision-friction scale in ``[0.3, 1.2]`` that stays fixed for the lane's
lifetime (upstream samples it once at startup and replays it on resets), the
ground slides at a fixed 0.3 during training, and evaluation pins everything
to the nominal 1.0. Contact friction combines both surfaces, so writing the
robot's collision geoms is what actually randomizes the contact. The sampled
values are published into the motion command's simulation-context buffer so
the differentiable world model and the critic stay conditioned on the lane's
dynamics (upstream context xi, first slot). Degenerate settings (eval mode)
write the nominal values back.
"""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ResetTerm, ResetTermCfg
from motrix_env_core.numba.kernel_data import Map, SharedArray, kernel_data
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import GeomFrictionWrite, LinkComWrite

# The G1's contact geoms, from ``motrix_robots``' ``g1_29dof.xml`` (every
# geom in the ``collision`` class is explicitly named there). Visual geoms
# carry ``contype="0"`` and never enter contacts.
ROBOT_COLLISION_GEOMS = (
    "pelvis_collision",
    "torso_collision1",
    "torso_collision2",
    "torso_collision3",
    "head_collision",
    "left_thigh",
    "left_shin",
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
    "left_foot4_collision",
    "left_foot5_collision",
    "left_foot6_collision",
    "left_foot7_collision",
    "right_thigh",
    "right_shin",
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
    "right_foot4_collision",
    "right_foot5_collision",
    "right_foot6_collision",
    "right_foot7_collision",
    "left_shoulder_yaw_collision",
    "left_elbow_yaw_collision",
    "left_hand_collision",
    "right_shoulder_yaw_collision",
    "right_elbow_yaw_collision",
    "right_hand_collision",
)


@kernel_data
class SuperMimicDynamicsParams:
    """Static nominal values, per-lane friction draws, and sampling ranges.

    Attributes:
        robot_friction_default: Nominal robot collision friction, ``(3,)``.
        ground_friction_default: Nominal ground friction parameters, ``(3,)``.
        lane_friction: Per-lane robot sliding friction, ``(N,)``, sampled once
            at build time and fixed for the lane's lifetime.
        com_default: Nominal torso CoM offset, ``(3,)``.
        com_noise: Per-axis uniform CoM offset widths ``(x, y, z)`` in m.
        eval_scale: Friction scale forced in eval mode (1.0); 0.0 disables
            the override in training.
    """

    robot_friction_default: SharedArray
    ground_friction_default: SharedArray
    lane_friction: SharedArray
    com_default: SharedArray
    com_noise: np.ndarray
    eval_scale: np.float32


@dispatch
def supermimic_dynamics_reset(
    ctx: ManagerContext, sim_writes: Map[np.ndarray], params: SuperMimicDynamicsParams
) -> None:
    """Write one lane's dynamics context into the sim writes and the command."""
    motion = ctx.commands["motion"]
    scale = params.lane_friction[ctx.env_id]
    if params.eval_scale > 0.0:
        scale = params.eval_scale
    robot_friction = sim_writes["robot_friction"]
    for geom in range(robot_friction.shape[0]):
        robot_friction[geom, 0] = scale
        robot_friction[geom, 1] = params.robot_friction_default[1]
        robot_friction[geom, 2] = params.robot_friction_default[2]
    ground_friction = sim_writes["ground_friction"]
    if params.eval_scale > 0.0:
        for i in range(3):
            ground_friction[0, i] = params.ground_friction_default[i]
    else:
        ground_friction[0, 0] = np.float32(0.3)
        ground_friction[0, 1] = params.ground_friction_default[1]
        ground_friction[0, 2] = params.ground_friction_default[2]
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 0.0
    if params.com_noise[0] > 0.0 or params.com_noise[1] > 0.0 or params.com_noise[2] > 0.0:
        rand = ctx.rand
        offset_x = rand.uniform_range(-params.com_noise[0], params.com_noise[0])
        offset_y = rand.uniform_range(-params.com_noise[1], params.com_noise[1])
        offset_z = rand.uniform_range(-params.com_noise[2], params.com_noise[2])
    com = sim_writes["com"]
    com[0, 0] = params.com_default[0] + offset_x
    com[0, 1] = params.com_default[1] + offset_y
    com[0, 2] = params.com_default[2] + offset_z
    context = motion.context
    context[0] = scale
    context[1] = offset_x
    context[2] = offset_y
    context[3] = offset_z


@configclass(kw_only=True)
class SuperMimicDynamicsResetCfg(ResetTermCfg):
    """Reset-time dynamics randomization with world-model context publication.

    Requires the model queries ``supermimic_friction_default`` (nominal ground
    friction), ``supermimic_robot_friction_default`` (nominal robot collision
    friction), and ``supermimic_torso_com`` (nominal torso CoM), declared by
    the task cfg when this term is enabled.
    """

    ground_geom: str = "floor"
    com_link: str = "torso_link"
    friction_range: tuple[float, float] = (0.3, 1.2)
    com_offset_noise: tuple[float, float, float] = (0.025, 0.05, 0.05)
    eval_mode: bool = False

    def __call__(self, ctx: BuildContext) -> ResetTerm:
        others = ctx.model.others
        low, high = self.friction_range
        lane_friction = np.random.default_rng(0).uniform(low, high, ctx.num_envs).astype(np.float32)
        params = SuperMimicDynamicsParams(
            robot_friction_default=np.asarray(others["supermimic_robot_friction_default"], dtype=np.float32),
            ground_friction_default=np.asarray(others["supermimic_friction_default"], dtype=np.float32),
            lane_friction=lane_friction,
            com_default=np.asarray(others["supermimic_torso_com"], dtype=np.float32),
            com_noise=np.asarray(self.com_offset_noise, dtype=np.float32),
            eval_scale=np.float32(1.0 if self.eval_mode else 0.0),
        )
        return ResetTerm(
            supermimic_dynamics_reset,
            params,
            writes={
                "robot_friction": GeomFrictionWrite(ROBOT_COLLISION_GEOMS),
                "ground_friction": GeomFrictionWrite((self.ground_geom,)),
                "com": LinkComWrite((self.com_link,)),
            },
        )


__all__ = ["ROBOT_COLLISION_GEOMS", "SuperMimicDynamicsResetCfg"]

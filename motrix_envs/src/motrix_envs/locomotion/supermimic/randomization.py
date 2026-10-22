# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dynamics-randomization reset terms for the SuperMimic task.

Uses MotrixLab's reset-time simulation-write machinery (the same surface as
the humanoid-walk randomization): per-lane uniform samples resampled every
episode reset and written through ``GeomFrictionWrite`` / ``LinkComWrite``.
The sampled values are additionally published into the motion command's
simulation-context buffer so the differentiable world model and the critic
stay conditioned on the lane's dynamics (upstream SuperMimic context xi).

Upstream xi randomizes the robot collision-geom friction absolutely in
[0.3, 1.2] and the torso CoM offset by ±(0.025, 0.05, 0.05); here the ground
geom's friction vector is scaled by a uniform factor in the same range
(contact friction combines both surfaces), and the torso CoM keeps the
upstream per-axis offsets. Degenerate ranges disable an item and write the
nominal values back (used by play mode).
"""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ResetTerm, ResetTermCfg
from motrix_env_core.numba.kernel_data import Map, SharedArray, kernel_data
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import GeomFrictionWrite, LinkComWrite

# Default nominal context: friction scale 1, zero CoM offset.
_NOMINAL_CONTEXT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@kernel_data
class SuperMimicDynamicsParams:
    """Static nominal values and per-item sampling ranges.

    Attributes:
        friction_default: Nominal ground friction parameters, ``(3,)``.
        com_default: Nominal torso CoM offset, ``(3,)``.
        friction_range: ``(min, max)`` friction scale; degenerate disables.
        com_noise: Per-axis uniform CoM offset widths ``(x, y, z)`` in m.
    """

    friction_default: SharedArray
    com_default: SharedArray
    friction_range: np.ndarray
    com_noise: np.ndarray


@dispatch
def supermimic_dynamics_reset(
    ctx: ManagerContext, sim_writes: Map[np.ndarray], params: SuperMimicDynamicsParams
) -> None:
    """Sample one lane's dynamics context into the sim writes and the command."""
    rand = ctx.rand
    motion = ctx.commands["motion"]
    scale = 1.0
    if params.friction_range[1] > params.friction_range[0]:
        scale = rand.uniform_range(params.friction_range[0], params.friction_range[1])
    friction = sim_writes["friction"]
    for i in range(3):
        friction[0, i] = params.friction_default[i] * scale
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 0.0
    if params.com_noise[0] > 0.0 or params.com_noise[1] > 0.0 or params.com_noise[2] > 0.0:
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
    friction) and ``supermimic_torso_com`` (nominal torso CoM), declared by
    the task cfg when this term is enabled.
    """

    ground_geom: str = "floor"
    com_link: str = "torso_link"
    friction_range: tuple[float, float] = (0.3, 1.2)
    com_offset_noise: tuple[float, float, float] = (0.025, 0.05, 0.05)

    def __call__(self, ctx: BuildContext) -> ResetTerm:
        others = ctx.model.others
        params = SuperMimicDynamicsParams(
            friction_default=np.asarray(others["supermimic_friction_default"], dtype=np.float32),
            com_default=np.asarray(others["supermimic_torso_com"], dtype=np.float32),
            friction_range=np.asarray(self.friction_range, dtype=np.float32),
            com_noise=np.asarray(self.com_offset_noise, dtype=np.float32),
        )
        return ResetTerm(
            supermimic_dynamics_reset,
            params,
            writes={
                "friction": GeomFrictionWrite((self.ground_geom,)),
                "com": LinkComWrite((self.com_link,)),
            },
        )


__all__ = ["SuperMimicDynamicsResetCfg"]

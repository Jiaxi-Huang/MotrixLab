# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Upstream-aligned SuperMimic dynamics context reset terms.

Effective ground-contact friction and torso CoM are sampled once per lane,
then replayed on episode reset. Joint offsets are a separate per-reset
randomization in the same 33-D context:
``xi = [friction, joint_offset[29], com_offset[3]]``.
"""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ResetTerm, ResetTermCfg
from motrix_env_core.numba.kernel_data import Map, SharedArray, kernel_data
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import GeomFrictionWrite, LinkComWrite
from motrix_envs.locomotion.supermimic.mdp import SUPERMIMIC_COM_OFFSET_START


@kernel_data
class SuperMimicDynamicsParams:
    ground_friction_default: SharedArray
    lane_friction: SharedArray
    lane_com: SharedArray
    com_default: SharedArray
    eval_mode: bool


@dispatch
def supermimic_dynamics_reset(
    ctx: ManagerContext, sim_writes: Map[np.ndarray], params: SuperMimicDynamicsParams
) -> None:
    """Replay startup dynamics and publish the fixed lane context."""
    motion = ctx.commands["motion"]
    if params.eval_mode:
        friction = np.float32(1.0)
        com_x = np.float32(0.0)
        com_y = np.float32(0.0)
        com_z = np.float32(0.0)
    else:
        friction = params.lane_friction[ctx.env_id]
        com_x = params.lane_com[ctx.env_id, 0]
        com_y = params.lane_com[ctx.env_id, 1]
        com_z = params.lane_com[ctx.env_id, 2]

    ground_friction = sim_writes["ground_friction"]
    ground_friction[0, 0] = friction
    ground_friction[0, 1] = params.ground_friction_default[1]
    ground_friction[0, 2] = params.ground_friction_default[2]

    com = sim_writes["com"]
    com[0, 0] = params.com_default[0] + com_x
    com[0, 1] = params.com_default[1] + com_y
    com[0, 2] = params.com_default[2] + com_z

    motion.context[0] = friction
    motion.context[SUPERMIMIC_COM_OFFSET_START] = com_x
    motion.context[SUPERMIMIC_COM_OFFSET_START + 1] = com_y
    motion.context[SUPERMIMIC_COM_OFFSET_START + 2] = com_z


@configclass(kw_only=True)
class SuperMimicDynamicsResetCfg(ResetTermCfg):
    """Replay per-lane friction/CoM context on every reset."""

    ground_geom: str = "floor"
    com_link: str = "torso_link"
    friction_range: tuple[float, float] = (0.3, 1.2)
    com_offset_noise: tuple[float, float, float] = (0.025, 0.05, 0.05)
    eval_mode: bool = False

    def __call__(self, ctx: BuildContext) -> ResetTerm:
        others = ctx.model.others
        low, high = self.friction_range
        rng = np.random.default_rng(0)
        lane_friction = rng.uniform(low, high, ctx.num_envs).astype(np.float32)
        lane_com = rng.uniform(
            -np.asarray(self.com_offset_noise, dtype=np.float32),
            np.asarray(self.com_offset_noise, dtype=np.float32),
            (ctx.num_envs, 3),
        ).astype(np.float32)
        params = SuperMimicDynamicsParams(
            ground_friction_default=np.asarray(others["supermimic_friction_default"], dtype=np.float32),
            lane_friction=lane_friction,
            lane_com=lane_com,
            com_default=np.asarray(others["supermimic_torso_com"], dtype=np.float32),
            eval_mode=self.eval_mode,
        )
        return ResetTerm(
            supermimic_dynamics_reset,
            params,
            writes={
                "ground_friction": GeomFrictionWrite((self.ground_geom,)),
                "com": LinkComWrite((self.com_link,)),
            },
        )


__all__ = ["SuperMimicDynamicsResetCfg"]

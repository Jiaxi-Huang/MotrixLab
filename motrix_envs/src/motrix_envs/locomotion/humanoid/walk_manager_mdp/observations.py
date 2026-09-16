# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the humanoid velocity-tracking task."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.mdp.terrain import heightfield_lookup
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.sim import LinkPositionQuery, LinkQuaternionQuery
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommand
from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid


@dispatch
def gait_phase_obs(ctx: ManagerContext, out: np.ndarray, offset: np.int64, size: np.int64) -> None:
    walk: WalkCommand = ctx.commands["walk"]
    for index in range(size):
        out[index] = walk.sin_cos[offset + index]


@configclass(kw_only=True)
class GaitPhaseObsCfg(ObservationTermCfg):
    """``sin``/``cos`` slice of the gait-phase clock.

    The command term's ``sin_cos`` lane layout is
    ``[sin_l, sin_r, cos_l, cos_r]``: offset 0 gives the two sin values and
    offset 2 the two cos values, matching the direct env's obs order.
    """

    offset: int = 0
    size: int = 2

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(self.size, gait_phase_obs, np.int64(self.offset), np.int64(self.size))


@dispatch
def terrain_height_scan_obs(
    ctx: ManagerContext,
    out: np.ndarray,
    grid,  # HeightFieldGrid
    offsets: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    scale: np.float32,
) -> None:
    """Sample the height-field at base-frame offsets, relative to base height."""
    x, y, z, w = base_quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    for index in range(offsets.shape[0]):
        dx = offsets[index, 0]
        dy = offsets[index, 1]
        wx = base_pos[0] + cy * dx - sy * dy
        wy = base_pos[1] + sy * dx + cy * dy
        out[index] = (heightfield_lookup(grid, wx, wy) - base_pos[2]) * scale


@configclass(kw_only=True)
class TerrainHeightScanObsCfg(ObservationTermCfg):
    """Terrain height samples around the base, in the base yaw frame.

    Each sample point ``(dx, dy)`` is rotated by the base yaw and looked up in
    the ground height-field; the value is the terrain height relative to the
    base link height, scaled by ``scale``. On flat presets the grid degenerates
    to the constant ground height, so the scan reads ``-base_z`` everywhere —
    wire it only for rough/stair presets that export a height-field.
    """

    body: str = "robot"
    # Base-frame (dx, dy) sample offsets, ordered row-major along +x (forward).
    pattern: list[list[float]] = [
        [0.00, 0.00],
        [0.06, -0.06],
        [0.06, 0.00],
        [0.06, 0.06],
        [0.12, -0.06],
        [0.12, 0.00],
        [0.12, 0.06],
        [0.18, -0.06],
        [0.18, 0.00],
        [0.18, 0.06],
    ]
    scale: float = 1.0
    ground_geom: str = ""

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        if not self.ground_geom:
            raise ValueError("TerrainHeightScanObsCfg requires a non-empty ground_geom.")
        offsets = np.asarray(self.pattern, dtype=np.float32)
        body = ctx.model.bodies[self.body]
        return ObsTerm(
            offsets.shape[0],
            terrain_height_scan_obs,
            ground_height_grid(ctx, self.ground_geom),
            offsets,
            LinkPositionQuery(link=body.base_link_name),
            LinkQuaternionQuery(link=body.base_link_name),
            np.float32(self.scale),
        )

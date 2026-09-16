# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable termination terms for manager-based environments."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.math.quaternion import rotate_inverse_components
from motrix_env_core.sim import GeomPairCollidingQuery, LinkQuaternionQuery


@dispatch
def colliding_termination(ctx: ManagerContext, colliding: np.ndarray) -> bool:
    return bool(colliding.any())


@configclass(kw_only=True)
class CollidingTerminationCfg(TerminationTermCfg):
    """Terminate when any of ``termination_geoms`` contacts ``ground_geom``.

    The collision query is passed as an argument; the dispatch receives the
    query's lane view (one bool per declared geom pair).
    """

    termination_geoms: tuple[str, ...] = ()
    ground_geom: str = ""

    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        if not self.termination_geoms or not self.ground_geom:
            raise ValueError("CollidingTerminationCfg requires non-empty termination_geoms and ground_geom.")
        query = GeomPairCollidingQuery(pairs=tuple((name, self.ground_geom) for name in self.termination_geoms))
        return TerminationTerm(colliding_termination, query)


@dispatch
def bad_orientation_termination(ctx: ManagerContext, base_quat: np.ndarray, gravity_z_limit: np.float32) -> bool:
    _, _, gz = rotate_inverse_components(base_quat, (0.0, 0.0, -1.0))
    # Upright reads gz = -1; tipping raises gz toward 0, so "too tilted" is
    # gz above the -cos(tilt) limit.
    return gz > gravity_z_limit


@configclass(kw_only=True)
class BadOrientationTerminationCfg(TerminationTermCfg):
    """Terminate when the base tilts past ``tilt_degrees`` from upright.

    Uses the projected gravity z of the base link: upright is ``-1``, and
    ``cos(tilt)`` falls off as the base tips over. This catches falls earlier
    than body-contact terminations, which only fire once the trunk scrapes the
    ground.
    """

    body: str = "robot"
    tilt_degrees: float = 30.0

    def __call__(self, ctx: BuildContext) -> TerminationTerm:
        limit = -math.cos(math.radians(self.tilt_degrees))
        link = ctx.model.bodies[self.body].base_link_name
        return TerminationTerm(
            bad_orientation_termination,
            LinkQuaternionQuery(link=link),
            np.float32(limit),
        )


__all__ = [
    "BadOrientationTerminationCfg",
    "CollidingTerminationCfg",
    "colliding_termination",
]

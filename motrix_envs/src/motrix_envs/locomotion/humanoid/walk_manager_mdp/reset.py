# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for the humanoid velocity-tracking task."""

import math

import numpy as np
from numba import literally, njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ResetTerm,
    ResetTermCfg,
    kernel_data,
)
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.numba.kernel_data.map import Map
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    JointPositionWrite,
    JointVelocityWrite,
)


@kernel_data
class WalkResetParams:
    """Reset parameters, including in-kernel rough-terrain spawn sampling.

    When the effective spawn range is positive, each lane samples its world xy
    from a uniform range and lifts the base above the highest terrain height
    in a +/-0.15 m 9-point grid around the spawn point (bilinear lookups on
    the static height-field grid). With ``spawn_flatness_tol > 0``, sampling
    retries up to ``spawn_attempts`` times and keeps the flattest candidate,
    giving flat-patch spawns on rough terrain. With ``spawn_curriculum``, the
    effective range ramps from zero to ``spawn_range`` as the walk command's
    average episode length grows from ``ramp_min_ep_len`` to
    ``ramp_max_ep_len``, so early episodes spawn on the flat center platform.
    """

    default_joint_angles: np.ndarray
    init_pose: np.ndarray
    heightfield: HeightFieldGrid
    spawn_range: np.float32
    spawn_flatness_tol: np.float32
    spawn_attempts: np.int64
    spawn_curriculum: bool
    ramp_min_ep_len: np.float32
    ramp_max_ep_len: np.float32


@njit(inline="always")
def _spawn_patch_bounds(grid: HeightFieldGrid, x: float, y: float) -> tuple[float, float]:
    """Lowest and highest terrain height over the +/-0.15 m 9-point grid."""
    lo = math.inf
    hi = -math.inf
    for i in range(3):
        for j in range(3):
            h = heightfield_lookup(grid, x + (i - 1) * 0.15, y + (j - 1) * 0.15)
            lo = min(lo, h)
            hi = max(hi, h)
    return lo, hi


@njit(inline="always")
def _spawn_ground_height(grid: HeightFieldGrid, x: float, y: float) -> float:
    """Highest terrain height over the +/-0.15 m 9-point spawn grid."""
    _, hi = _spawn_patch_bounds(grid, x, y)
    return hi


@njit(inline="always")
def _lane_sample_spawn(ctx, grid: HeightFieldGrid, spawn_range: np.float32, tol: np.float32, attempts: np.int64):
    """Sample the flattest spawn candidate within ``attempts`` tries."""
    rand = ctx.rand
    best_x = 0.0
    best_y = 0.0
    best_diff = math.inf
    for _ in range(attempts):
        x = rand.uniform_range(-spawn_range, spawn_range)
        y = rand.uniform_range(-spawn_range, spawn_range)
        lo, hi = _spawn_patch_bounds(grid, x, y)
        diff = hi - lo
        if diff < best_diff:
            best_diff = diff
            best_x = x
            best_y = y
        if diff <= tol:
            break
    return best_x, best_y


@dispatch
def reset_walk_state(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams, command_name) -> None:
    command_name = literally(command_name)
    pose = params.init_pose
    x, y, z = pose[0], pose[1], pose[2]
    spawn_range = params.spawn_range
    if params.spawn_curriculum:
        walk = ctx.commands[command_name]
        scale = (walk.avg_ep_len[0] - params.ramp_min_ep_len) / (params.ramp_max_ep_len - params.ramp_min_ep_len)
        scale = min(max(scale, 0.0), 1.0)
        spawn_range = np.float32(spawn_range * scale)
    if spawn_range > 0.0:
        x, y = _lane_sample_spawn(
            ctx, params.heightfield, spawn_range, params.spawn_flatness_tol, params.spawn_attempts
        )
    # Always lift the base above the local terrain: the default-pose z assumes
    # a flat world origin, which does not hold on stair platforms.
    z += _spawn_ground_height(params.heightfield, x, y)
    position = sim_writes["position"]
    position[0, 0] = x
    position[0, 1] = y
    position[0, 2] = z
    sim_writes["rotation"][0] = pose[3:]
    sim_writes["linear_velocity"][0, :] = 0.0
    sim_writes["angular_velocity"][0, :] = 0.0
    sim_writes["joints_position"][:] = params.default_joint_angles
    sim_writes["joints_velocity"][:] = 0.0


@configclass(kw_only=True)
class WalkStateResetCfg(ResetTermCfg):
    """Reset the floating base to the sampled spawn pose, joints to default.

    ``spawn_xy_range > 0`` samples each lane's world xy uniformly and lifts
    the base above the terrain. ``spawn_flatness_tol``/``spawn_attempts``
    bias sampling toward flat patches (flat-patch spawn). With
    ``spawn_curriculum``, the effective range ramps from zero to
    ``spawn_xy_range`` as the command term's average episode length grows
    from ``spawn_ramp_ep_len[0]`` to ``[1]``. ``ground_geom`` names the floor
    geom used for terrain height lookups; ``command_name`` supplies the
    curriculum signal.
    """

    spawn_xy_range: float = 0.0
    spawn_flatness_tol: float = 0.02
    spawn_attempts: int = 16
    spawn_curriculum: bool = False
    spawn_ramp_ep_len: tuple[float, float] = (100.0, 500.0)
    command_name: str = "walk"
    ground_geom: str = ""

    def __call__(self, ctx) -> ResetTerm:
        from motrix_env_core.sim.model import ActuatorType

        if not self.ground_geom:
            raise ValueError("WalkStateResetCfg requires ground_geom.")
        cfg = ctx.cfg
        robot = cfg.scene.objs.robot
        base_link = robot.resolved_base_link_name
        body = ctx.model.bodies["robot"]
        joint_names = body.joint_names
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("humanoid walk requires unique actuator target joints")
        # The dof_pos query (and every consumer aligned to it) uses the
        # key-pose declaration order; require it to match the body's
        # joint-DOF order so per-joint arrays stay aligned.
        key_pose_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        if joint_names != key_pose_names:
            raise ValueError(
                "robot key_pose joint order must match the body's joint order: "
                f"key_pose={key_pose_names}, body={joint_names}"
            )
        for actuator in body.actuators:
            if actuator.actuator_type is not ActuatorType.POSITION:
                raise TypeError(f"humanoid walk actuator {actuator.name!r} must be a position actuator")

        from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

        return ResetTerm(
            reset_walk_state,
            WalkResetParams(
                default_joint_angles=body.init_joint_pos,
                init_pose=np.concatenate([body.init_base_position, body.init_base_quat]).astype(np.float32),
                heightfield=ground_height_grid(ctx, self.ground_geom),
                spawn_range=np.float32(self.spawn_xy_range),
                spawn_flatness_tol=np.float32(self.spawn_flatness_tol),
                spawn_attempts=np.int64(self.spawn_attempts),
                spawn_curriculum=bool(self.spawn_curriculum),
                ramp_min_ep_len=np.float32(self.spawn_ramp_ep_len[0]),
                ramp_max_ep_len=np.float32(self.spawn_ramp_ep_len[1]),
            ),
            self.command_name,
            writes={
                "position": BodyPositionWrite((base_link,)),
                "rotation": BodyRotationWrite((base_link,)),
                "linear_velocity": BodyLinearVelocityWrite((base_link,)),
                "angular_velocity": BodyAngularVelocityWrite((base_link,)),
                "joints_position": JointPositionWrite(joint_names),
                "joints_velocity": JointVelocityWrite(joint_names),
            },
        )

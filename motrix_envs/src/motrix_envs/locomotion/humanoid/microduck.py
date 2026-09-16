# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pollen Robotics Microduck humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg, SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import TrackingAngVelZRewardCfg, TrackingLinVelXyRewardCfg
from motrix_env_core.mdp.terminations import CollidingTerminationCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.cfg import (
    HumanoidVelocityTrackingManagerEnvCfg,
    WalkCommandsCfg,
    WalkResetCfg,
    WalkRewardsCfg,
    WalkTerminationsCfg,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import WalkStateResetCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    FeetPhaseRewardCfg,
    PenaltyActionRateRewardCfg,
    PenaltyCloseFeetXyRewardCfg,
    PoseRewardCfg,
)
from motrix_envs.robot import Microduck


def _make_microduck_robot() -> Microduck:
    return Microduck()


# Pinned Microduck termination collision inventory, owned by the Microduck walk task.
_MICRODUCK_TERMINATION_GEOMS = ("trunk_collision",)


def _make_microduck_rewards(swing_height: float = 0.04) -> WalkRewardsCfg:
    return WalkRewardsCfg(
        tracking_lin_vel=TrackingLinVelXyRewardCfg(command_name="walk", sigma=0.15, weight=10.0),
        tracking_ang_vel=TrackingAngVelZRewardCfg(command_name="walk", sigma=0.15, weight=3.0),
        penalty_action_rate=PenaltyActionRateRewardCfg(weight=-0.5),
        feet_phase=FeetPhaseRewardCfg(
            sole_l_site="left_foot",
            sole_r_site="right_foot",
            swing_height=swing_height,
            feet_phase_sigma=0.002,
            weight=8.0,
        ),
        penalty_close_feet_xy=PenaltyCloseFeetXyRewardCfg(close_feet_threshold=0.05, weight=-10.0),
        pose=PoseRewardCfg(
            pose_weights={
                "left_hip_yaw": 5.0,
                "left_hip_roll": 1.0,
                "left_hip_pitch": 0.01,
                "left_knee": 0.01,
                "left_ankle": 5.0,
                "neck_pitch": 50.0,
                "head_pitch": 50.0,
                "head_yaw": 50.0,
                "head_roll": 50.0,
                "right_hip_yaw": 5.0,
                "right_hip_roll": 1.0,
                "right_hip_pitch": 0.01,
                "right_knee": 0.01,
                "right_ankle": 5.0,
            },
            weight=-0.5,
        ),
    )


def _make_microduck_scene() -> humanoid_cfg.HumanoidWalkSceneCfg:
    # Microduck is a ~25 cm robot; frame the lead env much closer than
    # the full-size humanoid default camera (grid center at z=0.75).
    return humanoid_cfg.HumanoidWalkSceneCfg(
        system_camera=SystemCameraCfg(
            lookat=(0.0, 0.0, 0.12),
            distance=0.35,
            elevation=-20.0,
            azimuth=180.0,
        ),
        objs=StandardSceneObjsCfg(robot=_make_microduck_robot()),
    )


@registry.envcfg("microduck-walk-flat")
def make_microduck_walk_flat_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Microduck on flat ground.

    zh_CN: 控制 Microduck 小型双足机器人在平地上跟踪行走指令。
    """
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=_make_microduck_scene(),
        commands=WalkCommandsCfg(walk=WalkCommandCfg(gait_period=0.5)),
        rewards=_make_microduck_rewards(),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_MICRODUCK_TERMINATION_GEOMS,
                ground_geom="floor",
            )
        ),
        sim=SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4),
    )


@registry.envcfg("microduck-walk-rough")
def make_microduck_walk_rough_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Microduck over uneven terrain.

    zh_CN: 控制 Microduck 小型双足机器人在起伏地形上跟踪行走指令。
    """
    flat = make_microduck_walk_flat_cfg()
    return replace(
        flat,
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            system_camera=flat.scene.system_camera,
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_microduck_robot(),
            ),
        ),
        sim_reset=WalkResetCfg(humanoid_state=WalkStateResetCfg(spawn_xy_range=4.0)),
        render_spacing=0.0,
    )


registry.env("microduck-walk-flat")(ManagerEnv)
registry.env("microduck-walk-rough")(ManagerEnv)


@registry.envcfg("microduck-walk-stairs")
def make_microduck_walk_stairs_cfg(step_height: float = 0.04) -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Microduck on a stair-flight field.

    The field is a 2x2 checkerboard of four-way pyramid stairs: convex tiles
    climb to a central platform, concave tiles sink to a central pit. The step
    geometry is sized for the ~25 cm robot by default and is configurable: call
    this factory with a different ``step_height``, or override further tile
    geometry (``tread_width``, ``platform_fraction``, grid layout) via
    ``humanoid_cfg.make_stair_terrain_assets``.

    zh_CN: 控制 Microduck 小型双足机器人在阶梯地形上跟踪行走指令；台阶高度可按
    机器人尺度配置（默认 4 cm，适配约 25 cm 的 Microduck）。
    """
    flat = make_microduck_walk_flat_cfg()
    # Microduck-scale field: 16 m is plenty for a ~25 cm biped, and the smaller
    # extent doubles the height-field resolution per meter (1.25 cm cells).
    assets = humanoid_cfg.make_stair_terrain_assets(
        step_height=step_height, tread_width=0.2, field_size=16.0, resolution=480
    )
    # Stair-specific tweaks: the swing target clears the risers (3x step_height —
    # the earlier 2x still showed feet grazing the edges), velocity commands stay
    # within what 0.2 m treads allow, and the 480-resolution grid keeps hfield
    # collision cheap (3.3 cm cells under 0.2 m treads).
    rewards = _make_microduck_rewards(swing_height=3 * step_height)
    commands = replace(
        flat.commands,
        walk=replace(flat.commands.walk, vel_limit=[[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]]),
    )
    return replace(
        flat,
        rewards=rewards,
        commands=commands,
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=assets,
            system_camera=SystemCameraCfg(
                # Wide framing so the stair flights are visible.
                lookat=(0.0, 0.0, 0.15),
                distance=2.0,
                elevation=-40.0,
                azimuth=45.0,
            ),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_microduck_robot(),
            ),
        ),
        sim_reset=WalkResetCfg(humanoid_state=WalkStateResetCfg(spawn_xy_range=4.0)),
        render_spacing=0.0,
    )


registry.env("microduck-walk-stairs")(ManagerEnv)

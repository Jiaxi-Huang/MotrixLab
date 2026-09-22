# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Go1 flat-, rough-, and stairs-terrain walk configuration and environment registration."""

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, HFieldTerrainCfg, SystemCameraCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    ControlConfig,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkRoughSceneAssetsCfg,
    QuadrupedWalkStairsSceneAssetsCfg,
    RewardConfig,
    RewardScales,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask
from motrix_envs.robot import UnitreeGo1Robot


@registry.envcfg("go1-walk-flat")
@configclass
class Go1WalkDirectEnvCfg(QuadrupedWalkEnvCfg):
    """Track walking commands with Unitree Go1 on flat ground.

    zh_CN: 控制 Unitree Go1 在平地上跟踪行走指令。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    control_config: ControlConfig = ControlConfig(action_scale=0.1)
    reward_config: RewardConfig = RewardConfig(
        scales=RewardScales(
            similar_to_default=-0.03,
            swing_contact=-1.0,
        ),
        # Tighten yaw-rate tracking (default 0.25) so the gait does not yaw/curve
        # while still satisfying the body-frame velocity-tracking reward.
        tracking_ang_vel_sigma=0.05,
        base_height_target=0.275,
    )
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo1Robot(),
        ),
    )


@registry.envcfg("go1-walk-rough")
@configclass
class Go1WalkRoughDirectEnvCfg(Go1WalkDirectEnvCfg):
    """Track walking commands with Unitree Go1 on a procedural rough height field.

    zh_CN: 控制 Unitree Go1 在程序化粗糙高度场上跟踪行走指令。
    """

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        assets=QuadrupedWalkRoughSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(
                hfield="terrain",
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo1Robot(),
        ),
    )


@registry.envcfg("go1-walk-stairs")
@configclass
class Go1WalkStairsDirectEnvCfg(Go1WalkDirectEnvCfg):
    """Track walking commands with Unitree Go1 over procedural stairs surrounded by flat ground.

    zh_CN: 控制 Unitree Go1 在四周为平地的程序化金字塔台阶地形上跟踪行走指令。

    Inherits the flat task and changes only what stairs traversal needs:
    spawns pick the stairs structures' centers (platform plateaus and pit
    floors), vertical motion is not punished as harshly, and the larger action
    scale leaves enough swing clearance for the 0.08 m risers.
    """

    # The flat-walk 0.1 couples with the raw-action action_rate penalty: the
    # same joint motion costs 6.25x the penalty of the 0.25 default, which
    # caps swing clearance below the 0.08 m risers.
    control_config: ControlConfig = ControlConfig(action_scale=0.25)

    def __post_init__(self) -> None:
        self.spawn_points = self.scene.assets.spawn_points()
        self.reward_config.scales.lin_vel_z = -0.5

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        system_camera=SystemCameraCfg(distance=7.0, elevation=-25.0, azimuth=90.0),
        assets=QuadrupedWalkStairsSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(
                hfield="terrain",
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo1Robot(),
        ),
    )


registry.env("go1-walk-flat")(QuadrupedWalkTask)
registry.env("go1-walk-rough")(QuadrupedWalkTask)
registry.env("go1-walk-stairs")(QuadrupedWalkTask)

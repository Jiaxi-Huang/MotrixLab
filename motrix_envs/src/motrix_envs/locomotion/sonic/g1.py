# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import MjcfFileCfg
from motrix_env_core.manager import ManagerEnv
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.sonic import mdp
from motrix_envs.locomotion.sonic.cfg import (
    SonicActionsCfg,
    SonicCommandsCfg,
    SonicManagerEnvCfg,
    SonicObservationsCfg,
    SonicPolicyObsCfg,
    SonicValueObsCfg,
)
from motrix_envs.robot import UnitreeG129Dof
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR


class SonicManagerEnv(ManagerEnv):
    """Manager environment for the SONIC G1 motion-tracking task."""


def _make_g1_sonic_cfg() -> SonicManagerEnvCfg:
    num_future_frames = mdp.SONIC_DEFAULT_NUM_FUTURE_FRAMES
    num_history_frames = mdp.SONIC_DEFAULT_NUM_HISTORY_FRAMES
    return SonicManagerEnvCfg(
        sim=SimCfg(dt=0.005, solver_iterations=3),
        scene=StandardSceneCfg(
            objs=StandardSceneObjsCfg(
                robot=UnitreeG129Dof(model=MjcfFileCfg(file=UNITREE_G1_ASSET_DIR / "g1_sonic.xml")),
            )
        ),
        commands=SonicCommandsCfg(
            motion=mdp.SonicMotionCommandCfg(
                motion_file=os.path.join(os.environ.get("SONIC_DATA_ROOT", "data/sonic"), "sonic.npz"),
                packed_store=os.environ.get("SONIC_PACKED_STORE", "data/sonic/lafan1-pack-smoke"),
                num_future_frames=num_future_frames,
            )
        ),
        actions=SonicActionsCfg(
            joint_position=mdp.SonicJointPositionActionCfg(
                actuator_names=mdp.G1_SONIC_JOINTS,
            )
        ),
        observations=SonicObservationsCfg(
            policy=SonicPolicyObsCfg(
                obs=mdp.SonicActorObservationCfg(num_history_frames=num_history_frames),
            ),
            value=SonicValueObsCfg(
                obs=mdp.SonicCriticObservationCfg(num_history_frames=num_history_frames),
            ),
        ),
    )


@registry.envcfg("g1-sonic")
def make_g1_sonic_cfg() -> SonicManagerEnvCfg:
    """Track SONIC motion with independent future-reference and state-history windows.

    zh_CN: 使用独立的未来参考帧和状态历史窗口在 G1 上跟踪动作。
    """
    return _make_g1_sonic_cfg()


registry.env("g1-sonic")(SonicManagerEnv)

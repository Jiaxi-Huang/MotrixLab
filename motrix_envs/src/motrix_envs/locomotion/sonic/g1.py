# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

from motrix_robots.unitree import UNITREE_G1_ASSET_DIR, UnitreeG129Dof

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import MjcfFileCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.action import JointPositionActionCfg
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


class SonicManagerEnv(ManagerEnv):
    """Manager environment for the SONIC G1 motion-tracking task.

    Drives :class:`~motrix_envs.motion.rotation.CorpusRotator` when the motion
    command enables corpus rotation: the command cfg publishes its rotator
    into ``_pending_motion_rotation`` during construction (the canonical,
    kernel-bound command terms only exist once ``super().__init__`` returns),
    and each control step ticks the rotator against the canonical motion term.
    """

    def __init__(self, cfg, num_envs: int = 1, backend: str | None = None, seed: int | None = None):
        self._pending_motion_rotation = None
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        self._motion_rotation = (
            (self._command_terms["motion"], self._pending_motion_rotation)
            if self._pending_motion_rotation is not None
            else None
        )
        self._pending_motion_rotation = None

    def step(self, actions):
        result = super().step(actions)
        if self._motion_rotation is not None:
            term, rotator = self._motion_rotation
            rotator.tick(term)
            # Scalars ride the step metrics into the trainer's
            # ``metrics/*`` TensorBoard curves.
            result.metrics["motion_rotation_window"] = float(rotator.window_index)
            result.metrics["motion_rotation_coverage"] = rotator.pass_coverage
        return result


def _make_g1_sonic_cfg() -> SonicManagerEnvCfg:
    num_future_frames = mdp.SONIC_DEFAULT_NUM_FUTURE_FRAMES
    num_history_frames = mdp.SONIC_DEFAULT_NUM_HISTORY_FRAMES
    # Motion source is a NPZ corpus directory (one schema v1 clip per file
    # with the SMPL extension channels). The default is the converter's cache
    # output, so `download_bones_seed.py` + `convert_bones_seed.py` + training
    # chain up with zero extra flags; SONIC_MOTION_DIR overrides (':'-separated
    # paths join multiple roots).
    default_corpus = os.path.join(Path.home(), ".cache", "motrixlab", "bones_seed_npz", "g1")
    motion_files = tuple(filter(None, os.environ.get("SONIC_MOTION_DIR", default_corpus).split(":")))
    return SonicManagerEnvCfg(
        sim=SimCfg(dt=0.005, solver_iterations=3),
        scene=StandardSceneCfg(
            objs=StandardSceneObjsCfg(
                robot=UnitreeG129Dof(model=MjcfFileCfg(file=UNITREE_G1_ASSET_DIR / "g1_sonic.xml")),
            )
        ),
        commands=SonicCommandsCfg(
            motion=mdp.SonicMotionCommandCfg(
                motion_files=motion_files,
                num_future_frames=num_future_frames,
                # Stream the full corpus (100k+ clips) through the resident
                # window. One swap per 2x buffer_size control steps: every
                # window's data churns out of the replay buffer before the
                # next swap arrives (per-transition ~16 draws at UTD 4), and
                # the ~470 MB window decode fits its period with headroom on
                # 16 background threads.
                rotation_period=4096,
                rotation_workers=16,
            )
        ),
        actions=SonicActionsCfg(
            joint_position=JointPositionActionCfg(
                actuator_names=mdp.G1_SONIC_JOINTS,
                action_scales_by_effort_limit_over_p_gain=True,
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

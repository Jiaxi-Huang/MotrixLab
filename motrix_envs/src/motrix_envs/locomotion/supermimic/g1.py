# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 SuperMimic task: goal-conditioned LAFAN1 corpus tracking."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from motrix_robots.unitree import UnitreeG129Dof

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import FlatTerrainCfg, SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.supermimic.cfg import SuperMimicCommandsCfg, SuperMimicEnvCfg
from motrix_envs.locomotion.supermimic.mdp import SuperMimicMotionCommandCfg
from motrix_envs.locomotion.supermimic.sampling import SuperMimicStartSampler
from motrix_envs.locomotion.wbt.g1.common import G1_TRACKED_BODY_NAMES

# Corpus location: the converter cache populated by
# ``scripts/motion/download_lafan.py --all`` (one schema v1 NPZ per LAFAN1
# clip). SUPERMIMIC_MOTION_DIR overrides — ':'-separated entries join
# multiple roots.
DEFAULT_CORPUS_DIR = Path.home() / ".cache" / "motrixlab" / "lafan" / "g1_npz"


class SuperMimicManagerEnv(ManagerEnv):
    """Manager environment that ticks the SuperMimic return sampler.

    The motion command cfg publishes the host-side sampler onto the env
    during construction (the canonical kernel-bound command term only exists
    once ``super().__init__`` returns); each control step folds the step
    reward into the per-lane episode returns and republishes the start-bin
    mixture. Scalars ride the step metrics into the trainer's curves.
    """

    def __init__(self, cfg, num_envs: int = 1, backend: str | None = None, seed: int | None = None):
        self._pending_supermimic_sampler: SuperMimicStartSampler | None = None
        super().__init__(cfg, num_envs=num_envs, backend=backend, seed=seed)
        motion_term = self._command_terms["motion"]
        action_term = self._action_terms["joint_position"]
        action_term.bind_motion(motion_term)
        sampler = self._pending_supermimic_sampler
        self._supermimic_sampler = (motion_term, sampler) if sampler is not None else None
        if self._supermimic_sampler is not None:
            sampler.attach(motion_term)
        self._pending_supermimic_sampler = None

    def step(self, actions):
        result = super().step(actions)
        if self._supermimic_sampler is not None:
            term, sampler = self._supermimic_sampler
            # Clip-end episodes are successes: bootstrap them like truncations
            # (upstream treats clip end as a time-out), unlike the height/tilt
            # failures that end the return with zero remaining value.
            clip_end = np.asarray(result.metrics["motion_clip_end"], dtype=bool)
            hard_done = np.asarray(result.terminated, dtype=bool) & ~clip_end
            result.metrics.update(sampler.tick(term, result.reward, hard_done))
        return result

    def compute_transition(self, state):
        motion = self._command_terms["motion"]
        if not motion.sequential_clips:
            steps = motion.steps[:, 0]
            np.minimum(steps + 1, motion.clip.frame_clip_end[steps], out=steps)
        return super().compute_transition(state)

    def supermimic_sampling_state(self) -> dict[str, np.ndarray] | None:
        if self._supermimic_sampler is None:
            return None
        return self._supermimic_sampler[1].state_dict()

    def load_supermimic_sampling_state(self, state: dict[str, np.ndarray]) -> None:
        if self._supermimic_sampler is None:
            return
        term, sampler = self._supermimic_sampler
        sampler.load_state_dict(term, state)

    def supermimic_sampling_metrics(self) -> dict[str, float]:
        if self._supermimic_sampler is None:
            return {}
        return self._supermimic_sampler[1].metrics()


def resolve_motion_sources() -> tuple[str, ...]:
    """Resolve the LAFAN1 corpus sources for the ``g1-supermimic`` task."""
    sources = tuple(filter(None, os.environ.get("SUPERMIMIC_MOTION_DIR", str(DEFAULT_CORPUS_DIR)).split(":")))
    if not sources or not any(Path(source).exists() for source in sources):
        raise FileNotFoundError(
            "g1-supermimic needs the LAFAN1 G1 corpus (one MotrixLab motion NPZ per clip) at "
            f"{DEFAULT_CORPUS_DIR} (or SUPERMIMIC_MOTION_DIR). Build it with:\n"
            "  uv run scripts/motion/download_lafan.py --all"
        )
    return sources


def make_g1_supermimic_cfg(motion_files: tuple[str, ...]) -> SuperMimicEnvCfg:
    """Assemble the G1 SuperMimic configuration over the given corpus sources."""
    return SuperMimicEnvCfg(
        scene=StandardSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            objs=StandardSceneObjsCfg(robot=UnitreeG129Dof(), floor=FlatTerrainCfg(material="mat_ground", priority=10)),
        ),
        commands=SuperMimicCommandsCfg(
            motion=SuperMimicMotionCommandCfg(
                motion_files=motion_files,
                tracked_body_names=G1_TRACKED_BODY_NAMES,
                reference_body_name="pelvis",
                hold_at_clip_end=True,
            )
        ),
    )


@registry.envcfg("g1-supermimic")
def g1_supermimic_envcfg() -> EnvCfg:
    """Goal-conditioned tracking of the LAFAN1 corpus on Unitree G1.

    zh_CN: 在 Unitree G1 上对 LAFAN1 动作语料做 SuperMimic 目标条件跟踪。
    """

    return make_g1_supermimic_cfg(resolve_motion_sources())


registry.env("g1-supermimic")(SuperMimicManagerEnv)

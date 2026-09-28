# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 dance tracking task."""

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv

from .common import MOTION_DIR, G1WbtEnvCfg


@registry.envcfg("g1-wbt-dance")
def make_g129dof_wbt_dance_cfg() -> EnvCfg:
    """Track the G1 dance motion corpus with the manager-based environment.

    The corpus directory holds one schema v1 npz per clip: a single clip
    trains one motion, and additional clips (e.g. pulled from LAFAN1) train
    the whole corpus.

    zh_CN: 让 Unitree G1 在 dance 语料目录上做全身跟踪；目录内一个 clip
    即单动作训练，放入多个 clip 即多动作训练。
    """

    cfg = G1WbtEnvCfg()
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "dance"),)
    return cfg


registry.env("g1-wbt-dance")(ManagerEnv)

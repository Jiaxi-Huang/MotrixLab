# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG provider: world-model + BPTT over derived manager modeling views."""

from motrix_rl_builtin.apg.config import (
    ActorCfg,
    ApgCfg,
    CollectorCfg,
    CriticCfg,
    ImaginationCfg,
    ReplayCfg,
    TrainerCfg,
    WorldModelCfg,
)
from motrix_rl_builtin.apg.framework import ApgProvider

__all__ = [
    "ActorCfg",
    "ApgCfg",
    "ApgProvider",
    "CollectorCfg",
    "CriticCfg",
    "ImaginationCfg",
    "ReplayCfg",
    "TrainerCfg",
    "WorldModelCfg",
]

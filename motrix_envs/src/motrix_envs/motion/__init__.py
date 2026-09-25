# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixLab motion NPZ format (v1) — schema constants and loader.

See `wiki/design/motrixlab-motion-npz-schema.md` for the full specification.
"""

from motrix_envs.motion.library import MotionLibrary
from motrix_envs.motion.loader import MotrixMotion
from motrix_envs.motion.sampler import AdaptiveTimestepsSampler
from motrix_envs.motion.schema import (
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
)
from motrix_envs.motion.tracked import MotionChannel, WbtMotionClip

__all__ = [
    "MotrixMotion",
    "MotionLibrary",
    "WbtMotionClip",
    "MotionChannel",
    "AdaptiveTimestepsSampler",
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "SCHEMA_VERSION",
]

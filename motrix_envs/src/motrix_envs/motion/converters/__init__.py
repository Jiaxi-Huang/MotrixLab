# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Motion file converters: each module converts one source format to MotrixLab v1.

Only the public LAFAN1 converter ships with the package (the BONES-SEED
converter is directory-shaped and served by ``scripts/motion/convert_bones_seed.py``
instead). The private holosoma / xMimic converters live under
``scripts/private/`` (internal, not published).
"""

from motrix_envs.motion.converters.lafan_converter import convert_lafan

CONVERTERS = {
    "lafan": convert_lafan,
}

__all__ = ["CONVERTERS", "convert_lafan"]

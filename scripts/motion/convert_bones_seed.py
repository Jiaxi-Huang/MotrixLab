# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert the downloaded BONES-SEED raw corpus into a SONIC NPZ corpus.

Consumes the ``robot_filtered/`` (G1 CSVs) and ``smpl_filtered/`` (SMPL PKLs)
layout produced by ``scripts/motion/download_bone_seed.py`` under
``~/.cache/motrixlab/bones_seed/<robot>/`` and writes one
MotrixLab motion NPZ v1 per paired clip (with the ``ext_smpl_*`` reference
channels) into ``--output``. The output directory plugs straight into SONIC
training via ``SONIC_MOTION_DIR``.

Examples:
    # convert everything under the default download cache
    uv run scripts/motion/convert_bones_seed.py

    # smoke: only the first 20 clips onto a custom model
    uv run scripts/motion/convert_bones_seed.py \\
        --robot-root ~/.cache/motrixlab/bones_seed/robot_filtered \\
        --smpl-root ~/.cache/motrixlab/bones_seed/smpl_filtered \\
        --output data/bones_seed_npz --max-clips 20
"""

from __future__ import annotations

from pathlib import Path

from absl import app, flags

from motrix_envs.motion.converters.bones_seed_converter import convert_bones_seed_dataset

_ROBOT_ROOT = flags.DEFINE_string("robot-root", None, "Directory holding BONES-SEED G1 CSVs (default: download cache).")
_SMPL_ROOT = flags.DEFINE_string("smpl-root", None, "Directory holding BONES-SEED SMPL PKLs (default: download cache).")
_OUTPUT = flags.DEFINE_string(
    "output", None, "Destination corpus directory (default: ~/.cache/motrixlab/bones_seed_npz/<robot>)."
)
_MAX_CLIPS = flags.DEFINE_integer("max-clips", None, "Convert only the first N paired clips.")
_OVERWRITE = flags.DEFINE_bool("overwrite", False, "Re-convert clips whose NPZ already exists.")
_MODEL_FILE = flags.DEFINE_string("model-file", None, "G1 scene xml used for forward kinematics.")


_ROBOT = flags.DEFINE_string("robot", "g1", "Robot subfolder of the download cache (the release is G1-only).")


def _default_cache() -> Path:
    return Path.home() / ".cache" / "motrixlab" / "bones_seed" / _ROBOT.value


def _default_output() -> Path:
    return Path.home() / ".cache" / "motrixlab" / "bones_seed_npz" / _ROBOT.value


def main(argv: list[str]) -> None:
    del argv  # unused
    cache = _default_cache()
    robot_root = Path(_ROBOT_ROOT.value).expanduser() if _ROBOT_ROOT.value else cache / "robot_filtered"
    smpl_root = Path(_SMPL_ROOT.value).expanduser() if _SMPL_ROOT.value else cache / "smpl_filtered"
    if _MAX_CLIPS.value is not None and _MAX_CLIPS.value <= 0:
        raise SystemExit("--max-clips must be positive")
    output = Path(_OUTPUT.value).expanduser() if _OUTPUT.value else _default_output()
    result = convert_bones_seed_dataset(
        robot_root,
        smpl_root,
        output,
        overwrite=_OVERWRITE.value,
        max_clips=_MAX_CLIPS.value,
        model_file=_MODEL_FILE.value,
    )
    print(
        f"Converted {len(result['converted'])} clips "
        f"(skipped {len(result['skipped'])} existing, failed {len(result['failed'])}) "
        f"-> {result['output_dir']}"
    )


if __name__ == "__main__":
    app.run(main)

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Convert the downloaded LAFAN raw corpus into a SONIC NPZ corpus.

Consumes the ``g1/`` (retargeted G1 CSVs), ``bvh/`` (raw LAFAN1 BVHs) and
``human_joints_info.pkl`` layout produced by
``scripts/motion/download_lafan4sonic.py`` under
``~/.cache/motrixlab/lafan4sonic/`` and writes one MotrixLab motion NPZ v1 per
paired clip (with the ``ext_smpl_*`` reference channels) into ``--output``.
The output directory plugs straight into SONIC training via
``SONIC_MOTION_DIR``.

Examples:
    # convert everything under the default download cache
    uv run scripts/motion/convert_lafan4sonic.py

    # smoke: only the first 5 clips, custom raw roots
    uv run scripts/motion/convert_lafan4sonic.py \\
        --robot-root ~/.cache/motrixlab/lafan4sonic/g1 \\
        --bvh-root ~/.cache/motrixlab/lafan4sonic/bvh \\
        --output data/lafan4sonic_npz --max-clips 5

    # train on the corpus
    SONIC_MOTION_DIR=~/.cache/motrixlab/lafan4sonic_npz \\
        uv run scripts/train.py task=g1-sonic/motrix.fastsac
"""

from __future__ import annotations

from pathlib import Path

from absl import app, flags

from motrix_envs.motion.lafan4sonic.converter import convert_lafan4sonic_dataset

_ROBOT_ROOT = flags.DEFINE_string(
    "robot-root", None, "Directory holding the retargeted G1 CSVs (default: download cache)."
)
_BVH_ROOT = flags.DEFINE_string("bvh-root", None, "Directory holding the raw LAFAN1 BVHs (default: download cache).")
_HUMAN_JOINTS_INFO = flags.DEFINE_string(
    "human-joints-info", None, "gear_sonic human_joints_info.pkl (default: download cache)."
)
_OUTPUT = flags.DEFINE_string(
    "output", None, "Destination corpus directory (default: ~/.cache/motrixlab/lafan4sonic_npz)."
)
_MAX_CLIPS = flags.DEFINE_integer("max-clips", None, "Convert only the first N paired clips.")
_OVERWRITE = flags.DEFINE_bool("overwrite", False, "Re-convert clips whose NPZ already exists.")
_MODEL_FILE = flags.DEFINE_string("model-file", None, "G1 scene xml used for forward kinematics.")


def _default_cache() -> Path:
    return Path.home() / ".cache" / "motrixlab" / "lafan4sonic"


def main(argv: list[str]) -> None:
    del argv  # unused
    cache = _default_cache()
    robot_root = Path(_ROBOT_ROOT.value).expanduser() if _ROBOT_ROOT.value else cache / "g1"
    bvh_root = Path(_BVH_ROOT.value).expanduser() if _BVH_ROOT.value else cache / "bvh"
    human_info = (
        Path(_HUMAN_JOINTS_INFO.value).expanduser() if _HUMAN_JOINTS_INFO.value else cache / "human_joints_info.pkl"
    )
    if _MAX_CLIPS.value is not None and _MAX_CLIPS.value <= 0:
        raise SystemExit("--max-clips must be positive")
    output = (
        Path(_OUTPUT.value).expanduser() if _OUTPUT.value else Path.home() / ".cache" / "motrixlab" / "lafan4sonic_npz"
    )
    result = convert_lafan4sonic_dataset(
        robot_root,
        bvh_root,
        output,
        human_joints_info=human_info,
        overwrite=_OVERWRITE.value,
        max_clips=_MAX_CLIPS.value,
        model_file=_MODEL_FILE.value,
    )
    print(
        f"Converted {len(result['converted'])} clips "
        f"(skipped {len(result['skipped'])} existing, failed {len(result['failed'])}) -> {result['output_dir']}"
    )


if __name__ == "__main__":
    app.run(main)

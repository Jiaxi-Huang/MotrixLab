# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Download a G1 animation from the LAFAN1 retargeting dataset and bake it to MotrixLab v1.

Pulls a single retargeted G1 csv from the HuggingFace dataset
``lvhaidong/LAFAN1_Retargeting_Dataset`` and runs it through the ``lafan``
converter (forward-kinematics bake) to produce a MotrixLab motion NPZ that the
G1 WBT env / ``scripts/motion/replay.py`` can consume directly.

The dataset (LAFAN1) is CC BY-NC-ND 4.0: non-commercial, attribution required.

Examples:
    # list the available G1 clips
    uv run scripts/motion/download_lafan.py --list

    # download + convert one clip into the package motion dir (default 50 fps)
    uv run scripts/motion/download_lafan.py --motion dance1_subject1

    # custom output path / frame rate
    uv run scripts/motion/download_lafan.py \\
        --motion walk1_subject1 \\
        --output motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/walk1_subject1.npz \\
        --output-fps 50
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import requests
from absl import app, flags

from motrix_envs.motion.converters import convert_lafan

_HF_REPO = "lvhaidong/LAFAN1_Retargeting_Dataset"
_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{robot}/{name}.csv"
_HF_TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{robot}"

_MOTION = flags.DEFINE_string("motion", None, "Clip name to fetch, e.g. 'dance1_subject1' (with or without .csv).")
_ROBOT = flags.DEFINE_string("robot", "g1", "Robot subfolder in the dataset (converter supports 'g1').")
_OUTPUT = flags.DEFINE_string("output", None, "Destination .npz path; defaults to the package motion dir.")
_OUTPUT_FPS = flags.DEFINE_float("output-fps", 50.0, "Target frame rate of the output npz.")
_INPUT_FPS = flags.DEFINE_float("input-fps", 30.0, "Source frame rate of the dataset csv (LAFAN = 30).")
_START_SEC = flags.DEFINE_float("start-sec", 0.0, "Trim: start time in seconds (default 0 = clip start).")
_END_SEC = flags.DEFINE_float("end-sec", None, "Trim: end time in seconds (default: end of clip).")
_REVISION = flags.DEFINE_string("revision", "main", "HuggingFace dataset git revision/branch/tag.")
_CACHE_DIR = flags.DEFINE_string("cache-dir", None, "Cache dir for downloaded csvs (default ~/.cache/motrixlab/lafan).")
_FORCE = flags.DEFINE_bool("force", False, "Re-download the csv even if it is already cached.")
_KEEP_CSV = flags.DEFINE_bool("keep-csv", True, "Keep the downloaded csv in the cache dir after converting.")
_LIST = flags.DEFINE_bool("list", False, "List available clips for --robot and exit.")


def _default_motion_dir() -> Path:
    """Package motion dir the G1 WBT cfg resolves motion_files against."""
    import motrix_envs.locomotion.wbt as wbt_pkg

    return Path(wbt_pkg.__file__).resolve().parent / "assets" / "motion" / _ROBOT.value


def _cache_dir() -> Path:
    root = Path(_CACHE_DIR.value).expanduser() if _CACHE_DIR.value else Path.home() / ".cache" / "motrixlab" / "lafan"
    return root / _ROBOT.value


def _list_clips() -> list[str]:
    url = _HF_TREE_API.format(repo=_HF_REPO, rev=_REVISION.value, robot=_ROBOT.value)
    with urlopen(url) as resp:  # noqa: S310 - fixed trusted HuggingFace host
        entries = json.load(resp)
    return sorted(Path(e["path"]).stem for e in entries if e.get("path", "").endswith(".csv"))


def _download_csv(name: str, dest: Path) -> None:
    url = _HF_RESOLVE.format(repo=_HF_REPO, rev=_REVISION.value, robot=_ROBOT.value, name=name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        if resp.status_code == 404:
            raise SystemExit(
                f"Clip {name!r} not found for robot {_ROBOT.value!r} at {url}.\nRun with --list to see available clips."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        # Atomic write: stream to a temp file in the same dir, then rename.
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            written = 0
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
                written += len(chunk)
        tmp_path.replace(dest)
    size_mb = (total or dest.stat().st_size) / 1e6
    print(f"Downloaded {name}.csv ({size_mb:.1f} MB) -> {dest}")


def main(argv):
    del argv  # unused

    if _LIST.value:
        clips = _list_clips()
        print(f"{len(clips)} clips available for robot {_ROBOT.value!r}:")
        for c in clips:
            print(f"  {c}")
        return

    if _MOTION.value is None:
        raise SystemExit("--motion is required (or pass --list to browse). Example: --motion dance1_subject1")

    name = _MOTION.value[:-4] if _MOTION.value.endswith(".csv") else _MOTION.value

    csv_path = _cache_dir() / f"{name}.csv"
    if _FORCE.value or not csv_path.exists():
        _download_csv(name, csv_path)
    else:
        print(f"Using cached {csv_path} (pass --force to re-download)")

    output = Path(_OUTPUT.value).expanduser() if _OUTPUT.value else _default_motion_dir() / f"{name}.npz"
    stats = convert_lafan(
        csv_path,
        output,
        robot=_ROBOT.value,
        input_fps=_INPUT_FPS.value,
        output_fps=_OUTPUT_FPS.value,
        start_sec=_START_SEC.value,
        end_sec=_END_SEC.value,
    )
    print(
        f"Converted {name} -> {stats['output_path']}: "
        f"{stats['num_frames']} frames @ {int(round(_OUTPUT_FPS.value))} fps, "
        f"{stats['num_joints']} joints, {stats['num_bodies']} bodies"
    )

    if not _KEEP_CSV.value:
        csv_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(main)

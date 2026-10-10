# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Download the paired LAFAN1 raw sources for SONIC corpus conversion.

Pulls the three public sources the lafan4sonic converter needs:

    - retargeted G1 CSVs:  HuggingFace dataset ``lvhaidong/LAFAN1_Retargeting_Dataset``
      (``g1/`` folder) — the robot side of every clip;
    - raw LAFAN1 BVHs:     HuggingFace dataset ``johnny095212/lafan1`` (mirror of
      the Ubisoft LaForge release) — the human side, downloaded only for stems
      that also have a G1 CSV;
    - ``human_joints_info.pkl``: the gear_sonic SMPL joint metadata (rest
      offsets + parents) from the public GR00T-WholeBodyControl repository.

LAFAN1 is CC BY-NC-ND 4.0 (non-commercial, attribution required); the corpus
is for internal training use.

Output layout (consumed by ``scripts/motion/convert_lafan4sonic.py``):

    <output>/g1/<clip>.csv
    <output>/bvh/<clip>.bvh
    <output>/human_joints_info.pkl
    <output>/manifest.json

Examples:
    uv run scripts/motion/download_lafan4sonic.py
    uv run scripts/motion/download_lafan4sonic.py --output data/lafan4sonic_raw --force
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import requests
from absl import app, flags

_G1_REPO = flags.DEFINE_string(
    "g1-repo", "lvhaidong/LAFAN1_Retargeting_Dataset", "HuggingFace dataset holding the retargeted G1 CSVs."
)
_BVH_REPO = flags.DEFINE_string("bvh-repo", "johnny095212/lafan1", "HuggingFace dataset holding the raw LAFAN1 BVHs.")
_G1_REVISION = flags.DEFINE_string("g1-revision", "main", "Revision of the G1 retarget dataset.")
_BVH_REVISION = flags.DEFINE_string("bvh-revision", "main", "Revision of the BVH dataset.")
_HUMAN_INFO_URL = flags.DEFINE_string(
    "human-info-url",
    "https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/main/"
    "gear_sonic/data/human/human_joints_info.pkl",
    "gear_sonic human_joints_info.pkl download URL.",
)
_OUTPUT = flags.DEFINE_string("output", None, "Raw corpus destination (default: ~/.cache/motrixlab/lafan4sonic).")
_FORCE = flags.DEFINE_bool("force", False, "Re-download files even if they already exist.")

_TREE_API = "https://huggingface.co/api/datasets/{repo}/tree/{rev}/{folder}"
_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
_LAFAN_FPS_TOLERANCE = 1.0e-6


def _default_output() -> Path:
    return Path.home() / ".cache" / "motrixlab" / "lafan4sonic"


def _list_dataset_files(repo: str, revision: str, folder: str, suffix: str) -> dict[str, str]:
    """List ``<folder>`` members of a dataset tree -> {stem: repo-relative path}."""
    url = _TREE_API.format(repo=repo, rev=revision, folder=folder)
    with urlopen(url) as resp:  # noqa: S310 - fixed trusted HuggingFace host
        entries = json.load(resp)
    return {Path(entry["path"]).stem: entry["path"] for entry in entries if entry.get("path", "").endswith(suffix)}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        if resp.status_code == 404:
            raise SystemExit(f"Remote file not found at {url}")
        resp.raise_for_status()
        # Atomic write: stream to a temp file in the same dir, then rename.
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            for chunk in resp.iter_content(chunk_size=1 << 20):
                tmp.write(chunk)
        tmp_path.replace(dest)
    print(f"Downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) -> {dest}")


def _fetch_if_needed(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        print(f"Using cached {dest} (pass --force to re-download)")
    else:
        _download(url, dest)


def _bvh_metadata(path: Path) -> tuple[int, float]:
    frames: int | None = None
    frame_time: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("Frames:"):
                frames = int(stripped.split(":", 1)[1])
            elif stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":", 1)[1])
                break
    if frames is None or frame_time is None:
        raise ValueError(f"Missing MOTION metadata in {path}")
    return frames, frame_time


def _csv_frame_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


def main(argv: list[str]) -> None:
    del argv  # unused
    output = Path(_OUTPUT.value).expanduser() if _OUTPUT.value else _default_output()
    g1_dir = output / "g1"
    bvh_dir = output / "bvh"
    info_path = output / "human_joints_info.pkl"

    csv_members = _list_dataset_files(_G1_REPO.value, _G1_REVISION.value, "g1", ".csv")
    if not csv_members:
        raise SystemExit(f"No G1 CSVs found under '{_G1_REPO.value}/g1' at revision '{_G1_REVISION.value}'")
    bvh_members = _list_dataset_files(_BVH_REPO.value, _BVH_REVISION.value, "", ".bvh")
    if not bvh_members:
        raise SystemExit(f"No BVHs found under '{_BVH_REPO.value}' at revision '{_BVH_REVISION.value}'")

    stems = sorted(set(csv_members) & set(bvh_members))
    if not stems:
        raise SystemExit("No stems pair a G1 CSV with a BVH; check the two dataset revisions")
    missing_bvh = sorted(set(csv_members) - set(bvh_members))

    for stem in stems:
        _fetch_if_needed(
            _RESOLVE.format(repo=_G1_REPO.value, rev=_G1_REVISION.value, path=csv_members[stem]),
            g1_dir / f"{stem}.csv",
            _FORCE.value,
        )
        _fetch_if_needed(
            _RESOLVE.format(repo=_BVH_REPO.value, rev=_BVH_REVISION.value, path=bvh_members[stem]),
            bvh_dir / f"{stem}.bvh",
            _FORCE.value,
        )
    _fetch_if_needed(_HUMAN_INFO_URL.value, info_path, _FORCE.value)

    clips = []
    for stem in stems:
        bvh_frames, frame_time = _bvh_metadata(bvh_dir / f"{stem}.bvh")
        csv_frames = _csv_frame_count(g1_dir / f"{stem}.csv")
        if abs(frame_time - 1.0 / 30.0) > _LAFAN_FPS_TOLERANCE:
            raise SystemExit(f"{stem}.bvh is not 30 Hz (frame time {frame_time})")
        if bvh_frames != csv_frames:
            raise SystemExit(f"Frame-count mismatch for {stem}: BVH={bvh_frames}, G1={csv_frames}")
        clips.append({"name": stem, "source_frames": bvh_frames})

    manifest = {
        "format": "motrixlab_lafan4sonic_source_v1",
        "fps": 30,
        "g1": {"repository": _G1_REPO.value, "revision": _G1_REVISION.value},
        "bvh": {"repository": _BVH_REPO.value, "revision": _BVH_REVISION.value},
        "human_joints_info": {"url": _HUMAN_INFO_URL.value, "path": str(info_path)},
        "num_clips": len(clips),
        "num_frames": sum(clip["source_frames"] for clip in clips),
        "clips": clips,
        "g1_clips_without_bvh": missing_bvh,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Paired {len(clips)} clips ({manifest['num_frames']} source frames) -> {manifest_path}")


if __name__ == "__main__":
    app.run(main)

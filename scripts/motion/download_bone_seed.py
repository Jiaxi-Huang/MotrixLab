# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Download the paired BONES-SEED corpus (G1 CSV + SMPL PKL) for SONIC training.

The full release is two large archives on HuggingFace (~53 GB total network):

    - ``bones-studio/seed`` (dataset, license-gated): ``g1.tar.gz`` — 142k
      retargeted G1 CSV clips
    - ``nvidia/GEAR-SONIC`` (model repo, public): ``bones_seed_smpl.tar`` split
      into seven ~5 GiB parts — 131k SMPL PKL clips

Neither archive is randomly accessible, so the corpus is assembled by
streaming both archives in full by default:

    1. all SMPL parts are streamed in order and every whole PKL is extracted
       (stop early with ``--smpl-max-bytes``);
    2. ``g1.tar.gz`` is streamed next, keeping only the CSVs whose stem already
       has a PKL, until the archive ends (stop early with ``--max-pairs`` or
       ``--g1-max-bytes``);
    3. PKLs without a CSV counterpart are pruned and a provenance manifest is
       written.

Member order inside both archives is unrelated, so the pair yield of an early
stop is the intersection of the two prefixes; with no budgets set, a run
downloads both archives to completion (~53 GB network, ~130k pairs kept).

Accept the ``bones-studio/seed`` license at
https://huggingface.co/datasets/bones-studio/seed once, then ``hf auth login``.
The corpus is for internal training use; check both dataset licenses before
redistribution.

The raw corpus lands under ``~/.cache/motrixlab/bones_seed/<robot>/`` — the
same per-robot cache layout ``download_lafan.py`` uses. Output layout
(consumed by ``scripts/motion/convert_bones_seed.py``):

    <output>/robot_filtered/<clip>.csv
    <output>/smpl_filtered/<clip>.pkl
    <output>/manifest.json

Examples:
    # full release: both archives streamed to completion (default)
    uv run scripts/motion/download_bone_seed.py

    # smaller smoke subset
    uv run scripts/motion/download_bone_seed.py --max-pairs 100 --smpl-max-bytes 1GiB

Partial runs resume: clips already on disk are skipped, and a completed run
(manifest.json present) must be re-run with ``--force`` to re-download.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tarfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import requests
from absl import app, flags

_BONES_REPO = "bones-studio/seed"
_GEARSNIC_REPO = "nvidia/GEAR-SONIC"
_G1_ARCHIVE = "g1.tar.gz"
_SMPL_PARTS = tuple(f"bones_seed_smpl/bones_seed_smpl.tar.part_a{suffix}" for suffix in "abcdefg")
_FORMAT = "motrixlab_bones_seed_v1"

_HF_RESOLVE_DATASET = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_HF_RESOLVE_MODEL = "https://huggingface.co/{repo}/resolve/main/{path}"

_OUTPUT = flags.DEFINE_string("output", None, "Corpus destination (default: ~/.cache/motrixlab/bones_seed/<robot>).")
_ROBOT = flags.DEFINE_string("robot", "g1", "Robot subfolder of the cache (the release is G1-only).")
_TOKEN = flags.DEFINE_string("token", None, "HuggingFace token; defaults to $HF_TOKEN or the hf CLI token file.")
_MAX_PAIRS = flags.DEFINE_integer(
    "max-pairs", None, "Stop once this many CSV/PKL pairs are complete (default: no cap)."
)
_SMPL_MAX_BYTES = flags.DEFINE_string(
    "smpl-max-bytes", None, "Extract whole SMPL PKLs until this much sits on disk (default: the entire split tar)."
)
_G1_MAX_BYTES = flags.DEFINE_string(
    "g1-max-bytes", None, "Scan at most this much (decompressed) content of g1.tar.gz (default: the whole archive)."
)
_FORCE = flags.DEFINE_bool("force", False, "Re-extract even if the destination already has pairs.")


def _parse_bytes(text: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?)(?:i?b)?", text.strip().lower())
    units = {"": 1, "k": 1024**1, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    if match is None:
        raise SystemExit(f"Cannot parse byte budget {text!r}; use e.g. 4GiB / 512MiB / 1048576")
    return int(float(match.group(1)) * units[match.group(2)])


def _resolve_token() -> str | None:
    if _TOKEN.value:
        return _TOKEN.value
    import os

    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        return token_file.read_text(encoding="utf-8").strip()
    return None


class _HttpStream(io.RawIOBase):
    """Sequential reader over a streamed HTTP response; closing aborts the download."""

    def __init__(self, url: str, token: str | None, session: requests.Session) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._response = session.get(url, stream=True, timeout=(15, 120), headers=headers)
        if self._response.status_code in (401, 403):
            raise PermissionError(
                f"HuggingFace denied access to {url} ({self._response.status_code}). "
                f"Accept the {_BONES_REPO} license and run `hf auth login`."
            )
        self._response.raise_for_status()
        self._chunks: Iterator[bytes] = self._response.iter_content(chunk_size=1 << 20)
        self._buffer = b""
        self._position = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        view = memoryview(buffer)
        while len(self._buffer) < len(view):
            try:
                chunk = next(self._chunks)
            except StopIteration:
                break
            self._buffer += chunk
        count = min(len(view), len(self._buffer))
        view[:count] = self._buffer[:count]
        self._buffer = self._buffer[count:]
        self._position += count
        return count

    def close(self) -> None:
        self._chunks = iter(())
        self._buffer = b""
        self._response.close()
        super().close()


class _PartChain(io.RawIOBase):
    """Chain the split SMPL tar parts lazily; parts are fetched only when reached."""

    def __init__(self, urls: tuple[str, ...], token: str | None, session: requests.Session) -> None:
        self._urls = iter(urls)
        self._token = token
        self._session = session
        self._current: _HttpStream | None = None

    def readable(self) -> bool:
        return True

    def _advance(self) -> bool:
        if self._current is not None:
            self._current.close()
            self._current = None
        url = next(self._urls, None)
        if url is None:
            return False
        print(f"  streaming {Path(urlparse(url).path).name} ...")
        self._current = _HttpStream(url, self._token, self._session)
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        view = memoryview(buffer)
        written = 0
        while written < len(view):
            if self._current is None and not self._advance():
                break
            assert self._current is not None
            count = self._current.readinto(view[written:])
            if count:
                written += count
            else:
                self._current.close()
                self._current = None
        return written

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
        super().close()


def _write_member(archive: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> int:
    """Extract one member atomically; returns bytes written, 0 when it exists."""
    if dest.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f".{dest.name}.part")
    source = archive.extractfile(member)
    if source is None:
        return 0
    try:
        with partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1 << 20)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
    return member.size


def _extract_smpl_prefix(
    root: Path, token: str | None, budget: int | None, session: requests.Session
) -> dict[str, object]:
    """Stream the split SMPL tar, keeping whole PKLs until ``budget`` bytes are on disk.

    ``budget=None`` streams every part to completion.
    """
    smpl_dir = root / "smpl_filtered"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    on_disk = {path.stem: path.stat().st_size for path in smpl_dir.glob("*.pkl")}
    total = sum(on_disk.values())
    if budget is None or total < budget:
        with _PartChain(_smpl_urls(), token, session) as raw:
            with io.BufferedReader(raw, buffer_size=1 << 20) as stream:
                archive = tarfile.open(fileobj=stream, mode="r|")
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".pkl"):
                        continue
                    dest = smpl_dir / f"{Path(member.name).stem}.pkl"
                    if dest.stem in on_disk:
                        continue
                    written = _write_member(archive, member, dest)
                    total += written
                    on_disk[dest.stem] = written
                    if len(on_disk) % 250 == 0:
                        print(f"  smpl: {len(on_disk)} clips, {total / 1024**3:.2f} GiB")
                    if budget is not None and total >= budget:
                        break
    return {"stems": on_disk, "bytes": total}


def _extract_g1_matches(
    root: Path,
    token: str | None,
    smpl_stems: dict[str, int],
    max_pairs: int | None,
    budget: int | None,
    session: requests.Session,
) -> dict[str, object]:
    """Stream g1.tar.gz keeping only CSVs whose stem has a PKL, within the budgets.

    ``max_pairs=None`` and ``budget=None`` scan the whole archive.
    """
    robot_dir = root / "robot_filtered"
    robot_dir.mkdir(parents=True, exist_ok=True)
    on_disk = {path.stem for path in robot_dir.glob("*.csv")}
    scanned = 0
    matched = set(on_disk)
    with _HttpStream(_g1_url(), token, session) as raw:
        with io.BufferedReader(raw, buffer_size=1 << 20) as stream:
            archive = tarfile.open(fileobj=stream, mode="r|gz")
            for member in archive:
                scanned += max(member.size, 0)
                if budget is not None and scanned >= budget:
                    break
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                stem = Path(member.name).stem
                if stem not in smpl_stems or stem in on_disk:
                    continue
                if _write_member(archive, member, robot_dir / f"{stem}.csv"):
                    matched.add(stem)
                    on_disk.add(stem)
                    if len(matched) % 250 == 0:
                        print(f"  g1: {len(matched)} pairs, scanned {scanned / 1024**3:.2f} GiB")
                if max_pairs is not None and len(matched) >= max_pairs:
                    break
    return {"pairs": sorted(matched), "scanned": scanned}


def _prune_unpaired(root: Path, pairs: list[str]) -> int:
    keep = set(pairs)
    removed = 0
    for path in (root / "smpl_filtered").glob("*.pkl"):
        if path.stem not in keep:
            path.unlink()
            removed += 1
    return removed


def _smpl_urls() -> tuple[str, ...]:
    return tuple(_HF_RESOLVE_MODEL.format(repo=_GEARSNIC_REPO, path=part) for part in _SMPL_PARTS)


def _g1_url() -> str:
    return _HF_RESOLVE_DATASET.format(repo=_BONES_REPO, path=_G1_ARCHIVE)


def main(argv: list[str]) -> None:
    del argv  # unused
    root = (
        Path(_OUTPUT.value).expanduser()
        if _OUTPUT.value
        else Path.home() / ".cache" / "motrixlab" / "bones_seed" / _ROBOT.value
    )
    smpl_budget = _parse_bytes(_SMPL_MAX_BYTES.value) if _SMPL_MAX_BYTES.value else None
    g1_budget = _parse_bytes(_G1_MAX_BYTES.value) if _G1_MAX_BYTES.value else None
    if _MAX_PAIRS.value is not None and _MAX_PAIRS.value <= 0:
        raise SystemExit("--max-pairs must be positive")

    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not _FORCE.value:
        raise SystemExit(
            f"{root} already holds a completed corpus (manifest.json present); pass --force to re-download."
        )
    root.mkdir(parents=True, exist_ok=True)
    token = _resolve_token()

    smpl_goal = "everything" if smpl_budget is None else f"{smpl_budget / 1024**3:.1f} GiB"
    print(f"[1/3] streaming SMPL split tar for {smpl_goal} ...")
    smpl = _extract_smpl_prefix(root, token, smpl_budget, requests.Session())
    print(f"  smpl kept: {len(smpl['stems'])} clips, {smpl['bytes'] / 1024**3:.2f} GiB")

    pair_goal = len(smpl["stems"]) if _MAX_PAIRS.value is None else min(_MAX_PAIRS.value, len(smpl["stems"]))
    print(f"[2/3] streaming {_G1_ARCHIVE} for at most {pair_goal} pairs ...")
    g1 = _extract_g1_matches(root, token, smpl["stems"], _MAX_PAIRS.value, g1_budget, requests.Session())
    print(f"  pairs: {len(g1['pairs'])}, scanned {g1['scanned'] / 1024**3:.2f} GiB of the archive")

    removed = _prune_unpaired(root, g1["pairs"])
    pairs_dir_bytes = sum(
        path.stat().st_size
        for name in ("robot_filtered", "smpl_filtered")
        for path in (root / name).glob("*")
        if path.is_file()
    )
    manifest = {
        "format": _FORMAT,
        "sources": {
            "g1": {"repo": _BONES_REPO, "file": _G1_ARCHIVE},
            "smpl": {"repo": _GEARSNIC_REPO, "files": list(_SMPL_PARTS)},
        },
        "num_pairs": len(g1["pairs"]),
        "pairs": g1["pairs"],
        "bytes_on_disk": pairs_dir_bytes,
        "smpl_prefix_bytes": smpl["bytes"],
        "g1_scanned_bytes": g1["scanned"],
        "licenses": "bones-studio/seed (gated, accept on HuggingFace) + nvidia/GEAR-SONIC; internal training use",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[3/3] kept {len(g1['pairs'])} pairs ({pairs_dir_bytes / 1024**3:.2f} GiB on disk, "
        f"pruned {removed} unmatched PKLs) -> {root}\n"
        f"next: python scripts/motion/convert_bones_seed.py && "
        f"python scripts/train.py task=g1-sonic/motrix.fastsac"
    )


if __name__ == "__main__":
    app.run(main)

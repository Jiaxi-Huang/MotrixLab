# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every SuperMimic clip in its own CPU simulator process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

from eval_supermimic import _motion_paths, _resolve_checkpoint
from motrix_robots.unitree import UNITREE_G1_ASSET_DIR

from motrix_env_core import registry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def _evaluate_one(path: Path, checkpoint: Path, output_dir: Path, timeout: int) -> dict:
    output = output_dir / f"{path.stem}.json"
    log = output_dir / f"{path.stem}.log"
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_supermimic.py")),
        "--ckpt",
        str(checkpoint),
        "--device",
        "cpu",
        "--motion-name",
        path.stem,
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
    except subprocess.TimeoutExpired as exc:
        log.write_text(f"Timed out after {timeout}s\n{exc}", encoding="utf-8")
        return {"status": "engine_error", "log": str(log), "reason": "timeout"}
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return {"status": "engine_error", "log": str(log), "returncode": completed.returncode}
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
        clip = next(iter(result["per_clip"].values()))
        if clip["clip"] != path.stem:
            raise ValueError("clip name mismatch")
        return {
            "status": "ok",
            "norm_return": float(clip["norm_return"]),
            "steps": int(clip["steps"]),
            "length": int(clip["length"]),
            "complete": not clip["failed"] and clip["steps"] == clip["length"] - 1,
            "result": str(output),
        }
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        return {"status": "engine_error", "log": str(log), "reason": str(exc)}


def _summary(entries: dict[str, dict], names: list[str], identity: dict) -> dict:
    valid = [entries[name] for name in names if entries.get(name, {}).get("status") == "ok"]
    full = len(valid) == len(names)
    return {
        "identity": identity,
        "n_clips": len(names),
        "n_evaluated": len(valid),
        "n_engine_errors": sum(entries.get(name, {}).get("status") == "engine_error" for name in names),
        "n_complete": sum(entry["complete"] for entry in valid),
        "norm_return": sum(entry["norm_return"] for entry in valid) / len(valid) if full else None,
        "norm_return_framew": (
            sum(entry["norm_return"] * (entry["length"] - 1) for entry in valid)
            / sum(entry["length"] - 1 for entry in valid)
            if full
            else None
        ),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900, help="Seconds per clip.")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.timeout < 1:
        parser.error("workers and timeout must be positive")
    checkpoint = _resolve_checkpoint(args.ckpt).resolve()
    paths = _motion_paths(registry.resolve("g1-supermimic").env_cfg.commands.motion.motion_files)
    names = [path.stem for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("Corpus clip names must be unique")
    source = Path(__file__).resolve().parents[1]
    identity = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "robot_sha256": _sha256(UNITREE_G1_ASSET_DIR / "g1_29dof.xml"),
        "motion_sha256": {path.stem: _sha256(path) for path in paths},
        "source_sha256": {
            str(path.relative_to(source)): _sha256(path)
            for path in (
                source / "scripts/eval_supermimic.py",
                source / "scripts/eval_supermimic_suite.py",
                source / "motrix_rl_builtin/src/motrix_rl_builtin/apg/evaluate.py",
                source / "motrix_envs/src/motrix_envs/locomotion/kinematics.py",
                source / "motrix_env_core/src/motrix_env_core/torch/geometry.py",
                source / "motrix_envs/src/motrix_envs/locomotion/supermimic/apg_head.py",
                source / "motrix_envs/src/motrix_envs/locomotion/supermimic/evaluate.py",
                source / "motrix_envs/src/motrix_envs/locomotion/supermimic/mdp.py",
            )
        },
        "versions": {name: version(name) for name in ("motrixsim", "torch", "motrixlab")},
        "protocol": "one CPU simulator process per clip; first frame; no evaluation noise or randomization",
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "identity.json", identity)
    entries: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_evaluate_one, path, checkpoint, output_dir, args.timeout): path.stem for path in paths}
        for future in as_completed(futures):
            name = futures[future]
            entries[name] = future.result()
            _write_json(output_dir / "suite.json", _summary(entries, names, identity))
            print(f"[{len(entries)}/{len(names)}] {name}: {entries[name]['status']}", flush=True)
    print(json.dumps(_summary(entries, names, identity), indent=2))


if __name__ == "__main__":
    main()

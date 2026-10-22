# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-clip evaluation of a trained g1-supermimic checkpoint.

Deterministic full-clip evaluation over the LAFAN1 corpus: one environment
lane per clip, every episode pinned to its clip's first frame, no
observation / exploration / reset noise, running until each lane finishes
its clip or fails a tracking check. Reports the mean and frame-weighted
normalized returns, the failed-clip list, and the per-clip breakdown, and
appends a row to ``eval.csv`` in the checkpoint directory.

Examples:
    # evaluate one checkpoint (json summary on stdout + <ckpt>_eval.json)
    python scripts/eval_supermimic.py --ckpt runs/g1-supermimic/<run>/checkpoints/latest.pt

    # resolve the latest policy of a run directory and evaluate it
    python scripts/eval_supermimic.py --run_dir runs/g1-supermimic/<run>

    # custom corpus / env and device
    SUPERMIMIC_MOTION_DIR=... python scripts/eval_supermimic.py --ckpt ... --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import torch
from motrix_rl_builtin.supermimic.evaluate import clip_bounds, evaluate, load_policy
from motrix_rl_builtin.supermimic.mdp import build_supermimic_mdp

from motrix_env_core import registry as env_registry

EVAL_CSV_FIELDS = (
    "iteration",
    "saved_at",
    "eval_at",
    "norm_return",
    "norm_return_framew",
    "n_failed",
    "n_clips",
)


def _count_corpus_clips(motion_files) -> int:
    """Count the corpus clips the MotionLibrary would assemble, in order."""
    count = 0
    for raw in motion_files:
        path = Path(raw).expanduser()
        if path.is_dir():
            count += sum(1 for child in sorted(path.iterdir()) if child.suffix == ".npz")
        else:
            count += 1
    if count == 0:
        raise FileNotFoundError(
            "No motion clips found for evaluation; build the corpus with "
            "scripts/motion/download_lafan.py --all or set SUPERMIMIC_MOTION_DIR."
        )
    return count


def _resolve_checkpoint(checkpoint: Path) -> Path:
    """Accept a checkpoint file or a run directory (manifest / latest fallback)."""
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Neither checkpoint file nor run directory: {checkpoint}")
    manifest = checkpoint / "checkpoints" / "manifest.json"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as stream:
            artifacts = json.load(stream).get("artifacts", {})
        for key in ("best_policy", "latest_training_state"):
            entry = artifacts.get(key)
            if entry:
                candidate = checkpoint / "checkpoints" / entry["path"]
                if candidate.is_file():
                    return candidate
    fallback = checkpoint / "checkpoints" / "latest.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No checkpoint found under {checkpoint} (manifest or checkpoints/latest.pt).")


def _clip_names(env, mdp) -> list[str]:
    """Per-clip labels from the configured motion sources, in corpus order."""
    sources = env.cfg.commands.motion.motion_files
    names: list[str] = []
    for raw in sources:
        path = Path(raw).expanduser()
        if path.is_dir():
            names.extend(child.stem for child in sorted(path.iterdir()) if child.suffix == ".npz")
        else:
            names.append(path.stem)
    starts, _ = clip_bounds(mdp.clip_end)
    if len(names) != int(starts.numel()):
        raise ValueError(
            f"Corpus source names ({len(names)}) do not match the assembled clips ({int(starts.numel())})."
        )
    return names


def _atomic_write_json(path: Path, result: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(result, stream, indent=1)
        stream.write("\n")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="g1-supermimic", help="Registered SuperMimic environment id.")
    parser.add_argument("--run_dir", type=Path, help="Run directory; evaluates its recorded policy checkpoint.")
    parser.add_argument("--ckpt", type=Path, help="Checkpoint file (or run directory) to evaluate.")
    parser.add_argument("--device", default=None, help="Torch device; default cuda when available else cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    args = parser.parse_args(argv)
    if args.ckpt is None and args.run_dir is None:
        parser.error("one of --run_dir or --ckpt is required")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint(args.ckpt if args.ckpt is not None else args.run_dir)

    spec = env_registry.resolve(args.env)
    train_cfg = spec.env_cfg
    for_eval = getattr(train_cfg, "for_eval", None)
    if for_eval is None:
        raise TypeError(f"Environment {args.env!r} does not provide a deterministic for_eval() variant.")
    eval_cfg = for_eval()
    num_clips = _count_corpus_clips(train_cfg.commands.motion.motion_files)
    env = env_registry.resolve(args.env, env_cfg=eval_cfg).make(num_envs=num_clips, mode="train", seed=args.seed)
    try:
        mdp = build_supermimic_mdp(env).to(device)
        names = _clip_names(env, mdp)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy = load_policy(mdp, checkpoint, device)
        result = evaluate(
            env,
            policy,
            mdp,
            goal_frames=int(checkpoint["args"]["goal_frames"]),
            hist=int(checkpoint["args"]["hist"]),
            device=device,
            clip_names=names,
        )
        del result["failed_clip_indices"]
    finally:
        renderer = getattr(env, "renderer", None)
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()

    result["checkpoint"] = str(checkpoint_path)
    result["iteration"] = int(checkpoint.get("iteration", -1))
    summary = {key: value for key, value in result.items() if key != "per_clip"}
    print(json.dumps(summary, indent=1))

    output = checkpoint_path.with_name(f"{checkpoint_path.stem}_eval.json")
    _atomic_write_json(output, result)
    csv_path = checkpoint_path.parent / "eval.csv"
    new_file = not csv_path.is_file()
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        if new_file:
            writer.writerow(EVAL_CSV_FIELDS)
        writer.writerow(
            (
                result["iteration"],
                checkpoint.get("saved_at", ""),
                result["evaluated_at"],
                f"{result['norm_return']:.4f}",
                f"{result['norm_return_framew']:.4f}",
                result["n_failed"],
                len(result["per_clip"]),
            )
        )
    print(f"[eval] wrote {output} and appended {csv_path}")


if __name__ == "__main__":
    main()

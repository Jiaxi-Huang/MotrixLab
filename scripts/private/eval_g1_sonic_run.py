# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Headless deterministic evaluation of a g1-sonic fastsac run checkpoint.

Loads a run's ``latest.pt`` through the standard play machinery (same env spec
and agent construction as ``motrix_rl.fastsac`` play) and reports episode
return / length statistics plus reward-term and env-metric summaries without
opening a renderer.

Two regimes:
- ``train``: training conditions (mixed encoder sampling, adaptive sampling,
  10 s episode cap) — directly comparable to ``rollout/mean_return`` in TB.
- ``play``: ``for_play()`` conditions (encoder sampling pinned to g1, adaptive
  sampling off, no episode cap) — pure g1-reference tracking.

``--encoders g1 smpl teleop`` adds per-encoder evaluations: play conditions
with the encoder sampling pinned to each named stream. The g1 stream carries
dense future joint references, so a healthy run tracks g1 best; comparing the
three regimes is the sanity check for a corpus (e.g. LAFAN via lafan4sonic).
Pinning happens at env construction (``for_play()`` already pins g1; the
pinned builds start from the same play cfg with ``encoder_sampling``
overwritten), because the compiled reset kernel captures the sampling mode at
build time and ignores later Python-side mutation.

Usage:
    python scripts/private/eval_g1_sonic_run.py runs/g1-sonic/... --steps 2000
    python scripts/private/eval_g1_sonic_run.py runs/g1-sonic/... --mode play \
        --encoders g1 smpl teleop
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_rl import runner, runs


def _summarize(values: list[float], name: str) -> None:
    v = np.asarray(values)
    print(
        f"  {name:18s} n={len(v):5d}  mean {v.mean():+8.3f}  median {np.median(v):+8.3f}  "
        f"p5 {np.percentile(v, 5):+8.3f}  p95 {np.percentile(v, 95):+8.3f}  min {v.min():+8.3f}  max {v.max():+8.3f}"
    )


def _make_pinned_env(trainer, encoder: str, num_envs: int):
    """Build a play-conditions env with the encoder sampling pinned to ``encoder``.

    ``for_play()`` pins g1; the pin lives in the compiled reset kernel, so the
    override must land in the env config before construction. The pinned cfg is
    a play cfg (``for_play()`` applied) with ``encoder_sampling`` overwritten,
    built with ``mode="train"`` so ``EnvBuildSpec.make`` does not re-apply
    ``for_play()`` and clobber the pin.
    """
    spec = trainer._env_spec  # noqa: SLF001
    pinned_cfg = copy.deepcopy(spec.env_cfg).for_play()
    pinned_cfg.commands.motion.encoder_sampling = encoder
    original = trainer._env_spec  # noqa: SLF001
    trainer._env_spec = replace(spec, env_cfg=pinned_cfg)  # noqa: SLF001
    try:
        return trainer._make_env(num_envs, render=None, mode="train")  # noqa: SLF001
    finally:
        trainer._env_spec = original  # noqa: SLF001


def evaluate(trainer, policy: str, num_envs: int, steps: int, mode: str, encoder: str | None = None) -> None:
    if encoder is not None:
        env = _make_pinned_env(trainer, encoder, num_envs)
        mode = f"play/{encoder}"
    else:
        env = trainer._make_env(num_envs, render=None, mode=mode)  # noqa: SLF001
    agent = trainer._make_agent(env)  # noqa: SLF001
    ckpt = torch.load(policy, map_location=agent.device, weights_only=False)
    agent.load_state_dict(ckpt)
    agent.actor.eval()

    obs, _ = env.reset()
    ep_ret = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.int64)
    returns: list[float] = []
    lengths: list[int] = []
    step_reward_means = np.zeros(steps, dtype=np.float64)
    terminations = 0
    truncations = 0
    reward_sums: dict[str, float] = {}
    reward_counts = 0
    metric_sums: dict[str, float] = {}
    metric_counts = 0

    for step in range(steps):
        actions = agent.act(obs, deterministic=True)
        obs, _, reward, terminated, truncated = env.step(actions)
        r = reward.cpu().numpy()
        t = terminated.cpu().numpy()
        c = truncated.cpu().numpy()
        ep_ret += r
        ep_len += 1
        step_reward_means[step] = r.mean()
        terminations += int(t.sum())
        truncations += int(c.sum())
        done = t | c
        if done.any():
            returns.extend(ep_ret[done].tolist())
            lengths.extend(ep_len[done].tolist())
            ep_ret[done] = 0.0
            ep_len[done] = 0
        terms = env.last_info["Reward"]
        flat = {k: float(np.sum(v)) for k, v in terms.items()} if isinstance(terms, dict) else {}
        for k, v in flat.items():
            reward_sums[k] = reward_sums.get(k, 0.0) + v
        reward_counts += num_envs
        for k, v in env.last_info["metrics"].items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(np.mean(v))
        metric_counts += 1
        if step % 250 == 0:
            print(f"  [mode={mode}] step {step:5d}/{steps}  reward/step {r.mean():+.4f}  episodes {len(returns)}")

    print(f"\n== mode={mode} results ({steps} steps x {num_envs} envs, deterministic policy) ==")
    if lengths:
        _summarize(returns, "episode_return")
        _summarize(lengths, "episode_length")
        cap = 500 if mode == "train" else None
        if cap:
            full = sum(1 for l in lengths if l >= cap)
            print(f"  episodes reaching {cap}-step cap: {full}/{len(lengths)} ({full / len(lengths):.1%})")
    print(f"  terminations {terminations}  truncations {truncations}")
    print(
        f"  reward/step mean {step_reward_means.mean():+.5f}  p5 {np.percentile(step_reward_means, 5):+.5f}  "
        f"p95 {np.percentile(step_reward_means, 95):+.5f}"
    )
    print("  reward terms (mean per env-step):")
    for k in sorted(reward_sums):
        print(f"    {k:50s} {reward_sums[k] / reward_counts:+.5f}")
    print("  env metrics (mean over steps):")
    for k in sorted(metric_sums):
        print(f"    {k:50s} {metric_sums[k] / metric_counts:+.5f}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--policy", default=None, help="checkpoint path (default: <run>/checkpoints/latest.pt)")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--mode", default="both", choices=["train", "play", "both"])
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["g1", "teleop", "smpl"],
        default=None,
        help="Additionally evaluate play conditions with the encoder sampling pinned to each stream.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = args.policy or str(args.run_dir / "checkpoints" / "latest.pt")
    context = runs.open_run_context(args.run_dir)
    handle = runner.create_run_handle(context, play_num_envs=args.num_envs, seed=args.seed, render=None)
    trainer = handle.trainer
    trainer._set_seed(args.seed)  # noqa: SLF001

    modes = ["train", "play"] if args.mode == "both" else [args.mode]
    for mode in modes:
        evaluate(trainer, policy, args.num_envs, args.steps, mode)
    for encoder in args.encoders or ():
        evaluate(trainer, policy, args.num_envs, args.steps, "play", encoder=encoder)


if __name__ == "__main__":
    main()

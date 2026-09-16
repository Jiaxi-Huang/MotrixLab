# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the FastSAC collector actor-only inference boundary.

The measured interval is exactly ``Collector._infer``: CPU policy observation
staging, optional H2D, read-only normalization, stochastic actor inference,
and optional D2H plus synchronization. Environment stepping, transition-ring
push, weight synchronization and learner work are intentionally excluded.

With ``--num-collectors > 1`` the benchmark spawns one process per collector
(each bound to its NUMA node / CPU slice, mirroring the multi-collector async
trainer topology) and reports per-collector results plus the aggregate
transitions-per-second, so CUDA-vs-CPU collector inference can be compared at
the intended process count and core budget.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

from motrix_rl.fastsac.async_impl.collector import Collector
from motrix_rl.fastsac.async_impl.numa import bind_process
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing, WeightSnapshot
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.networks import Actor


class _BenchmarkEnv:
    def __init__(self, num_envs: int, obs_dim: int, critic_obs_dim: int):
        self.num_envs = num_envs
        self._obs = torch.randn(num_envs, obs_dim)
        self._critic_obs = torch.zeros(num_envs, critic_obs_dim)
        self.last_info = {}

    def reset(self):
        return self._obs, self._critic_obs


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[index]


def _source_policy(args, num_envs: int):
    action_scale = torch.ones(args.act_dim)
    action_bias = torch.zeros(args.act_dim)
    actor = Actor(
        n_obs=args.obs_dim,
        n_act=args.act_dim,
        hidden_dim=args.hidden_dim,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    with torch.no_grad():
        actor.fc_mu.weight.normal_(0.0, 0.02)
        actor.fc_mu.bias.normal_(0.0, 0.02)
        actor.fc_logstd.weight.normal_(0.0, 0.02)
        actor.fc_logstd.bias.normal_(0.0, 0.02)
    normalizer = EmpiricalNormalization(args.obs_dim, device="cpu")
    normalizer._mean.normal_(0.0, 0.1)
    normalizer._std.uniform_(0.8, 1.2)
    normalizer._var.copy_(normalizer._std.square())
    normalizer.count.fill_(1_000_000)
    return actor, normalizer, action_scale, action_bias


def _collector(args, num_envs: int, source_actor, source_normalizer, action_scale, action_bias):
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            actor_hidden_dim=args.hidden_dim,
            log_std_max=0.0,
            log_std_min=-5.0,
            use_tanh=True,
            use_layer_norm=True,
            obs_normalization=True,
            learning_starts=0,
        ),
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(
                collector_inference_device=args.device,
                collector_compile=args.compile,
                collector_amp=args.amp,
                collector_amp_dtype=args.amp_dtype,
                weight_poll_interval=1,
            )
        ),
    )
    env = _BenchmarkEnv(num_envs, args.obs_dim, args.critic_obs_dim)
    ring = SharedTransitionRing(1, num_envs, args.obs_dim, args.critic_obs_dim, args.act_dim)
    weights = WeightSnapshot(sum(p.numel() for p in source_actor.parameters()), args.obs_dim)
    weights.publish(source_actor, source_normalizer)
    collector = Collector(
        env,
        cfg,
        args.obs_dim,
        args.critic_obs_dim,
        args.act_dim,
        action_scale,
        action_bias,
        ring,
        weights,
        Control(),
    )
    collector.reset()
    collector.sync_weights()
    collector.warmup_inference()
    return collector, weights


def measure_single(args, num_envs: int, collector_id: int = 0, numa_node: int | None = None) -> dict:
    """Run the single-collector inference benchmark in the current process."""
    torch.manual_seed(7 + collector_id)
    torch.set_num_threads(args.cpu_threads)
    bind_process(
        f"bench-collector[{collector_id}]",
        numa_node,
        cpus_per_collector=args.cpus_per_collector,
        collector_id=collector_id,
        num_collectors=args.num_collectors,
    )
    source_actor, source_normalizer, action_scale, action_bias = _source_policy(args, num_envs)
    collector, weights = _collector(args, num_envs, source_actor, source_normalizer, action_scale, action_bias)
    obs = torch.randn(num_envs, args.obs_dim)

    for _ in range(args.warmup):
        collector._infer(obs)

    samples_ms = []
    first = None
    last = None
    for index in range(args.samples):
        start = time.perf_counter()
        actions = collector._infer(obs)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
        if index == 0:
            first = actions.clone()
        if index == args.samples - 1:
            last = actions.clone()

    with torch.no_grad():
        normalized = source_normalizer(obs, update=False)
        cpu_deterministic = source_actor.explore(normalized, deterministic=True)
        with collector._autocast():
            device_deterministic = collector._policy.deterministic(obs.to(collector.device)).cpu()
    abs_error = (device_deterministic - cpu_deterministic).abs()
    sync_samples_ms = []
    for _ in range(args.sync_samples):
        weights.publish(source_actor, source_normalizer)
        start = time.perf_counter()
        collector.sync_weights()
        sync_samples_ms.append((time.perf_counter() - start) * 1000.0)
    staging_ptrs = [
        buffer.data_ptr()
        for buffer in (collector._obs_host, collector._obs_device, collector._actions_host)
        if buffer is not None
    ]
    return {
        "collector_id": collector_id,
        "num_envs": num_envs,
        "torch_version": torch.__version__,
        "device": str(collector.device),
        "device_name": torch.cuda.get_device_name(collector.device) if collector.device.type == "cuda" else None,
        "compile": args.compile,
        "amp": args.amp,
        "amp_dtype": args.amp_dtype if args.amp else None,
        "obs_dim": args.obs_dim,
        "critic_obs_dim": args.critic_obs_dim,
        "act_dim": args.act_dim,
        "hidden_dim": args.hidden_dim,
        "cpu_threads": torch.get_num_threads(),
        "numa_node": numa_node,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "warmup": args.warmup,
        "samples": args.samples,
        "latency_ms": {
            "median": statistics.median(samples_ms),
            "p90": _percentile(samples_ms, 0.90),
            "mean": statistics.fmean(samples_ms),
            "min": min(samples_ms),
            "max": max(samples_ms),
        },
        "weight_sync_ms": {
            "median": statistics.median(sync_samples_ms),
            "p90": _percentile(sync_samples_ms, 0.90),
            "mean": statistics.fmean(sync_samples_ms),
            "min": min(sync_samples_ms),
            "max": max(sync_samples_ms),
        },
        "deterministic_vs_cpu": {
            "max_abs_error": float(abs_error.max()),
            "mean_abs_error": float(abs_error.mean()),
        },
        "stochastic_samples_differ": not torch.equal(first, last),
        "actions_finite": bool(torch.isfinite(last).all()),
        "actions_in_bounds": bool(torch.all(last >= -1.0) and torch.all(last <= 1.0)),
        "staging_ptrs": staging_ptrs,
        # transitions/s this collector sustains on the measured interval alone
        "env_steps_per_s": num_envs * 1000.0 / statistics.median(samples_ms),
    }


def _run_bench_child(args, num_envs: int, collector_id: int, numa_node: int | None, result_queue) -> None:
    try:
        result_queue.put(measure_single(args, num_envs, collector_id, numa_node))
    except BaseException as exc:  # noqa: BLE001 - propagate failures to the parent
        result_queue.put({"collector_id": collector_id, "error": repr(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--num-envs", type=int, default=4096, help="total envs, split evenly across collectors")
    parser.add_argument("--obs-dim", type=int, default=154)
    parser.add_argument("--critic-obs-dim", type=int, default=154)
    parser.add_argument("--act-dim", type=int, default=29)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sync-samples", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=12, help="torch threads per collector")
    parser.add_argument("--num-collectors", type=int, default=1)
    parser.add_argument("--numa-nodes", type=int, nargs="*", help="one NUMA node id per collector")
    parser.add_argument("--cpus-per-collector", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    num_collectors = args.num_collectors
    if num_collectors < 1 or args.num_envs % num_collectors != 0:
        parser.error("--num-envs must divide evenly across --num-collectors (>= 1)")
    if args.numa_nodes and len(args.numa_nodes) != num_collectors:
        parser.error("--numa-nodes must provide one node id per collector")
    shard_envs = args.num_envs // num_collectors

    if num_collectors == 1:
        # single-collector path: identical in-process measurement as before
        result = measure_single(args, args.num_envs)
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_run_bench_child,
                args=(args, shard_envs, i, args.numa_nodes[i] if args.numa_nodes else None, result_queue),
                name=f"bench-collector-{i}",
            )
            for i in range(num_collectors)
        ]
        for p in processes:
            p.start()
        results = [result_queue.get() for _ in processes]
        for p in processes:
            p.join(timeout=60)
        errors = [r for r in results if "error" in r]
        if errors:
            raise RuntimeError(f"bench collector processes failed: {errors}")
        results.sort(key=lambda r: r["collector_id"])
        result = {
            "num_collectors": num_collectors,
            "numa_nodes": args.numa_nodes,
            "cpus_per_collector": args.cpus_per_collector,
            "total_num_envs": args.num_envs,
            "aggregate_env_steps_per_s": sum(r["env_steps_per_s"] for r in results),
            "collectors": results,
        }

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

# Copyright Motphys Technology Co., Ltd. 2025, 2026

"""Multi-collector topology unit tests: control aggregation, env sharding,
strict-UTD accounting across sharded slots, stats aggregation and NUMA
helper selection. Process-level behavior is covered by end-to-end training."""

import math
from types import SimpleNamespace

import pytest
import torch

from motrix_rl.fastsac.async_impl.learner import Learner
from motrix_rl.fastsac.async_impl.numa import parse_cpulist, select_collector_cpus, validate_numa_options
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing
from motrix_rl.fastsac.async_impl.worker import aggregate_collector_stats, split_num_envs


def test_control_aggregates_per_collector_steps() -> None:
    control = Control(num_collectors=3)
    control.inc_collector_steps()
    control.inc_collector_steps(1)
    control.inc_collector_steps(1)
    control.inc_collector_steps(2)

    assert control.collector_steps == 4
    assert control.collector_steps_at(0) == 1
    assert control.collector_steps_at(1) == 2
    assert control.collector_steps_at(2) == 1


def test_control_resume_restarts_each_collector_from_checkpointed_iteration() -> None:
    control = Control(num_collectors=2)
    control.collector_steps = 5

    # v is in full-batch equivalents: each collector's own counter restarts at v
    assert (control.collector_steps_at(0), control.collector_steps_at(1)) == (5, 5)
    assert control.collector_steps == 10


def test_single_collector_control_matches_previous_single_counter() -> None:
    control = Control()
    control.collector_steps = 7
    control.inc_collector_steps()

    assert control.collector_steps == 8


def test_split_num_envs_requires_even_division() -> None:
    assert split_num_envs(2048, 2) == [1024, 1024]
    assert split_num_envs(2048, 1) == [2048]
    with pytest.raises(ValueError, match="divide evenly"):
        split_num_envs(2048, 3)
    with pytest.raises(ValueError, match="num_collectors must be >= 1"):
        split_num_envs(2048, 0)


def _strict_learner(num_updates: int) -> Learner:
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(cfg=SimpleNamespace(num_updates=num_updates))
    learner.async_options = SimpleNamespace(utd_mode="strict")
    learner.control = SimpleNamespace(collector_steps=0)
    return learner


def test_strict_utd_counts_merged_full_batches() -> None:
    learner = _strict_learner(num_updates=4)

    assert [learner._num_updates_for(n) for n in (3, 0, 2)] == [12, 0, 8]


def _push_batch(ring, tag: float) -> None:
    n = ring.num_envs
    obs = torch.full((n, ring.obs.shape[-1]), tag)
    actions = torch.zeros(n, ring.actions.shape[-1])
    rewards = torch.zeros(n)
    dones = torch.zeros(n, dtype=torch.long)
    pushed = ring.push(obs, obs, actions, rewards, dones, dones, obs, obs)
    assert pushed


def _multi_learner(rings, extends):
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(
        device=torch.device("cpu"),
        rb=SimpleNamespace(extend_batch=lambda *columns: extends.append([c.clone() for c in columns])),
        cfg=SimpleNamespace(learning_starts=0),
    )
    learner.async_options = SimpleNamespace(max_ingest_per_iter=8)
    learner.control = Control(num_collectors=len(rings))
    learner.rings = rings
    learner.weights = []
    learner._pending = {}
    learner._next_gen = 0
    return learner


def test_drain_merges_shard_generations_into_full_batches() -> None:
    rings = [SharedTransitionRing(4, 4, 3, 3, 2) for _ in range(2)]
    extends: list = []
    learner = _multi_learner(rings, extends)

    # only collector 0 produced generation 0: nothing is ingestible yet
    _push_batch(rings[0], 1.0)
    assert learner.drain() == 0
    assert extends == []

    # collector 1 catches up: generation 0 merges into one full 8-env batch
    _push_batch(rings[1], 2.0)
    assert learner.drain() == 1
    assert len(extends) == 1
    merged_obs = extends[0][0]
    # batched columns are stacked along the time axis: (num_envs, count, dim)
    assert merged_obs.shape == (8, 1, 3)
    # env blocks concatenate in collector order
    assert merged_obs[:4, 0].eq(1.0).all()
    assert merged_obs[4:, 0].eq(2.0).all()
    # both read cursors advanced past the consumed generation
    assert all(ring.read_idx == 1 for ring in rings)


def test_drain_holds_generation_until_all_rings_deliver() -> None:
    rings = [SharedTransitionRing(4, 4, 3, 3, 2) for _ in range(2)]
    extends: list = []
    learner = _multi_learner(rings, extends)

    _push_batch(rings[0], 1.0)
    _push_batch(rings[0], 1.5)
    _push_batch(rings[1], 2.0)
    assert learner.drain() == 1
    # generation 1 waits for collector 1's second slot
    assert learner._next_gen == 1
    _push_batch(rings[1], 2.5)
    assert learner.drain() == 1
    assert all(ring.read_idx == 2 for ring in rings)


def test_single_ring_drain_batches_multiple_slots() -> None:
    ring = SharedTransitionRing(4, 4, 3, 3, 2)
    extends: list = []
    learner = _multi_learner([ring], extends)

    for tag in (1.0, 2.0, 3.0):
        _push_batch(ring, tag)
    assert learner.drain() == 3
    # one extend_batch with all three batches stacked on the time axis
    assert len(extends) == 1
    assert extends[0][0].shape == (4, 3, 3)
    assert extends[0][0][:, 0].eq(1.0).all()
    assert extends[0][0][:, 2].eq(3.0).all()
    assert ring.read_idx == 3


def _snapshot(collector_id: int, ret: float, episodes: int, lag: int, collect_ms: float) -> dict:
    return {
        "collector_id": collector_id,
        "return": ret,
        "ep_len": 100.0,
        "episodes": episodes,
        "reward_terms": {"alive": 0.5},
        "env_metrics": {},
        "policy_lag": lag,
        "timing_ms": {"collect": collect_ms},
    }


def test_aggregate_collector_stats_merges_snapshots() -> None:
    per_collector = {0: _snapshot(0, 10.0, 4, 1, 8.0), 1: _snapshot(1, 20.0, 6, 3, 12.0)}
    stats = aggregate_collector_stats(per_collector)

    assert stats["return"] == pytest.approx(15.0)
    assert stats["episodes"] == 10
    assert stats["policy_lag"] == 3
    assert stats["timing_ms"]["collect"] == pytest.approx(10.0)


def test_aggregate_collector_stats_skips_missing_collectors() -> None:
    stats = aggregate_collector_stats({0: _snapshot(0, 10.0, 4, 1, 8.0), 1: None})

    assert stats["return"] == pytest.approx(10.0)
    assert stats["episodes"] == 4


def test_aggregate_collector_stats_defaults_without_snapshots() -> None:
    stats = aggregate_collector_stats({0: None, 1: None})

    assert math.isnan(stats["return"])
    assert stats["episodes"] == 0
    assert stats["policy_lag"] == 0


def test_extend_batch_matches_repeated_extend_including_wraparound() -> None:
    """extend_batch writes exactly what successive extend calls would, including
    a run that wraps the circular buffer end — n-step time adjacency depends on it."""
    from motrix_rl.fastsac.buffer import SimpleReplayBuffer

    def make():
        return SimpleReplayBuffer(n_env=4, buffer_size=5, n_obs=3, n_act=2, n_critic_obs=3, device="cpu")

    def batch(tag: float, count: int):
        obs = torch.full((4, count, 3), tag)
        actions = torch.zeros(4, count, 2)
        rewards = torch.arange(count, dtype=torch.float).expand(4, count).clone()
        flags = torch.zeros(4, count, dtype=torch.long)
        return obs, obs.clone(), actions, rewards, flags, flags.clone(), obs.clone(), obs.clone()

    stepwise, batched = make(), make()
    tags = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # runs past buffer_size -> wraps
    i = 0
    while i < len(tags):
        count = 2 if i + 1 < len(tags) else 1
        b = batch(tags[i], count)
        # per-column tags so the batched group matches the stepwise column stream
        obs = torch.stack([torch.full((4, 3), tags[i + j]) for j in range(count)], dim=1)
        rewards = torch.tensor([float(i + j) for j in range(count)]).expand(4, count).clone()
        b = (obs, obs.clone(), b[2], rewards, b[4], b[5], obs.clone(), obs.clone())
        batched.extend_batch(*b)
        for j in range(count):
            single = batch(tags[i + j], 1)
            single = (
                single[0],
                single[1],
                single[2],
                torch.full((4, 1), float(i + j)),
                single[4],
                single[5],
                single[6],
                single[7],
            )
            stepwise.extend(*(f[:, 0] for f in single))
        i += count

    for name in ("observations", "actions", "rewards", "dones", "next_observations"):
        torch.testing.assert_close(getattr(stepwise, name), getattr(batched, name))
    assert batched.ptr == stepwise.ptr
    assert batched.num_stored == 5


def test_parse_cpulist() -> None:
    assert parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpulist(" 5 ") == [5]
    assert parse_cpulist("") == []


def test_select_collector_cpus_chunks_and_shares() -> None:
    base = [0, 1, 2, 3, 4, 5]

    # without cpus_per_collector every collector shares the full set
    assert select_collector_cpus(base, 0, 2, None) == base
    assert select_collector_cpus(base, 1, 2, None) == base

    assert select_collector_cpus(base, 0, 2, 3) == [0, 1, 2]
    assert select_collector_cpus(base, 1, 2, 3) == [3, 4, 5]
    # over-subscription is a config error, not a silent empty binding
    with pytest.raises(ValueError, match="no CPUs"):
        select_collector_cpus(base, 2, 3, 4)

    with pytest.raises(ValueError, match="positive"):
        select_collector_cpus(base, 0, 1, 0)
    with pytest.raises(ValueError, match="empty"):
        select_collector_cpus([], 0, 1, None)


def test_validate_numa_options_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="numa_nodes must have exactly"):
        validate_numa_options(2, [0], None)

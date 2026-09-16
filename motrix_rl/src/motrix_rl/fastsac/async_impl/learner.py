# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Learner: owns a full ``FastSacAgent`` and drives training off the shared ring.

Unlike the sync trainer it does NOT step the env. It drains raw transitions from
:class:`~motrix_rl.fastsac.async_impl.shm.SharedTransitionRing` into the agent's GPU
replay buffer, runs gradient updates governed by ``utd_mode`` (§6 of the
design), and periodically publishes actor weights + obs-normalizer stats to the
collector via :class:`~motrix_rl.fastsac.async_impl.shm.WeightSnapshot`.

The update math is reused unchanged from the sync agent: this module delegates
the per-step gradient work to ``agent.update(n)`` and only owns the
async-specific orchestration (drain, UTD-ratio governance, weight publishing).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import torch

from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.shm import (
    Control,
    SharedTransitionRing,
    TransitionFields,
    WeightSnapshot,
)
from motrix_rl.fastsac.config import FastSacCfg


class Learner:
    """Drains N independent SPSC rings (one per collector) and broadcasts weights.

    Each collector owns its own ring and its own :class:`WeightSnapshot`; the
    single-writer invariants of both primitives are preserved unchanged. The
    learner round-robins the rings (per-ring ingest bound so a fast collector's
    full ring never starves the others) and publishes the current actor snapshot
    to every collector in turn — each publish is independent and lock-free, so
    one slow reader never blocks the others.

    Ring slots carry ``num_envs / num_collectors`` transitions each. The replay
    buffer (and its n-step time adjacency per env) is built around full
    ``num_envs`` batches, so shards are merged by generation: the k-th slot of
    every collector assembles into the k-th full batch (env ids map to
    contiguous shard blocks). A slow collector's generation stays pending on
    the CPU slot views until complete — its own ring fills up and backpressures
    only that collector; read cursors are committed only after the merged batch
    reached the GPU, preserving the ring's no-clobber guarantee.
    """

    def __init__(
        self,
        agent: FastSacAgent,
        cfg: FastSacCfg,
        rings: list[SharedTransitionRing],
        weights: list[WeightSnapshot],
        control: Control,
    ):
        self.agent = agent
        self.cfg = cfg
        self.async_options = cfg.trainer.async_options
        self.rings = rings
        self.weights = weights
        self.control = control
        self._learning_starts = agent.cfg.learning_starts
        self._last_publish_ms = 0.0
        # generation assembly: generation index -> one shard slot per ring,
        # still None while that ring's shard for the generation is in flight.
        self._pending: dict[int, list[TransitionFields | None]] = {}
        self._next_gen = 0

        # keep normalizers/actor in train mode: the learner is the update side.
        self.agent.set_train_mode()

    # The persistent gradient-update counter lives on the agent so both sync
    # and async trainers share the same policy-frequency gating source-of-truth.
    # Exposed here as a read-only property for the worker's UTD / publish panel.
    @property
    def update_idx(self) -> int:
        return self.agent.update_idx

    # ------------------------------------------------------------------ ingest
    def drain(self) -> int:
        """Move up to ``max_ingest_per_iter`` slots per ring into the replay buffer.

        Returns the number of full ``num_envs`` batches ingested. Slots are
        assembled into generations (one shard per collector); complete
        generations are merged, moved to the GPU and written with one
        :meth:`extend_batch` per field per flush. With a single ring every
        consumed slot is a complete generation, so the same path degenerates
        to plain batched ingest with no merge barrier. Read cursors advance
        only after the GPU copy, so a collector cannot clobber an in-flight
        slot; a full or slow ring only blocks itself.
        """
        device = self.agent.device
        num_rings = len(self.rings)
        for ring_id, ring in enumerate(self.rings):
            base = ring.read_idx
            for offset in range(max(self.async_options.max_ingest_per_iter, 1)):
                slot = ring.read_slot(offset)
                if slot is None:
                    break
                parts = self._pending.setdefault(base + offset, [None] * num_rings)
                parts[ring_id] = slot
        return self._flush_complete_generations(device)

    def _flush_complete_generations(self, device) -> int:
        generations: list[list[TransitionFields]] = []
        while True:
            parts = self._pending.get(self._next_gen)
            if parts is None or any(part is None for part in parts):
                break
            generations.append(parts)
            del self._pending[self._next_gen]
            self._next_gen += 1
        if not generations:
            return 0
        # env blocks concatenate in collector order -> env id mapping is
        # stable across batches, so per-env trajectories stay contiguous
        # in the buffer's time dimension. A single shard skips the cat.
        merged: Iterator[TransitionFields] = (
            parts[0] if len(parts) == 1 else tuple(torch.cat(field) for field in zip(*parts)) for parts in generations
        )
        self._ingest(merged)
        for _ in generations:
            for ring in self.rings:
                ring.commit_read()
        return len(generations)

    def _ingest(self, batches: Iterable[TransitionFields]) -> None:
        """Write one or more full ``num_envs`` batches with a single buffer
        write per field.

        Batches move to the GPU first, then stack along a new time axis
        (``(num_envs, count, dim)``), so :meth:`extend_batch` issues eight
        contiguous GPU-side column writes instead of eight per batch — both
        the blocking H2D count and the GPU kernel count stay flat as the
        ingest batch grows.
        """
        device = self.agent.device
        on_device = (tuple(field.to(device) for field in batch) for batch in batches)
        self.agent.rb.extend_batch(*(torch.stack(column, dim=1) for column in zip(*on_device)))

    # ------------------------------------------------------------------ update
    def _ready(self) -> bool:
        # Each collector warms up for learning_starts of its own (sharded)
        # batches, so the aggregate threshold scales with the collector count;
        # in full-batch equivalents this keeps `learning_starts` iterations of
        # num_envs transitions before the first update, like the sync trainer.
        return (
            self.control.collector_steps >= self._learning_starts * self.control.num_collectors
            and self.agent.rb.num_stored > 0
        )

    def _num_updates_for(self, ingested: int) -> int:
        """Decide how many gradient updates to run this iteration.

        ``ingested`` counts full ``num_envs`` batches (shards are merged before
        ingestion), so the strict ratio needs no per-collector rescaling.
        """
        base = self.agent.cfg.num_updates
        mode = self.async_options.utd_mode
        if mode == "strict":
            # exactly num_updates per ingested env-step batch -> matches sync UTD.
            return ingested * base
        # learner_bound: run a full batch of updates whenever ready.
        # In the two-process path the learner loops continuously; here per-call.
        return base

    def maybe_train(self, ingested: int) -> dict | None:
        """Run ratio-governed updates. Returns last metrics dict or ``None``."""
        if not self._ready():
            return None
        n = self._num_updates_for(ingested)
        # Delegate the per-step work to the agent; this module no longer keeps
        # its own update-loop / update_idx / _last_actor — the agent's
        # counter is the single source of truth for policy-frequency gating.
        metrics = self.agent.update(n)
        self._last_publish_ms = 0.0
        if metrics is not None:
            started = time.perf_counter()
            self.publish_if_due()
            self._last_publish_ms = (time.perf_counter() - started) * 1000.0
        return metrics

    # ------------------------------------------------------------------ publish
    def publish_weights(self) -> None:
        """Broadcast the current actor snapshot to every collector's snapshot."""
        for weights in self.weights:
            weights.publish(self.agent.actor, self.agent.obs_normalizer)

    def publish_if_due(self) -> None:
        if self.agent.update_idx % max(self.async_options.weight_publish_interval, 1) == 0:
            self.publish_weights()

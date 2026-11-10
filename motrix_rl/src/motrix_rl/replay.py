# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Episode-boundary-correct sequence replay for world-model trainers.

Device-resident per-lane ring over one
:class:`~motrix_rl.transport.schema.TransitionSchema`: ``extend`` stores one
full-batch step (fields addressed by schema name), sampling draws indices and
gathers entirely on device. Windows never cross episode boundaries: the
stored episode and step ids — derived here from the done flags — keep samples
inside one episode and break sequences at ring-wrap points where older slots
of the same lane were overwritten. Successor states are explicit schema
columns (``next_*``), so a slot is self-contained and the done rows carry the
assembler-written pre-reset terminal. ``extend`` copies every leaf tensor, so
later steps, resets, and buffer reuse cannot mutate stored samples. See
[RL 序列数据平面设计](../../../wiki/design/rl-sequence-data-plane.md).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from motrix_rl.transport.schema import TransitionSchema

# Schema fields the replay treats as the fixed transition core; everything
# else (state fields and any provider extras) rides the per-name stores and
# the "states" output group.
_ACTIONS = "actions"
_REWARDS = "rewards"
_TERMINATED = "terminated"
_TRUNCATED = "truncated"
_FIXED = frozenset({_ACTIONS, _REWARDS, _TERMINATED, _TRUNCATED})
_NEXT_PREFIX = "next_"


class SequenceReplay:
    """Device-resident ring of schema-named transitions with episode continuity.

    Storage is one ``(num_envs, capacity, *per_env_shape)`` tensor per schema
    field plus internal ``episode_id``/``step_id`` int64 rows; extends are
    full-batch, so every lane shares one ring cursor. CUDA ingests go through
    per-field pinned staging under one event (the staging is not reused until
    the previous transfer completed).
    """

    def __init__(
        self,
        *,
        schema: TransitionSchema,
        num_envs: int,
        capacity: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ):
        self.schema = schema
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = torch.device(device)
        self._stores: dict[str, torch.Tensor] = {
            field.name: torch.zeros((num_envs, capacity, *field.per_env_shape), dtype=field.dtype, device=self.device)
            for field in schema.fields
        }
        self.episode_id = torch.zeros((num_envs, capacity), dtype=torch.int64, device=self.device)
        self.step_id = torch.zeros((num_envs, capacity), dtype=torch.int64, device=self.device)
        # Full-batch extends keep every lane on one cursor and one written
        # count; the device mirror feeds the weighted sampler without host
        # round-trips, the host scalar backs the empty/short-replay guards.
        self._cursor = 0
        self._written = 0
        self._written_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        # Running per-lane counters write the next slot's ids without reading
        # stored state back: done rows step the episode counter and zero the
        # step counter for the following slot.
        self._episode_counter = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._step_counter = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        pinned = self.device.type == "cuda"
        self._staging = {
            field.name: torch.zeros((num_envs, *field.per_env_shape), dtype=field.dtype, pin_memory=pinned)
            for field in schema.fields
        }
        self._staging_dev = {name: t.to(self.device) for name, t in self._staging.items()}
        self._staging_event = torch.cuda.Event() if pinned else None
        self._arange = torch.arange(num_envs, device=self.device)

    # -- ingest -----------------------------------------------------------------

    def _stored(self) -> int:
        """Transitions currently stored per lane (capped by the ring)."""
        return min(self._written, self.capacity)

    def __len__(self) -> int:
        return self.num_envs * self._stored()

    def extend(self, batch: Mapping[str, torch.Tensor]) -> None:
        """Store one full-batch step; every leaf tensor is copied.

        Args:
            batch: Per-field tensors of shape ``(num_envs,) + per_env_shape``,
                keyed exactly by the schema's field names.
        """
        missing = sorted(set(self._stores) - set(batch))
        unknown = sorted(set(batch) - set(self._stores))
        if missing or unknown:
            raise ValueError(
                f"extend batch fields {sorted(batch)} do not match the schema {sorted(self._stores)}; "
                f"missing: {missing}, unknown: {unknown}."
            )
        for name, store in self._stores.items():
            value = batch[name]
            expected = (self.num_envs, *store.shape[2:])
            if tuple(value.shape) != expected:
                raise ValueError(f"extend field {name!r} shape {tuple(value.shape)} does not match {expected}.")
        if self._staging_event is not None:
            self._staging_event.synchronize()
            for name in self._stores:
                self._staging[name].copy_(batch[name])
                self._staging_dev[name].copy_(self._staging[name], non_blocking=True)
            source = self._staging_dev
        else:
            # CPU: the scatter itself copies; staging would only double it.
            source = dict(batch)
        position = torch.tensor(self._cursor % self.capacity, device=self.device)
        rows = self._arange
        for name, store in self._stores.items():
            store[rows, position] = source[name]
        self.episode_id[rows, position] = self._episode_counter
        self.step_id[rows, position] = self._step_counter
        done = (source[_TERMINATED].to(torch.bool)) | (source[_TRUNCATED].to(torch.bool))
        self._episode_counter += done.to(torch.int64)
        self._step_counter = torch.where(done, torch.zeros_like(self._step_counter), self._step_counter + 1)
        if self._staging_event is not None:
            self._staging_event.record()
        self._cursor += 1
        self._written += 1
        self._written_dev += 1

    # -- checkpoint -------------------------------------------------------------

    def rng_state(self) -> torch.Tensor:
        """Checkpoint the sampling stream so resumed runs continue it."""
        return self._generator.get_state()

    def load_rng_state(self, state: torch.Tensor) -> None:
        self._generator.set_state(state)

    # -- sampling internals -------------------------------------------------------

    def _stored_dev(self) -> torch.Tensor:
        """``_stored`` on the sampling device, for the weighted draws."""
        return torch.clamp(self._written_dev, max=self.capacity)

    def _oldest(self) -> int:
        """Ring index of the oldest stored slot (lanes share one cursor)."""
        return (self._cursor - self._stored()) % self.capacity

    def _sample_slots(self, batch_size: int, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample ``(env, position)`` slots among valid starts.

        ``available`` is the per-lane count of valid start positions; the
        weighted draw runs entirely on device (float64 cumulative intervals,
        ``searchsorted``) so batch sampling never crosses to the host.
        """
        if self._stored() == 0:
            raise ValueError("Cannot sample from an empty replay")
        cumulative = torch.cumsum(available.to(torch.float64), dim=0)
        draws = torch.rand(batch_size, device=self.device, generator=self._generator, dtype=torch.float64)
        envs = torch.searchsorted(cumulative, draws * cumulative[-1]).clamp(max=self.num_envs - 1)
        phases = torch.rand(batch_size, device=self.device, generator=self._generator, dtype=torch.float64)
        ages = (phases * available[envs]).to(torch.int64)
        starts = (self._oldest() + ages) % self.capacity
        return envs, starts

    def _history_intact(self, envs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Vectorized previous-slot link check for sampled positions.

        Step 0 of an episode is always intact. Later steps are intact when the
        chronologically previous ring slot still holds the same episode's
        ``step_id - 1``; otherwise the ring overwrote the history.
        """
        steps = self.step_id[envs, positions]
        prev_positions = (positions - 1) % self.capacity
        linked = steps > 0
        linked &= (self.episode_id[envs, prev_positions] == self.episode_id[envs, positions]) & (
            self.step_id[envs, prev_positions] == steps - 1
        )
        return linked | (steps == 0)

    def _previous_actions(self, envs: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Previous actions for sampled positions and whether they are trustworthy.

        Step 0 of an episode carries zero previous action. Later steps carry
        the stored action of the chronologically previous slot when the link
        is intact; broken rows carry zeros and are flagged instead of being
        silently patched. The where-gather keeps this a fixed-shape device
        computation (no data-dependent sync).
        """
        steps = self.step_id[envs, positions]
        prev_positions = (positions - 1) % self.capacity
        linked = self._history_intact(envs, positions) & (steps > 0)
        gathered = self._stores[_ACTIONS][envs, prev_positions]
        previous = torch.where(linked.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        return previous, linked | (steps == 0)

    # -- sampling API -------------------------------------------------------------

    def sample_windows(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        """Sample windows of ``length`` transitions with a supervision mask.

        A window may cross an episode boundary: world-model supervision is
        only valid while the stored slots chain without a reset, so the
        returned ``valid`` mask keeps the steps up to and including the
        window's first done flag (that step's ``next_*`` column is the
        assembler-written pre-reset terminal) and masks the post-reset tail.
        Ring-wrap discontinuities mask the same way.

        Returns per window: the ``length`` rows of every state field
        (``states``/``next_states`` name-keyed dicts), the fixed
        actions/rewards/terminated/truncated columns, the first step's
        previous action, and the ``(batch, length)`` valid mask.
        """
        if length < 1:
            raise ValueError(f"Window length must be at least 1, got {length}")
        if self._stored() < length:
            raise ValueError(
                f"Replay has too few transitions for the requested window: {self._stored()} stored, {length} needed"
            )
        # Valid starts per lane: every stored position whose window of
        # ``length`` still fits before the ring head.
        available = torch.clamp(self._stored_dev() - (length - 1), min=0).expand(self.num_envs)
        envs, starts = self._sample_slots(batch_size, available)
        offsets = torch.arange(length, device=self.device)
        candidate_positions = (starts[:, None] + offsets[None, :]) % self.capacity  # (B, L)
        # A step chains while it stays in the start's episode with consecutive
        # step ids; the first chain break (reset or ring wrap) ends the
        # supervisable prefix, cummin turns that into the running AND.
        episodes = self.episode_id[envs[:, None], candidate_positions] == self.episode_id[envs, starts][:, None]
        expected = self.step_id[envs, starts][:, None] + offsets[None, :]
        consecutive = self.step_id[envs[:, None], candidate_positions] == expected
        chained = episodes & consecutive
        valid = torch.cummin(chained.to(torch.int8), dim=1).values.to(torch.bool)
        rows = envs[:, None]
        states: dict[str, torch.Tensor] = {}
        next_states: dict[str, torch.Tensor] = {}
        for name in self._stores:
            if name in _FIXED or name.startswith(_NEXT_PREFIX):
                continue
            states[name] = self._stores[name][rows, candidate_positions]
            successor = self._stores.get(f"{_NEXT_PREFIX}{name}")
            if successor is not None:
                next_states[name] = successor[rows, candidate_positions]
        previous, _ = self._previous_actions(envs, starts)
        return {
            "states": states,
            "next_states": next_states,
            "actions": self._stores[_ACTIONS][rows, candidate_positions],
            "rewards": self._stores[_REWARDS][rows, candidate_positions],
            "terminated": self._stores[_TERMINATED][rows, candidate_positions].to(torch.bool),
            "truncated": self._stores[_TRUNCATED][rows, candidate_positions].to(torch.bool),
            "previous_action": previous,
            "valid": valid,
        }

    def sample_starts(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample imagination start candidates with previous actions.

        Rows whose previous-action link is broken carry zeros there and are
        flagged by the returned ``intact`` mask; the caller folds it into its
        acceptance count so one device sync covers the whole sampling round.
        The slot indices ride along for follow-up :meth:`history` calls.
        """
        available = self._stored_dev().expand(self.num_envs)
        envs, positions = self._sample_slots(batch_size, available)
        previous, intact = self._previous_actions(envs, positions)
        states = {
            name: store[envs, positions]
            for name, store in self._stores.items()
            if name not in _FIXED and not name.startswith(_NEXT_PREFIX)
        }
        return {
            "states": states,
            "previous_action": previous,
            "intact": intact,
            "envs": envs,
            "positions": positions,
        }

    def history(self, envs: torch.Tensor, positions: torch.Tensor, field: str, length: int) -> torch.Tensor:
        """``(B, length, *shape)`` stack of one field, oldest first.

        The stack is the slot's own value followed by the preceding slots of
        the same episode; before the episode start the earliest value is
        repeated, and gaps in the stored history (ring wrap, unrecorded
        spans) repeat the earliest contiguous value the same way.
        """
        store = self._stores[field]
        steps = self.step_id[envs, positions]
        offsets = torch.arange(length, device=self.device)
        # Before the episode start, repeat its first value; never reach past
        # the oldest stored slot of the lane.
        back = torch.minimum(offsets[None, :], steps[:, None])
        age = (positions - self._oldest()) % self.capacity
        back = torch.minimum(back, age[:, None])
        past = (positions[:, None] - back) % self.capacity
        # Stop at the first gap, even if older slots happen to match again.
        contiguous = (self.episode_id[envs[:, None], past] == self.episode_id[envs, positions, None]) & (
            self.step_id[envs[:, None], past] == steps[:, None] - back
        )
        available = contiguous.long().cumprod(dim=1).sum(dim=1) - 1
        back = torch.minimum(back, available[:, None])
        past = (positions[:, None] - back) % self.capacity
        stacked = store[envs[:, None], past]  # newest first
        return torch.flip(stacked, dims=[1])


__all__ = ["SequenceReplay"]

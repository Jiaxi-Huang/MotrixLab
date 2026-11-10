# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WM transition schema template and sequence replay contracts (P3.1).

The fixtures drive the replay on CPU with a two-field modeling state; the
episode-boundary, ring-wrap, and history semantics are the contracts the
world-model learner will rely on (see wiki/design/rl-sequence-data-plane.md).
"""

import pytest
import torch

from motrix_rl.replay import SequenceReplay
from motrix_rl.transport.schema import TransitionField, TransitionSchema, wm_transition_schema

# ---------------------------------------------------------------------------
# WM transition schema template
# ---------------------------------------------------------------------------


def test_wm_schema_template_layout() -> None:
    schema = wm_transition_schema(
        (TransitionField("dof_pos", (23,), torch.float32), TransitionField("phase", (), torch.float32)),
        4,
    )
    assert schema.field_names == (
        "dof_pos",
        "phase",
        "next_dof_pos",
        "next_phase",
        "actions",
        "rewards",
        "terminated",
        "truncated",
    )
    by_name = {field.name: field for field in schema.fields}
    assert by_name["next_dof_pos"].per_env_shape == (23,)
    assert by_name["next_dof_pos"].dtype == torch.float32
    assert by_name["phase"].per_env_shape == ()
    assert by_name["actions"].per_env_shape == (4,)
    assert by_name["rewards"].dtype == torch.float32
    assert by_name["terminated"].dtype == torch.int64
    assert by_name["truncated"].dtype == torch.int64


def test_wm_schema_template_optional_obs_and_provider_extras() -> None:
    schema = wm_transition_schema((TransitionField("dof_pos", (23,), torch.float32),), 4, obs_dim=30)
    assert "obs" in schema.field_names
    # Provider extras concatenate onto the template's core.
    extended = TransitionSchema(
        fields=schema.fields + (TransitionField("goal", (3,), torch.float32),),
    )
    assert "goal" in extended.field_names


def test_wm_schema_template_rejects_next_prefixed_states() -> None:
    with pytest.raises(ValueError, match="next_"):
        wm_transition_schema((TransitionField("next_state", (3,), torch.float32),), 2)


# ---------------------------------------------------------------------------
# Fixtures: schema-driven replay with a two-field modeling state
# ---------------------------------------------------------------------------

_NUM_ENVS = 3
_STATE_FIELDS = (TransitionField("dof_pos", (2,), torch.float32), TransitionField("phase", (), torch.float32))
_ACT_DIM = 1
_SCHEMA = wm_transition_schema(_STATE_FIELDS, _ACT_DIM)


def _make_replay(capacity: int = 16, seed: int = 0) -> SequenceReplay:
    return SequenceReplay(schema=_SCHEMA, num_envs=_NUM_ENVS, capacity=capacity, device="cpu", seed=seed)


def _step_batch(
    step: int,
    *,
    terminated: torch.Tensor | None = None,
    truncated: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    values = {
        "dof_pos": torch.full((_NUM_ENVS, 2), float(step)),
        "phase": torch.full((_NUM_ENVS,), float(step)),
        "next_dof_pos": torch.full((_NUM_ENVS, 2), float(step) + 0.5),
        "next_phase": torch.full((_NUM_ENVS,), float(step) + 0.5),
        "actions": torch.full((_NUM_ENVS, _ACT_DIM), float(step) + 0.25),
        "rewards": torch.full((_NUM_ENVS,), float(step) * 0.1),
        "terminated": torch.zeros(_NUM_ENVS, dtype=torch.int64),
        "truncated": torch.zeros(_NUM_ENVS, dtype=torch.int64),
    }
    if terminated is not None:
        values["terminated"] = terminated.to(torch.int64)
    if truncated is not None:
        values["truncated"] = truncated.to(torch.int64)
    return values


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def test_extend_copies_leaves_and_counts_stored() -> None:
    replay = _make_replay()
    batch = _step_batch(0)
    replay.extend(batch)
    assert len(replay) == _NUM_ENVS
    # Later mutation of the caller's tensors must not reach stored samples.
    batch["dof_pos"].fill_(-99.0)
    batch["rewards"].fill_(-99.0)
    assert torch.equal(replay._stores["dof_pos"][:, 0], torch.full((_NUM_ENVS, 2), 0.0))


def test_extend_validates_field_set_and_shapes() -> None:
    replay = _make_replay()
    batch = _step_batch(0)
    del batch["phase"]
    with pytest.raises(ValueError, match="do not match the schema"):
        replay.extend(batch)
    bad = _step_batch(0)
    bad["dof_pos"] = torch.zeros(_NUM_ENVS, 5)
    with pytest.raises(ValueError, match="shape"):
        replay.extend(bad)


def test_capacity_wrap_keeps_latest_slots() -> None:
    capacity = 4
    steps_written = capacity + 3
    replay = _make_replay(capacity=capacity)
    for step in range(steps_written):
        replay.extend(_step_batch(step))
    assert len(replay) == _NUM_ENVS * capacity
    # cursor = 7: the newest slot (step 6) sits at (cursor-1) % capacity.
    newest = (steps_written - 1) % capacity
    for name in ("dof_pos", "phase"):
        assert torch.equal(
            replay._stores[name][:, newest],
            torch.full_like(replay._stores[name][:, newest], float(steps_written - 1)),
        )


# ---------------------------------------------------------------------------
# Episode / step id derivation (T-C)
# ---------------------------------------------------------------------------


def test_done_flags_derive_episode_and_step_ids() -> None:
    replay = _make_replay()
    replay.extend(_step_batch(0))
    replay.extend(_step_batch(1, terminated=torch.tensor([True, False, False])))
    replay.extend(_step_batch(2, truncated=torch.tensor([False, True, False])))
    replay.extend(_step_batch(3))

    assert replay.episode_id[:, 0].tolist() == [0, 0, 0]
    assert replay.step_id[:, 0].tolist() == [0, 0, 0]
    assert replay.episode_id[:, 1].tolist() == [0, 0, 0]
    assert replay.step_id[:, 1].tolist() == [1, 1, 1]
    # Lane 0 restarted (terminated at step 1); lane 1 restarted (truncated at
    # step 2); lane 2 continues one episode throughout.
    assert replay.episode_id[:, 2].tolist() == [1, 0, 0]
    assert replay.step_id[:, 2].tolist() == [0, 2, 2]
    assert replay.episode_id[:, 3].tolist() == [1, 1, 0]
    assert replay.step_id[:, 3].tolist() == [1, 0, 3]


# ---------------------------------------------------------------------------
# Windows: boundary-inclusive validity, ring wrap, terminal successors
# ---------------------------------------------------------------------------


def test_sample_windows_masks_steps_past_the_first_reset() -> None:
    replay = _make_replay()
    for step in range(6):
        done = torch.ones(_NUM_ENVS, dtype=torch.bool) if step == 2 else None
        batch = _step_batch(step, terminated=done)
        # The done rows' successor column is the assembler-written terminal.
        if step == 2:
            batch["next_dof_pos"] = torch.full((_NUM_ENVS, 2), 99.0)
        replay.extend(batch)

    windows = replay.sample_windows(batch_size=64, length=4)
    valid = windows["valid"]
    assert valid.shape == (64, 4)
    steps = windows["states"]["phase"]  # (B, L) holds the step counter
    # Windows starting at step 0 cover the reset at step 2: the done step
    # itself stays valid (inclusive), the post-reset tail does not.
    reset_rows = steps[:, 0] == 0.0
    assert bool(reset_rows.any())
    row = int(torch.nonzero(reset_rows)[0])
    assert valid[row].tolist() == [True, True, True, False]
    # The done step's next-state column is the terminal, not step 3's state.
    assert torch.equal(windows["next_states"]["dof_pos"][row, 2], torch.full((2,), 99.0))
    # A window starting exactly at the reset keeps the done step valid and
    # masks everything after it.
    at_reset = steps[:, 0] == 2.0
    assert bool(at_reset.any())
    assert bool(valid[at_reset][:, 0].all())
    assert not bool(valid[at_reset][:, 1:].any())


def test_sample_windows_stay_inside_episodes() -> None:
    replay = _make_replay()
    for step in range(8):
        done = torch.tensor([True, False, False]) if step % 3 == 2 else None
        replay.extend(_step_batch(step, terminated=done))

    windows = replay.sample_windows(batch_size=64, length=4)
    steps = windows["states"]["phase"]
    starts = windows["states"]["phase"][:, 0]
    lengths = windows["valid"].sum(dim=1)
    # Each valid run is a contiguous step sequence inside one episode.
    for row in range(steps.shape[0]):
        run = int(lengths[row])
        assert run >= 1
        assert steps[row, :run].tolist() == [int(starts[row]) + offset for offset in range(run)]


def test_ring_wrap_keeps_windows_off_overwritten_history() -> None:
    capacity = 6
    replay = _make_replay(capacity=capacity)
    for step in range(12):
        replay.extend(_step_batch(step))

    windows = replay.sample_windows(batch_size=128, length=6)
    steps = windows["states"]["phase"]
    # Valid runs never chain across the wrap: values jump by more than one
    # exactly where the mask ends.
    for row in range(steps.shape[0]):
        run = int(windows["valid"][row].sum())
        assert run >= 1
        assert steps[row, :run].tolist() == [int(steps[row, 0]) + offset for offset in range(run)]


def test_sample_windows_guards() -> None:
    replay = _make_replay()
    # Empty and short replays both fail the stored-count guard (the window
    # needs ``length`` chained slots); the start sampler reports emptiness.
    with pytest.raises(ValueError, match="too few transitions"):
        replay.sample_windows(batch_size=4, length=2)
    with pytest.raises(ValueError, match="empty replay"):
        replay.sample_starts(batch_size=4)
    replay.extend(_step_batch(0))
    with pytest.raises(ValueError, match="too few transitions"):
        replay.sample_windows(batch_size=4, length=2)
    with pytest.raises(ValueError, match="at least 1"):
        replay.sample_windows(batch_size=4, length=0)


# ---------------------------------------------------------------------------
# Imagination starts and history stacks
# ---------------------------------------------------------------------------


def test_sample_starts_previous_action_respects_episode_boundaries() -> None:
    replay = _make_replay()
    for step in range(5):
        done = torch.tensor([True, False, False]) if step == 1 else None
        batch = _step_batch(step, terminated=done)
        if step == 1:
            batch["actions"] = torch.full((_NUM_ENVS, _ACT_DIM), 7.0)
        replay.extend(batch)

    starts = replay.sample_starts(batch_size=128)
    intact = starts["intact"]
    steps = starts["states"]["phase"]
    previous = starts["previous_action"]
    assert intact.shape == (128,)
    for row in range(128):
        step = int(steps[row])
        if step == 0:
            # Episode start: the zero previous action is the correct value.
            assert bool(intact[row])
            assert torch.equal(previous[row], torch.zeros(_ACT_DIM))
        elif step == 2:
            # Lane 0 reset at step 1: its step-2 row links to the done slot.
            continue
        else:
            assert bool(intact[row])
            expected = 7.0 if step == 2 else float(step - 1) + 0.25
            assert torch.equal(previous[row], torch.full((_ACT_DIM,), expected))


def test_history_clamps_at_episode_start_and_gaps() -> None:
    replay = _make_replay()
    for step in range(4):
        done = torch.tensor([True, False, False]) if step == 1 else None
        replay.extend(_step_batch(step, terminated=done))

    starts = replay.sample_starts(batch_size=64)
    stacks = replay.history(starts["envs"], starts["positions"], "phase", 3)
    assert stacks.shape == (64, 3)
    newest = stacks[:, -1]
    oldest = stacks[:, 0]
    # Newest frame is the sampled slot's own value.
    assert torch.equal(newest, starts["states"]["phase"])
    # At episode starts the earliest frame repeats the first step's value.
    at_start = starts["states"]["phase"] == 0.0
    if bool(at_start.any()):
        assert torch.equal(oldest[at_start], torch.zeros(int(at_start.sum())))


# ---------------------------------------------------------------------------
# RNG stream
# ---------------------------------------------------------------------------


def test_rng_state_round_trip_restores_stream() -> None:
    replay = _make_replay(seed=7)
    for step in range(4):
        replay.extend(_step_batch(step))
    state = replay.rng_state()
    first = replay.sample_windows(batch_size=8, length=2)

    twin = _make_replay(seed=7)
    for step in range(4):
        twin.extend(_step_batch(step))
    twin.load_rng_state(state)
    second = twin.sample_windows(batch_size=8, length=2)
    assert torch.equal(first["states"]["phase"], second["states"]["phase"])
    assert torch.equal(first["previous_action"], second["previous_action"])

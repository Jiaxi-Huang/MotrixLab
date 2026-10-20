# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-source adaptive sampling: probability math and window projection."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from motrix_envs.motion.adaptive import (
    AdaptiveMotionSampler,
    build_adaptive_probabilities,
    cap_per_bin_probability,
    clip_segment_bounds,
    count_segment_bins,
)


# ---------------------------------------------------------------------
# Probability builder (gear_sonic update_adaptive_sampling_probabilities)
# ---------------------------------------------------------------------
def test_rate_signal_not_volume() -> None:
    """Equal failure counts mean different difficulty at different exposure."""
    weights = np.ones(4, dtype=np.float32)
    baseline = build_adaptive_probabilities(
        np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
        weights,
        np.float32(0.2),
        np.float32(50.0),
        np.float32(0.0),
    )
    failures = np.ones(4, dtype=np.float32)
    episodes = np.ones(4, dtype=np.float32)
    episodes[2] = 0.1  # bin 2 fails as often, in 10x less exposure
    focused = build_adaptive_probabilities(
        failures,
        episodes,
        weights,
        np.float32(0.2),
        np.float32(50.0),
        np.float32(0.0),
    )

    np.testing.assert_allclose(baseline.sum(), 1.0)
    np.testing.assert_allclose(focused.sum(), 1.0)
    assert np.all(focused > 0.0)
    assert np.argmax(focused) == 2
    assert focused[2] > baseline[2]
    # Count-degenerate: equal failure counts with equal exposure stay uniform —
    # sampling volume alone must never shape the distribution.
    degenerate = build_adaptive_probabilities(
        np.full(4, 3.0, np.float32),
        np.full(4, 5.0, np.float32),
        weights,
        np.float32(0.2),
        np.float32(50.0),
        np.float32(0.0),
    )
    np.testing.assert_allclose(degenerate, np.full(4, 0.25, np.float32), atol=1e-7)


def test_outlier_rate_clips_at_mean_multiple() -> None:
    num_bins = 100
    failures = np.ones(num_bins, np.float32)
    failures[0] = 1000.0  # one extreme outlier bin: rate 100 vs 0.1 elsewhere
    episodes = np.full(num_bins, 10.0, np.float32)
    # Two clips of fifty bins each; equal weights inside a clip.
    weights = np.full(num_bins, 1.0 / 50.0, np.float32)

    got = build_adaptive_probabilities(
        failures,
        episodes,
        weights,
        np.float32(0.0),
        np.float32(50.0),
        np.float32(0.0),
    )
    np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    rates = failures / episodes
    bound = rates.mean() * 50.0
    assert rates[0] > bound  # pre-condition: the outlier exceeds the clip
    assert np.argmax(got) == 0
    # Post-clip, the outlier's share is reduced below its raw rate share
    # instead of monopolizing the distribution.
    uncapped_share = rates[0] / rates.sum()
    assert got[0] < uncapped_share


def test_uniform_blend_and_clip_weights_reshape() -> None:
    num_bins = 100
    episodes = np.full(num_bins, 10.0, np.float32)
    weights = np.full(num_bins, 1.0 / 50.0, np.float32)

    even = build_adaptive_probabilities(
        np.full(num_bins, 2.0, np.float32),
        episodes,
        weights,
        np.float32(0.0),
        np.float32(50.0),
        np.float32(0.0),
    )
    np.testing.assert_allclose(even, np.full(num_bins, 1.0 / num_bins, np.float32), atol=1e-7)

    # Unequal clip weights reshape even rates toward the heavier clip.
    weights[50:] = np.float32(3.0 / 50.0)
    weighted = build_adaptive_probabilities(
        np.full(num_bins, 2.0, np.float32),
        episodes,
        weights,
        np.float32(0.0),
        np.float32(50.0),
        np.float32(0.0),
    )
    assert weighted[50:].sum() > weighted[:50].sum()


def test_water_fill_cap_survives_normalization() -> None:
    num_bins = 100
    failures = np.ones(num_bins, np.float32)
    failures[0] = 1000.0  # outlier: rate 100 vs 0.1 elsewhere, share ~0.91 uncapped
    episodes = np.full(num_bins, 10.0, np.float32)
    weights = np.full(num_bins, 1.0 / 50.0, np.float32)

    got = build_adaptive_probabilities(
        failures,
        episodes,
        weights,
        np.float32(0.0),
        np.float32(50.0),
        np.float32(5.0),
    )
    assert got.max() <= np.float32(5.0 / num_bins) + 1e-6
    np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-5)
    assert np.argmax(got) == 0  # still preferred, just bounded
    assert got[1] > got[-1] - 1e-6 and got[1] > 0.0


def test_cap_is_identity_below_the_bound() -> None:
    uniform = np.full(8, 1.0 / 8, np.float32)
    np.testing.assert_array_equal(cap_per_bin_probability(uniform, np.float32(5.0)), uniform)


def test_zero_mass_falls_back_to_uniform() -> None:
    got = build_adaptive_probabilities(
        np.zeros(4, np.float32),
        np.full(4, 5.0, np.float32),
        np.ones(4, np.float32),
        np.float32(0.2),
        np.float32(50.0),
        np.float32(0.0),
    )
    np.testing.assert_allclose(got, np.full(4, 0.25, np.float32), atol=1e-7)


def test_segment_bounds_and_bin_count() -> None:
    frame_clip_end = np.asarray([2, 2, 2, 6, 6, 6, 6, 8, 8], np.int64)
    assert clip_segment_bounds(frame_clip_end, 0, 9) == [(0, 3), (3, 7), (7, 9)]
    # Window-relative walk over the first clip only.
    assert clip_segment_bounds(frame_clip_end, 0, 7) == [(0, 3), (3, 7)]
    assert count_segment_bins([(0, 3), (3, 7), (7, 9)], 2) == 2 + 2 + 1
    assert count_segment_bins([(0, 5)], 2) == 3  # ceil(5 / 2)


def test_segment_bounds_terminate_on_degenerate_boundary() -> None:
    # A boundary that does not advance (corrupt data) must degrade to one-
    # frame segments instead of looping — the regression guard for the
    # unbounded-list MemoryError this walk could produce.
    degenerate = np.asarray([1, 1, 1], np.int64)
    assert clip_segment_bounds(degenerate, 0, 3) == [(0, 2), (2, 3)]


# ---------------------------------------------------------------------
# Sampler projection and statistics over a fake command term
# ---------------------------------------------------------------------
@dataclass
class _FakeClip:
    joint_pos: np.ndarray
    frame_clip_end: np.ndarray
    frame_source_id: np.ndarray
    frame_source_start: np.ndarray


@dataclass
class _FakeTerm:
    clip: _FakeClip
    steps: np.ndarray
    window_range: np.ndarray
    sampling_cdf: np.ndarray
    adaptive_active_bin_count: np.ndarray
    adaptive_bin_start: np.ndarray
    adaptive_bin_end: np.ndarray
    adaptive_failure_pending: np.ndarray
    adaptive_failure_source_id: np.ndarray
    adaptive_failure_local_step: np.ndarray = field(default_factory=lambda: np.zeros(2, np.int64))


def _term(clips: list[int], bin_size: int, num_envs: int = 2) -> _FakeTerm:
    """Build a resident-corpus term: one entry per (frames,) clip length."""
    offsets = _offsets(clips)
    frame_clip_end = np.concatenate([np.full(frames, start + frames - 1, np.int64) for start, frames in offsets])
    source_id = np.concatenate([np.full(frames, index, np.int64) for index, frames in enumerate(clips)])
    source_start = np.concatenate([np.full(frames, start, np.int64) for start, frames in offsets])
    total = sum(clips)
    capacity = sum(-(-frames // bin_size) for frames in clips)
    return _FakeTerm(
        clip=_FakeClip(np.zeros((total, 2), np.float32), frame_clip_end, source_id, source_start),
        steps=np.zeros((num_envs, 1), np.int64),
        window_range=np.asarray([0, total], np.int64),
        sampling_cdf=np.zeros(capacity, np.float32),
        adaptive_active_bin_count=np.zeros(1, np.int64),
        adaptive_bin_start=np.empty(capacity, np.int64),
        adaptive_bin_end=np.empty(capacity, np.int64),
        adaptive_failure_pending=np.zeros(num_envs, bool),
        adaptive_failure_source_id=np.full(num_envs, -1, np.int64),
    )


def _offsets(clips: list[int]) -> list[tuple[int, int]]:
    offsets = []
    start = 0
    for frames in clips:
        offsets.append((start, frames))
        start += frames
    return offsets


def _sampler(**overrides) -> AdaptiveMotionSampler:
    kwargs = {
        "bin_size": 2,
        "init_count": 1.0,
        "uniform_ratio": 0.2,
        "failure_rate_max_over_mean": 50.0,
        "update_period": 1,
    }
    kwargs.update(overrides)
    return AdaptiveMotionSampler(**kwargs)


def test_projection_builds_source_local_bins() -> None:
    # Two clips (6 + 4 frames), bin_size 2 -> bins [0,2),[2,4),[4,6) and
    # [6,8),[8,10). A bin never straddles the clip boundary at frame 6.
    term = _term([6, 4], bin_size=2)
    _sampler().attach(term)

    assert int(term.adaptive_active_bin_count[0]) == 5
    np.testing.assert_array_equal(term.adaptive_bin_start[:5], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(term.adaptive_bin_end[:5], [2, 4, 6, 8, 10])
    assert term.sampling_cdf[4] == pytest.approx(1.0)
    assert np.all(np.diff(term.sampling_cdf[:5]) >= 0.0)


def test_initial_distribution_equalizes_clip_mass() -> None:
    # With no failures recorded, every rate is init/init = 1, so the mixture
    # is the uniform floor over sequence-length-agnostic weights: each clip
    # carries the same total mass regardless of bin count.
    term = _term([6, 4], bin_size=2)
    _sampler().attach(term)
    count = int(term.adaptive_active_bin_count[0])
    probabilities = np.diff(np.concatenate(([0.0], term.sampling_cdf[:count])))
    np.testing.assert_allclose(probabilities[:3].sum(), probabilities[3:].sum(), rtol=1e-5)


def test_failures_focus_their_bin_and_exposure_dilutes() -> None:
    term = _term([6, 4], bin_size=2)
    sampler = _sampler()
    sampler.attach(term)

    # Two failures in source 0, bin 1 (local frames [2, 4)).
    term.adaptive_failure_source_id[:] = 0
    term.adaptive_failure_local_step[:] = 2
    term.adaptive_failure_pending[:] = True
    sampler.tick(term)

    count = int(term.adaptive_active_bin_count[0])
    probabilities = np.diff(np.concatenate(([0.0], term.sampling_cdf[:count])))
    assert probabilities[1] > probabilities[0]
    assert probabilities[1] > probabilities[3]

    # Exposure in the failed bin lowers its rate back toward the pack: two
    # lanes parked on frame 2 for one step each add 1 episode of exposure.
    term.steps[:] = 2
    exposure_before = sampler.source_episode_mass(0)
    sampler.tick(term)
    assert sampler.source_episode_mass(0) == pytest.approx(exposure_before + 1.0)


def test_exposure_counts_episodes_per_bin_length() -> None:
    # bin_size 2: one lane-step in a full bin adds 1/2; the clip's tail bin
    # has length 1, so a lane-step there adds a full episode.
    term = _term([5], bin_size=2, num_envs=1)
    sampler = _sampler()
    sampler.attach(term)

    term.steps[0, 0] = 4  # local step 4 -> tail bin of length 1
    before = sampler.source_episode_mass(0)
    sampler.tick(term)
    assert sampler.source_episode_mass(0) == pytest.approx(before + 1.0)

    term.steps[0, 0] = 0  # head bin of length 2
    before = sampler.source_episode_mass(0)
    sampler.tick(term)
    assert sampler.source_episode_mass(0) == pytest.approx(before + 0.5)


def test_window_change_reprojects_eagerly() -> None:
    # A two-window axis: the second half carries different clips. Changing
    # window_range (a rotation publish) must republish the bin tables at the
    # next tick, without waiting for the update period.
    clips = [6, 4, 5, 3]
    offsets = []
    start = 0
    for frames in clips:
        offsets.append((start, frames))
        start += frames
    frame_clip_end = np.concatenate([np.full(frames, s + frames - 1, np.int64) for s, frames in offsets])
    source_id = np.concatenate([np.full(frames, i, np.int64) for i, frames in enumerate(clips)])
    source_start = np.concatenate([np.full(frames, s, np.int64) for s, frames in offsets])
    capacity = sum(-(-frames // 2) for frames in clips)

    term = _FakeTerm(
        clip=_FakeClip(np.zeros((start, 2), np.float32), frame_clip_end, source_id, source_start),
        steps=np.zeros((2, 1), np.int64),
        window_range=np.asarray([0, 10], np.int64),
        sampling_cdf=np.zeros(capacity, np.float32),
        adaptive_active_bin_count=np.zeros(1, np.int64),
        adaptive_bin_start=np.empty(capacity, np.int64),
        adaptive_bin_end=np.empty(capacity, np.int64),
        adaptive_failure_pending=np.zeros(2, bool),
        adaptive_failure_source_id=np.full(2, -1, np.int64),
    )
    sampler = _sampler(update_period=10_000)
    sampler.attach(term)
    assert int(term.adaptive_active_bin_count[0]) == 5  # clips 0 + 1

    term.window_range[:] = (10, 8)  # clips 2 + 3
    sampler.tick(term)
    # ceil(5/2) + ceil(3/2) = 5 bins; each clip's tail bin is short.
    assert int(term.adaptive_active_bin_count[0]) == 5
    np.testing.assert_array_equal(term.adaptive_bin_start[:5], [10, 12, 14, 15, 17])
    np.testing.assert_array_equal(term.adaptive_bin_end[:5], [12, 14, 15, 17, 18])
    assert term.sampling_cdf[4] == pytest.approx(1.0)


def test_source_sampling_weights_rank_learned_difficulty() -> None:
    # Three sources, two registered: source 1 fails far more often. Weights
    # rank it first, keep a uniform floor for every entry (including the
    # never-seen source 2), and sum to one.
    term = _term([6, 6, 6], bin_size=2, num_envs=2)
    sampler = _sampler()
    sampler.attach(term)

    for _ in range(20):  # lanes sit in source 0's head bin, exposure only
        sampler.tick(term)
    term.adaptive_failure_source_id[:] = 1
    term.adaptive_failure_local_step[:] = 0
    term.adaptive_failure_pending[:] = True
    sampler.tick(term)
    sampler.tick(term)  # second tick republishes and invalidates the cache

    weights = sampler.source_sampling_weights(3)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    # Hard clip ranks first; the never-seen clip keeps its initial rate of one
    # (above the exposure-diluted easy clip); everything stays positive.
    assert weights[1] > weights[2] > weights[0] > 0.0


def test_exposure_folds_into_global_counters_at_republish() -> None:
    # With a long update period, per-step exposure lands in the window-local
    # histogram (visible through source_episode_mass) and only folds into
    # the global counters at the next projection.
    term = _term([6, 4], bin_size=2)
    sampler = _sampler(update_period=10_000)
    sampler.attach(term)
    offset = int(sampler._offset_lut[0])

    term.steps[:] = 0
    sampler.tick(term)
    assert sampler.source_episode_mass(0) == pytest.approx(1.0)  # 2 lanes x 1/2
    assert sampler._episodes[offset] == pytest.approx(1.0)  # still the init prior

    sampler._project(term)
    assert sampler._episodes[offset] == pytest.approx(2.0)  # folded
    assert sampler.source_episode_mass(0) == pytest.approx(1.0)  # unchanged by the fold
    assert float(np.sum(sampler._pending_exposure[:5])) == pytest.approx(0.0)


def test_draining_lane_exposure_reaches_its_source() -> None:
    # A lane stranded on a retired window still credits its own source:
    # outside-window steps scatter directly into the global counters.
    term = _term([6, 4, 6], bin_size=2)
    sampler = _sampler(update_period=10_000)
    term.window_range[:] = (0, 16)
    sampler.attach(term)  # registers sources 0, 1, 2
    term.window_range[:] = (0, 10)  # source 2's clip leaves the window
    term.steps[0, 0] = 12  # draining lane on the retired clip
    term.steps[1, 0] = 1  # in-window lane
    sampler.tick(term)

    assert sampler.source_episode_mass(2) == pytest.approx(0.5)  # direct global scatter
    assert sampler.source_episode_mass(0) == pytest.approx(0.5)  # window-local histogram


def test_realized_failure_rate_tracks_performance_under_sampling() -> None:
    # Two lanes parked in a length-2 bin contribute one episode unit per
    # step; one failure per step therefore converges to a rate of ~1.0, and
    # the counter decays once failures stop.
    term = _term([6, 4], bin_size=2, num_envs=2)
    sampler = _sampler(update_period=10_000)
    sampler.attach(term)
    sampler._rate_decay = 0.9  # short half-life for the test
    term.steps[:] = 0

    for _ in range(10):
        sampler.tick(term)
    assert sampler.metrics()["adaptive_failure_rate"] == pytest.approx(0.0, abs=1e-6)

    # One lane fails per step against one episode unit of exposure.
    for _ in range(30):
        term.adaptive_failure_pending[0] = True
        term.adaptive_failure_source_id[0] = 0
        term.adaptive_failure_local_step[0] = 0
        sampler.tick(term)
    rate = sampler.metrics()["adaptive_failure_rate"]
    assert 0.85 < rate < 1.1

    for _ in range(40):  # ~4 half-lives without failures
        sampler.tick(term)
    assert sampler.metrics()["adaptive_failure_rate"] < 0.15


def test_tick_reports_metrics() -> None:
    term = _term([6, 4], bin_size=2)
    sampler = _sampler()
    sampler.attach(term)
    # Startup distribution: clip-equalized uniform (three 1/6 bins plus two
    # 1/4 bins), not max-entropy uniform over bins.
    startup = sampler.metrics()
    assert startup["adaptive_sampling_entropy"] == pytest.approx(0.9873, abs=1e-3)
    assert startup["adaptive_sampling_top1_prob"] == pytest.approx(0.25)
    assert startup["adaptive_failure_mass"] == pytest.approx(0.0)

    metrics = sampler.tick(term)
    for key in (
        "adaptive_sampling_entropy",
        "adaptive_sampling_top1_prob",
        "adaptive_sampling_top1_bin",
        "adaptive_failure_mass",
        "adaptive_failure_rate",
    ):
        assert key in metrics

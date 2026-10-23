# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Success Guided Sampling: kernel math and window projection over fake terms."""

from dataclasses import dataclass

import numpy as np
import pytest

from motrix_envs.motion.sgs import SuccessGuidedSampler, build_sgs_probabilities


# ---------------------------------------------------------------------
# Beta-kernel softmax (paper Eqs. 1-2)
# ---------------------------------------------------------------------
def test_kernel_peaks_at_target_success_rate() -> None:
    rates = np.asarray([0.2, 0.66, 0.95])
    got = build_sgs_probabilities(rates, target=0.66, concentration=5.0, epsilon=1e-2, temperature=2.0)
    np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    assert int(np.argmax(got)) == 1
    # The kernel is Beta(1 + kappa*t, 1 + kappa*(1 - t)) — mode exactly at
    # the target — which at t = 0.66 carries more density at 0.95 than 0.2.
    assert got[2] > got[0]
    assert np.all(got > 0.0)


def test_zero_concentration_and_high_temperature_flatten() -> None:
    rates = np.asarray([0.1, 0.5, 0.9])
    # kappa = 0 collapses the kernel to a constant: exact uniform.
    flat = build_sgs_probabilities(rates, target=0.66, concentration=0.0, epsilon=1e-8, temperature=2.0)
    np.testing.assert_allclose(flat, np.full(3, 1.0 / 3.0), atol=1e-7)
    # Uninformative rates all sit at the prior: uniform regardless of kappa.
    prior = build_sgs_probabilities(np.full(4, 0.5), target=0.66, concentration=5.0, epsilon=1e-8, temperature=2.0)
    np.testing.assert_allclose(prior, np.full(4, 0.25), atol=1e-7)
    # A hotter softmax softens the same kernel's ordering.
    sharp = build_sgs_probabilities(rates, 0.5, 5.0, 1e-2, 0.5)
    hot = build_sgs_probabilities(rates, 0.5, 5.0, 1e-2, 50.0)
    assert np.max(hot) < np.max(sharp)
    np.testing.assert_allclose(hot.sum(), 1.0, rtol=1e-6)


def test_epsilon_keeps_extreme_rates_nonzero() -> None:
    rates = np.asarray([0.0, 1.0])
    got = build_sgs_probabilities(rates, target=0.66, concentration=5.0, epsilon=1e-8, temperature=2.0)
    np.testing.assert_allclose(got.sum(), 1.0, rtol=1e-6)
    assert np.all(got > 0.0)


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
    adaptive_failure_local_step: np.ndarray
    adaptive_outcome_success: np.ndarray


def _term(clips: list[int], bin_size: int, num_envs: int = 2) -> _FakeTerm:
    """Build a resident-corpus term: one entry per (frames,) clip length."""
    offsets = []
    start = 0
    for frames in clips:
        offsets.append((start, frames))
        start += frames
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
        adaptive_failure_local_step=np.zeros(num_envs, np.int64),
        adaptive_outcome_success=np.zeros(num_envs, bool),
    )


def _record(term: _FakeTerm, lane: int, source: int, local_step: int, success: bool) -> None:
    term.adaptive_failure_source_id[lane] = source
    term.adaptive_failure_local_step[lane] = local_step
    term.adaptive_outcome_success[lane] = success
    term.adaptive_failure_pending[lane] = True


def _probabilities(term: _FakeTerm) -> np.ndarray:
    count = int(term.adaptive_active_bin_count[0])
    return np.diff(np.concatenate(([0.0], term.sampling_cdf[:count])))


def _sampler(**overrides) -> SuccessGuidedSampler:
    kwargs = {
        "bin_size": 2,
        "history": 3,
        "target": 0.66,
        "concentration": 5.0,
        "epsilon": 1e-2,
        "temperature": 2.0,
        "update_period": 1,
    }
    kwargs.update(overrides)
    return SuccessGuidedSampler(**kwargs)


def test_initial_distribution_is_uniform_before_any_outcome() -> None:
    # Unseen bins carry the 0.5 prior, where the kernel weight is flat, so
    # sampling starts uniform exactly as in the paper's pre-training phase.
    term = _term([2, 2, 2], bin_size=2)
    _sampler().attach(term)
    assert int(term.adaptive_active_bin_count[0]) == 3
    np.testing.assert_allclose(_probabilities(term), np.full(3, 1.0 / 3.0), atol=1e-7)
    assert term.sampling_cdf[2] == pytest.approx(1.0)


def test_outcomes_concentrate_mass_at_the_policy_frontier() -> None:
    # Three one-bin sources: fully solved (rate 1.0), frontier (rate 2/3 —
    # the target), unsolved (rate 0.0). SGS samples the frontier bin the most
    # and prefers the mastered bin over the unsolved one.
    term = _term([2, 2, 2], bin_size=2, num_envs=3)
    sampler = _sampler()
    sampler.attach(term)
    for _ in range(3):
        _record(term, 0, 0, 0, True)  # solved
        _record(term, 1, 2, 0, False)  # unsolved
        sampler.tick(term)
    # Frontier bin: two successes and one failure across three lanes.
    _record(term, 0, 1, 0, True)
    _record(term, 1, 1, 0, True)
    _record(term, 2, 1, 0, False)
    sampler.tick(term)
    assert sampler.bin_success_rate(0, 0) == pytest.approx(1.0)
    assert sampler.bin_success_rate(1, 0) == pytest.approx(2.0 / 3.0)
    assert sampler.bin_success_rate(2, 0) == pytest.approx(0.0)

    probabilities = _probabilities(term)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=1e-6)
    assert np.all(probabilities > 0.0)
    assert int(np.argmax(probabilities)) == 1
    assert probabilities[0] > probabilities[2]
    assert not term.adaptive_failure_pending.any()


def test_success_rate_is_a_last_h_window_not_a_lifetime_mean() -> None:
    # The estimate must keep adapting after the policy improves: a bin whose
    # early failures age out of the window recovers its sampling mass.
    term = _term([2], bin_size=2)
    sampler = _sampler(history=2)
    sampler.attach(term)
    for _ in range(2):
        _record(term, 0, 0, 0, False)
        sampler.tick(term)
    assert sampler.bin_success_rate(0, 0) == pytest.approx(0.0)
    for _ in range(2):
        _record(term, 0, 0, 0, True)
        sampler.tick(term)
    assert sampler.bin_success_rate(0, 0) == pytest.approx(1.0)


def test_metrics_and_source_weights_follow_learned_rates() -> None:
    term = _term([2, 2, 2], bin_size=2, num_envs=3)
    sampler = _sampler()
    sampler.attach(term)
    metrics = sampler.tick(term)
    for key in ("sgs_sampling_entropy", "sgs_sampling_top1_prob", "sgs_mean_success_rate"):
        assert key in metrics
    assert metrics["sgs_mean_success_rate"] == pytest.approx(0.0)

    weights = sampler.source_sampling_weights(5)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    # Unseen sources (3, 4) share the prior; every registered source is still
    # at the prior, so the initial weights are uniform.
    np.testing.assert_allclose(weights, np.full(5, 0.2), atol=1e-7)

    # Source 1 reaches the frontier (two successes, one failure); sources 0
    # and 2 sit at the extremes.
    for _ in range(3):
        _record(term, 0, 0, 0, True)
        _record(term, 1, 2, 0, False)
        sampler.tick(term)
    _record(term, 0, 1, 0, True)
    _record(term, 1, 1, 0, True)
    _record(term, 2, 1, 0, False)
    sampler.tick(term)

    weights = sampler.source_sampling_weights(5)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    assert int(np.argmax(weights)) == 1
    assert np.all(weights > 0.0)
    metrics = sampler.metrics()
    assert 0.0 < metrics["sgs_mean_success_rate"] < 1.0
    assert 0.0 < metrics["sgs_sampling_entropy"] <= 1.0


def test_multi_bin_source_aggregates_and_window_changes_keep_statistics() -> None:
    # One 4-frame source (two bins) plus one 2-frame source. Window changes
    # republish tables; per-source statistics survive them because outcomes
    # address source-local bins, not window slots.
    term = _term([4, 2], bin_size=2)
    sampler = _sampler(update_period=1000)
    sampler.attach(term)
    _record(term, 0, 0, 0, True)  # source 0, bin 0: success
    _record(term, 1, 0, 3, False)  # source 0, bin 1: failure
    sampler.tick(term)
    assert sampler.bin_success_rate(0, 0) == pytest.approx(1.0)
    assert sampler.bin_success_rate(0, 1) == pytest.approx(0.0)
    # Source-level aggregation mixes both bins' rates for rotation weights;
    # the mean-0.5 source ties the unseen source's prior.
    weights = sampler.source_sampling_weights(2)
    np.testing.assert_allclose(weights, np.full(2, 0.5), atol=1e-7)

    # Shrink the active window to source 0 only (a rotation swap): tables
    # republish over the new domain while the drained statistics survive.
    term.window_range[1] = 4
    sampler.tick(term)
    assert int(term.adaptive_active_bin_count[0]) == 2
    assert sampler.bin_success_rate(0, 0) == pytest.approx(1.0)
    term.window_range[1] = 6
    sampler.tick(term)
    assert int(term.adaptive_active_bin_count[0]) == 3
    assert sampler.bin_success_rate(0, 1) == pytest.approx(0.0)

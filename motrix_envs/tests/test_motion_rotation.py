# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Corpus rotation behavior: coverage, drain gating, publication, determinism."""

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_envs.locomotion.sonic.mdp import G1_SONIC_BODY_NAMES, G1_SONIC_JOINTS
from motrix_envs.motion.adaptive import AdaptiveMotionSampler
from motrix_envs.motion.rotation import CorpusRotator
from motrix_envs.motion.sgs import SuccessGuidedSampler

_TASK_ORDER = {
    "joint_names": ["joint_2", "joint_0"],
    "tracked_body_names": ("body_1", "body_3"),
    "reference_body_name": "body_2",
    "root_body_name": "body_0",
}

# Per-file frame counts; the constant joint_pos tag ``index + 1`` identifies
# each file inside a published window (0.0 is the unwritten capacity default).
_FRAMES = [6, 10, 4, 8, 12, 6, 10, 8]
_SEED = 11


def _write_clip(path, *, num_frames, tag):
    joints = ("joint_0", "joint_1", "joint_2")
    body_names = ("body_0", "body_1", "body_2", "body_3")
    num_bodies = len(body_names)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), np.float32)
    body_quat_w[..., 3] = 1.0
    smpl_quat = np.zeros((num_frames, 4), np.float32)
    smpl_quat[:, 3] = 1.0
    np.savez(
        path,
        schema_version=np.int32(1),
        fps=np.int32(50),
        num_frames=np.int32(num_frames),
        joint_names=np.asarray(joints),
        body_names=np.asarray(body_names),
        joint_pos=np.full((num_frames, len(joints)), tag, np.float32),
        joint_vel=np.zeros((num_frames, len(joints)), np.float32),
        body_pos_w=np.zeros((num_frames, num_bodies, 3), np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((num_frames, num_bodies, 3), np.float32),
        body_ang_vel_w=np.zeros((num_frames, num_bodies, 3), np.float32),
        clip_name=np.asarray(f"clip_{tag}"),
        ext_smpl_joints=np.zeros((num_frames, 24, 3), np.float32),
        ext_smpl_root_quat=smpl_quat,
    )


@pytest.fixture()
def corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    for index, frames in enumerate(_FRAMES):
        _write_clip(directory / f"clip_{index}.npz", num_frames=frames, tag=float(index + 1))
    return directory


@dataclass
class _FakeTerm:
    clip: object
    steps: np.ndarray
    window_range: np.ndarray
    sampling_cdf: np.ndarray = field(default_factory=lambda: np.linspace(0.25, 1.0, 4).astype(np.float32))


def _term(clip, rotator):
    return _FakeTerm(clip=clip, steps=np.zeros((4, 1), np.int64), window_range=np.asarray([0, rotator.active_frames]))


def _adaptive_term(clip, rotator):
    term = _term(clip, rotator)
    capacity = rotator.max_adaptive_bins(2)
    term.sampling_cdf = np.zeros(capacity, np.float32)
    term.adaptive_active_bin_count = np.zeros(1, np.int64)
    term.adaptive_bin_start = np.empty(capacity, np.int64)
    term.adaptive_bin_end = np.empty(capacity, np.int64)
    term.adaptive_failure_pending = np.zeros(term.steps.shape[0], bool)
    term.adaptive_failure_source_id = np.full(term.steps.shape[0], -1, np.int64)
    term.adaptive_failure_local_step = np.zeros(term.steps.shape[0], np.int64)
    return term


def _build(corpus_directory, **overrides):
    kwargs = {
        "paths": (corpus_directory,),
        "window_clips": 2,
        "period": 3,
        "workers": 2,
        "seed": _SEED,
        "fps": 50,
        **_TASK_ORDER,
    }
    kwargs.update(overrides)
    return CorpusRotator.build(**kwargs)


def _active_tags(clip, rotator):
    base = rotator.active_base
    frames = rotator.active_frames
    return {float(value) for value in np.unique(clip.joint_pos[base : base + frames])}


def _drive(rotator, term, *, until_window, max_ticks=10_000):
    for _ in range(max_ticks):
        if rotator.window_index >= until_window:
            return
        rotator.tick(term)
        # Lanes migrate to the active window between swaps, as real envs do
        # through resets and clip-end re-anchoring. The sleep also yields the
        # GIL so the decode pool can progress between (here, near-instant)
        # control steps.
        term.steps[:] = rotator.active_base
        time.sleep(0.0005)
    raise AssertionError(f"rotation did not reach window {until_window}")


def test_build_allocates_two_window_capacity(corpus):
    clip, rotator = _build(corpus)
    first = np.random.default_rng(_SEED).permutation(len(_FRAMES))[:2]
    first_frames = sum(_FRAMES[index] for index in first)
    assert clip.joint_pos.shape[0] == 2 * int(np.ceil(first_frames * 1.25))
    assert rotator.active_base == 0
    assert rotator.active_frames == first_frames
    # Window 0 keeps per-clip boundaries; the unused half stays unwritten.
    assert clip.frame_clip_end[first_frames - 1] == first_frames - 1
    assert set(np.unique(clip.joint_pos[first_frames:])) == {0.0}
    assert _active_tags(clip, rotator) == {float(index + 1) for index in first}


def test_rotation_serves_each_clip_once_per_pass(corpus):
    clip, rotator = _build(corpus)
    term = _term(clip, rotator)
    all_tags = {float(index + 1) for index in range(len(_FRAMES))}
    corpus_size = len(_FRAMES)
    # Window 0 is part of the pass: the next window must not repeat it (the
    # permutation cursor starts past the build-time prefix).
    assert rotator.pass_coverage == pytest.approx(2 / corpus_size)
    seen: set[float] = _active_tags(clip, rotator)
    for target in range(1, 12):
        term.steps[:] = rotator.active_base
        _drive(rotator, term, until_window=target)
        tags = _active_tags(clip, rotator)
        if len(seen) < len(all_tags):
            # Without-replacement: nothing repeats before the pass completes.
            assert tags & seen == set(), f"window {target} repeated already-served clips {tags & seen}"
        seen |= tags
        if seen >= all_tags:
            break
    assert seen >= all_tags


def test_publish_gates_on_drain_of_retiring_half(corpus):
    clip, rotator = _build(corpus)
    term = _term(clip, rotator)
    # The first swap targets the empty half; the second must wait for window 0
    # (still occupied by parked lanes) to drain.
    _drive(rotator, term, until_window=1)
    term.steps[:] = 2
    for _ in range(200):
        rotator.tick(term)
        time.sleep(0.0005)
    assert rotator.window_index == 1
    term.steps[:] = rotator.active_base
    _drive(rotator, term, until_window=2)
    assert rotator.window_index == 2


def test_publish_flips_window_domain(corpus):
    clip, rotator = _build(corpus)
    term = _term(clip, rotator)
    _drive(rotator, term, until_window=1)
    base, frames = int(term.window_range[0]), int(term.window_range[1])
    assert base == rotator.active_base == rotator._frame_budget
    assert frames == rotator.active_frames > 0
    # Published clip-end boundaries are window-global and monotone.
    window_ends = clip.frame_clip_end[base : base + frames]
    assert window_ends[-1] == base + frames - 1
    assert np.all(np.diff(window_ends) >= 0)
    assert np.all((window_ends >= base) & (window_ends < base + frames))


def test_rotation_adaptive_statistics_follow_source_clips_and_draining_lanes(corpus):
    clip, rotator = _build(corpus, track_sources=True)
    term = _adaptive_term(clip, rotator)
    sampler = AdaptiveMotionSampler(
        bin_size=2,
        init_count=1.0,
        uniform_ratio=0.2,
        failure_rate_max_over_mean=50.0,
        update_period=1,
    )
    sampler.attach(term)

    source_id = int(clip.frame_source_id[0])
    term.adaptive_failure_source_id[0] = source_id
    term.adaptive_failure_local_step[0] = 2
    term.adaptive_failure_pending[0] = True
    sampler.tick(term)
    assert sampler.source_failure_mass(source_id) == pytest.approx(1.0)
    # The pending handoff is drained exactly once.
    assert not term.adaptive_failure_pending.any()

    # Publish the next window, then report a delayed failure from a lane still
    # draining the retired half.  It must return to the old source, not the
    # new window's matching time bin.
    _drive(rotator, term, until_window=1)
    term.adaptive_failure_source_id[1] = source_id
    term.adaptive_failure_local_step[1] = 3
    term.adaptive_failure_pending[1] = True
    sampler.tick(term)
    assert sampler.source_failure_mass(source_id) == pytest.approx(2.0)

    # The source eventually returns in the seeded without-replacement pass;
    # its learned bin must receive more than the fixed 20% uniform floor.
    for _ in range(12):
        if source_id in clip.frame_source_id[rotator.active_base : rotator.active_base + rotator.active_frames]:
            break
        _drive(rotator, term, until_window=rotator.window_index + 1)
    else:
        raise AssertionError("source clip did not return during the rotation pass")
    sampler.tick(term)
    count = int(term.adaptive_active_bin_count[0])
    probabilities = np.diff(np.concatenate(([0.0], term.sampling_cdf[:count])))
    assert np.all(probabilities >= 0.2 / count - 1e-7)
    active_sources = clip.frame_source_id[rotator.active_base : rotator.active_base + rotator.active_frames]
    source_start = rotator.active_base + int(np.flatnonzero(active_sources == source_id)[0])
    target_bin = int(np.flatnonzero(term.adaptive_bin_start[:count] == source_start + 2)[0])
    assert probabilities[target_bin] > 0.2 / count


def test_rotation_adaptive_path_compiles_and_preserves_uniform_controller_absence(tmp_path):
    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()
    for index in range(5):
        _write_g1_clip(corpus_directory / f"clip_{index}.npz", num_frames=8)

    uniform_cfg = registry.make_env_config("g1-sonic")
    uniform_cfg.commands.motion.motion_files = (str(corpus_directory),)
    uniform_cfg.commands.motion.num_motions = 2
    uniform_cfg.commands.motion.rotation_period = 2
    # The default enables the adaptive curriculum; pin plain uniform draws to
    # exercise the uniform rotation controller.
    uniform_cfg.commands.motion.sampling_mode = "uniform"
    uniform_env = registry.resolve("g1-sonic", env_cfg=uniform_cfg).make(num_envs=2, seed=1)
    assert uniform_env._motion_adaptive is None
    uniform_motion = uniform_env._command_terms["motion"]
    assert uniform_motion.sampling_cdf.size == 0
    assert uniform_motion.clip.frame_source_id.size == 0
    assert uniform_motion.clip.frame_source_start.size == 0
    uniform_env.init_state()

    adaptive_cfg = registry.make_env_config("g1-sonic")
    adaptive_cfg.commands.motion.motion_files = (str(corpus_directory),)
    adaptive_cfg.commands.motion.num_motions = 2
    adaptive_cfg.commands.motion.rotation_period = 2
    adaptive_cfg.commands.motion.sampling_mode = "adaptive"
    adaptive_env = registry.resolve("g1-sonic", env_cfg=adaptive_cfg).make(num_envs=2, seed=1)
    assert adaptive_env._motion_adaptive is not None
    adaptive_motion = adaptive_env._command_terms["motion"]
    # The sampler published per-source bins over the startup window.
    assert adaptive_motion.sampling_cdf.size > 0
    assert int(adaptive_motion.adaptive_active_bin_count[0]) > 0
    assert adaptive_motion.sampling_cdf[-1] == pytest.approx(1.0)
    adaptive_env.init_state()
    state = adaptive_env.step(np.zeros((2, len(G1_SONIC_JOINTS)), np.float32))
    for key in (
        "adaptive_sampling_entropy",
        "adaptive_sampling_top1_prob",
        "adaptive_failure_mass",
        "adaptive_failure_rate",
    ):
        assert key in state.metrics


def test_rotation_sgs_path_publishes_bins_and_records_outcomes(tmp_path):
    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()
    for index in range(5):
        _write_g1_clip(corpus_directory / f"clip_{index}.npz", num_frames=8)

    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus_directory),)
    cfg.commands.motion.num_motions = 2
    cfg.commands.motion.rotation_period = 2
    cfg.commands.motion.sampling_mode = "sgs"
    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)
    assert isinstance(env._motion_adaptive[1], SuccessGuidedSampler)
    motion = env._command_terms["motion"]
    # The initial tables publish like the adaptive path: uniform SGS mass
    # over per-source bins (no outcomes recorded yet).
    count = int(motion.adaptive_active_bin_count[0])
    assert count > 0
    assert motion.sampling_cdf[count - 1] == pytest.approx(1.0)
    probabilities = np.diff(np.concatenate(([0.0], motion.sampling_cdf[:count])))
    np.testing.assert_allclose(probabilities, np.full(count, 1.0 / count), atol=1e-6)
    assert motion.record_episode_outcomes
    env.init_state()
    state = env.step(np.zeros((2, len(G1_SONIC_JOINTS)), np.float32))
    for key in ("sgs_sampling_entropy", "sgs_sampling_top1_prob", "sgs_mean_success_rate"):
        assert key in state.metrics
    # init_state resets lanes as failures, so the first tick records outcomes.
    assert state.metrics["sgs_mean_success_rate"] == pytest.approx(0.0)


def test_budget_trim_keeps_coverage_via_carry(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    # Window 0 = [2, 2] frames -> budget 5; every later 4-frame clip fits only
    # one per window, so the decoder must carry the overflow forward instead
    # of dropping corpus files.
    frames = [2, 2, 4, 4, 4, 4]
    for index, count in enumerate(frames):
        _write_clip(directory / f"clip_{index}.npz", num_frames=count, tag=float(index + 1))
    clip, rotator = _build(directory)
    term = _term(clip, rotator)
    all_tags = {float(index + 1) for index in range(len(frames))}
    seen = _active_tags(clip, rotator)
    for target in range(1, 12):
        term.steps[:] = rotator.active_base
        _drive(rotator, term, until_window=target)
        assert rotator.active_frames <= rotator._frame_budget
        seen |= _active_tags(clip, rotator)
        if seen >= all_tags:
            break
    assert seen >= all_tags


def test_difficulty_slots_shape_window_composition(corpus):
    clip, rotator = _build(corpus)
    term = _term(clip, rotator)
    hot = 3  # concentrate every difficulty draw on one corpus entry

    def provider(count):
        weights = np.full(count, 1e-9)
        weights[hot] = 1.0
        return weights / weights.sum()

    # ratio 1.0: no permutation fill, deduplicated draws leave exactly the
    # hot clip in the published window.
    rotator.enable_difficulty_sampling(provider, 1.0)
    _drive(rotator, term, until_window=1)
    assert _active_tags(clip, rotator) == {float(hot + 1)}

    # ratio 0.5 with a four-clip window: the hot clip plus two permutation
    # entries; the coverage backbone keeps every clip reachable.
    clip, rotator = _build(corpus, window_clips=4)
    term = _term(clip, rotator)
    rotator.enable_difficulty_sampling(provider, 0.5)
    _drive(rotator, term, until_window=1)
    tags = _active_tags(clip, rotator)
    assert float(hot + 1) in tags
    assert len(tags) == 3  # dedup(2 draws) + 2 permutation slots, hot not repeated


def test_rotation_is_deterministic_for_a_fixed_seed(corpus):
    clip_a, rotator_a = _build(corpus)
    clip_b, rotator_b = _build(corpus)
    np.testing.assert_array_equal(clip_a.joint_pos, clip_b.joint_pos)
    term_a = _term(clip_a, rotator_a)
    term_b = _term(clip_b, rotator_b)
    for _ in range(3):
        for term, rotator in ((term_a, rotator_a), (term_b, rotator_b)):
            term.steps[:] = rotator.active_base
            _drive(rotator, term, until_window=rotator.window_index + 1)
    assert _active_tags(clip_a, rotator_a) == _active_tags(clip_b, rotator_b)
    assert rotator_a.active_frames == rotator_b.active_frames
    assert rotator_a.active_base == rotator_b.active_base


def test_rotation_stays_dormant_before_the_first_period(corpus):
    clip, rotator = _build(corpus, period=5)
    term = _term(clip, rotator)
    for _ in range(4):
        rotator.tick(term)
    assert rotator._pool is None
    assert rotator.window_index == 0
    rotator.tick(term)
    assert rotator._pool is not None


def test_small_corpus_falls_back_to_static_resident_clip(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    _write_clip(directory / "a.npz", num_frames=6, tag=1.0)
    _write_clip(directory / "b.npz", num_frames=10, tag=2.0)
    clip, rotator = _build(directory, window_clips=4)
    assert rotator is None
    assert clip.joint_pos.shape[0] == 16
    assert clip.frame_clip_end.tolist() == [5] * 6 + [15] * 10


def _write_g1_clip(path, num_frames):
    bodies = len(G1_SONIC_BODY_NAMES)
    identity = np.zeros((num_frames, bodies, 4), np.float32)
    identity[..., 3] = 1.0
    root = np.zeros((num_frames, 4), np.float32)
    root[:, 3] = 1.0
    body_pos = np.zeros((num_frames, bodies, 3), np.float32)
    body_pos[:, 0, 2] = 0.79
    np.savez(
        path,
        schema_version=np.int32(1),
        fps=np.int32(50),
        num_frames=np.int32(num_frames),
        joint_names=np.asarray(G1_SONIC_JOINTS),
        body_names=np.asarray(G1_SONIC_BODY_NAMES),
        joint_pos=np.zeros((num_frames, len(G1_SONIC_JOINTS)), np.float32),
        joint_vel=np.zeros((num_frames, len(G1_SONIC_JOINTS)), np.float32),
        body_pos_w=body_pos,
        body_quat_w=identity,
        body_lin_vel_w=np.zeros((num_frames, bodies, 3), np.float32),
        body_ang_vel_w=np.zeros((num_frames, bodies, 3), np.float32),
        root_body_name=np.asarray("pelvis"),
        reference_body_name=np.asarray("pelvis"),
        ext_smpl_joints=np.zeros((num_frames, 24, 3), np.float32),
        ext_smpl_root_quat=root,
    )


def test_sonic_env_rotates_windows_while_stepping(tmp_path):
    corpus_directory = tmp_path / "corpus"
    corpus_directory.mkdir()
    for index in range(5):
        _write_g1_clip(corpus_directory / f"clip_{index}.npz", num_frames=8)
    cfg = registry.make_env_config("g1-sonic")
    cfg.commands.motion.motion_files = (str(corpus_directory),)
    cfg.commands.motion.num_motions = 2
    cfg.commands.motion.rotation_period = 2
    env = registry.resolve("g1-sonic", env_cfg=cfg).make(num_envs=2, seed=1)
    motion = env._command_terms["motion"]
    rotator = env._motion_rotation[1]
    capacity = motion.clip.joint_pos.shape[0]

    state = env.init_state()
    joints = len(cfg.commands.motion.joint_names)
    policy_width = state.obs.policy.shape[1]
    for _ in range(60):
        state = env.step(np.zeros((2, joints), dtype=np.float32))
        assert np.isfinite(state.obs.policy).all()
        assert np.isfinite(state.obs.value).all()
        assert float(np.max(state.metrics["motion_step"])) < capacity
    assert rotator.window_index >= 2
    # The final publish can coincide with a seeded permutation wrap, where a
    # fresh pass has correctly consumed zero entries so far.
    assert 0.0 <= state.metrics["motion_rotation_coverage"] <= 1.0
    assert state.metrics["motion_rotation_window"] >= 2
    base, frames = int(motion.window_range[0]), int(motion.window_range[1])
    assert 0 <= base and base + frames <= capacity
    assert motion.clip.frame_clip_end[base] <= base + frames - 1
    state = env.step(np.zeros((2, joints), dtype=np.float32))
    assert state.obs.policy.shape[1] == policy_width

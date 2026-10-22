# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic task contracts: start-bin machinery, return sampler, and env kernels."""

import numpy as np
import pytest

from motrix_env_core import registry  # noqa: E402
from motrix_envs.locomotion.supermimic.g1 import (  # noqa: E402
    SuperMimicManagerEnv,
    make_g1_supermimic_cfg,
)
from motrix_envs.locomotion.supermimic.mdp import (  # noqa: E402
    SUPERMIMIC_MAX_STEP_REWARD,
    SuperMimicMotionCommand,
    build_start_bins,
)
from motrix_envs.locomotion.supermimic.sampling import SuperMimicStartSampler  # noqa: E402
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR  # noqa: E402

_DANCE_MOTION = MOTION_DIR / "dance" / "dance1_subject2.npz"
_CTRL_DT = 0.02
_NUM_JOINTS = 29
_PROPRIO_DIM = 3 * _NUM_JOINTS + 3 + 3 + 6  # q, qd, prev action, local ang vel, gravity, relative rotation 6D
_GOAL_FRAME_DIM = 15 + 2 * _NUM_JOINTS


# ---------------------------------------------------------------------------
# Start-bin machinery (pure numpy, no simulation)
# ---------------------------------------------------------------------------


def test_build_start_bins_partitions_corpus() -> None:
    # Two clips (11 + 6 frames) with one-second bins at 4 frames per bin.
    frame_clip_end = np.concatenate([np.full(11, 10, np.int64), np.full(6, 16, np.int64)])
    bin_start, bin_end, frame_bin = build_start_bins(frame_clip_end, bin_frames=4)

    # Clip 1 (0..10): bins [0,3], [4,7], [8,10]; clip 2 (11..16): [11,14], [15,16].
    assert bin_start.tolist() == [0, 4, 8, 11, 15]
    assert bin_end.tolist() == [3, 7, 10, 14, 16]
    # Every frame is owned by exactly one bin and bins never cross clips.
    assert frame_bin.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    for start, end in zip(bin_start, bin_end):
        assert end >= start
        assert frame_clip_end[start] == frame_clip_end[end]


def _sampler(num_bins: int = 4, **overrides) -> SuperMimicStartSampler:
    return SuperMimicStartSampler(
        num_envs=2,
        bin_start=np.arange(num_bins, dtype=np.int64) * 50,
        bin_end=np.arange(num_bins, dtype=np.int64) * 50 + 49,
        gamma=0.99,
        return_ema=0.2,
        ctrl_dt=_CTRL_DT,
        max_step_reward=SUPERMIMIC_MAX_STEP_REWARD,
        **overrides,
    )


class _FakeTerm:
    """Minimal command stand-in exposing the sampler's kernel interface."""

    def __init__(self, num_envs: int, num_bins: int) -> None:
        self.reset_counter = np.zeros((num_envs, 1), dtype=np.int64)
        self.start_bin = np.zeros((num_envs, 1), dtype=np.int64)
        self.bin_cdf = np.cumsum(np.full(num_bins, 1.0 / num_bins, dtype=np.float32))
        self.bin_cdf[-1] = 1.0


def test_sampler_initial_mixture_is_uniform() -> None:
    sampler = _sampler(num_bins=8)
    term = _FakeTerm(2, 8)
    sampler.attach(term)
    # The construction bookend (first detected reset) only anchors the lanes.
    term.reset_counter[:] = 1
    metrics = sampler.tick(term, np.full((2,), 0.1, np.float32), np.zeros(2, bool))
    assert list(metrics)  # the bookend republishes the mixture once
    np.testing.assert_allclose(sampler.probabilities, 1.0 / 8, rtol=1e-6)
    np.testing.assert_allclose(term.bin_cdf, np.cumsum(sampler.probabilities), rtol=1e-6)
    assert term.bin_cdf[-1] == 1.0
    # The bookend is not a creditable episode: no bin has been seen yet.
    assert not sampler.bin_seen.any()


def test_sampler_credits_hard_and_bootstrapped_returns() -> None:
    sampler = _sampler(num_bins=4)
    term = _FakeTerm(2, 4)
    sampler.attach(term)
    # Anchor both lanes through one bookend step.
    term.reset_counter[:] = 1
    term.start_bin[:, 0] = [0, 1]
    sampler.tick(term, np.full((2,), 0.1, np.float32), np.zeros(2, bool))
    assert not sampler.bin_seen.any()

    # One 10-step episode per lane: lane 0 fails hard, lane 1 times out.
    reward = 0.05
    for _ in range(10):
        sampler.tick(term, np.full((2,), reward, np.float32), np.zeros(2, bool))
    term.reset_counter[:] = 2
    term.start_bin[:, 0] = [2, 3]
    hard = np.array([True, False])
    sampler.tick(term, np.full((2,), reward, np.float32), hard)

    gamma = 0.99
    undiscounted = 11 * reward
    discounted = reward * (1.0 - gamma**11) / (1.0 - gamma)
    bootstrap = gamma**11 * undiscounted / 11 / (1.0 - gamma)
    np.testing.assert_allclose(sampler.bin_returns[0], 0.2 * discounted, rtol=1e-6)
    np.testing.assert_allclose(sampler.bin_returns[1], 0.2 * (discounted + bootstrap), rtol=1e-6)
    assert sampler.bin_seen[0] and sampler.bin_seen[1]
    assert not sampler.bin_seen[2]

    # A failed bin earns more sampling mass than a successful one.
    assert sampler.probabilities[0] > sampler.probabilities[1]


def test_sampler_mixture_shares_sum_to_one() -> None:
    sampler = _sampler(num_bins=5)
    term = _FakeTerm(1, 5)
    sampler.attach(term)
    term.reset_counter[:] = 1
    term.start_bin[:, 0] = [2]
    rng = np.random.default_rng(0)
    for _ in range(20):
        sampler.tick(term, np.asarray(rng.uniform(0.0, 0.1, 1), np.float32), np.zeros(1, bool))
        if rng.random() < 0.3:
            term.reset_counter += 1
    np.testing.assert_allclose(float(np.sum(sampler.probabilities)), 1.0, rtol=1e-6)
    assert float(np.sum(term.bin_cdf)) >= 0.99
    assert term.bin_cdf[-1] == 1.0
    assert np.all(np.diff(term.bin_cdf) >= 0.0)


# ---------------------------------------------------------------------------
# Configuration contracts
# ---------------------------------------------------------------------------


def _cfg(motion_files: tuple[str, ...], **motion_overrides):
    import motrix_envs  # noqa: F401 - register built-in environments

    cfg = make_g1_supermimic_cfg(motion_files)
    for name, value in motion_overrides.items():
        setattr(cfg.commands.motion, name, value)
    return cfg


def test_supermimic_reward_table_matches_upstream(tmp_path) -> None:
    cfg = _cfg((str(tmp_path / "unused.npz"),))
    rewards = cfg.rewards
    assert rewards.motion_relative_body_position_error_exp.weight == 1.0
    assert rewards.motion_relative_body_position_error_exp.sigma == 0.3
    assert rewards.motion_relative_body_orientation_error_exp.weight == 1.0
    assert rewards.motion_relative_body_orientation_error_exp.sigma == 0.4
    assert rewards.motion_global_body_lin_vel.weight == 1.0
    assert rewards.motion_global_body_lin_vel.sigma == 1.0
    assert rewards.motion_global_body_ang_vel.weight == 1.0
    assert rewards.motion_global_body_ang_vel.sigma == 3.14
    assert rewards.motion_global_ref_position_error_exp.weight == 0.5
    assert rewards.motion_global_ref_position_error_exp.sigma == 0.3
    assert rewards.motion_global_ref_orientation_error_exp.weight == 0.5
    assert rewards.motion_global_ref_orientation_error_exp.sigma == 0.4
    assert rewards.action_rate_l2.weight == -0.1
    assert rewards.limits_dof_pos.weight == -2.0
    assert rewards.limits_dof_pos.soft_limit == 0.9
    # Four unit tracking kernels plus two half-weighted root kernels.
    positive = sum(
        getattr(rewards, name).weight
        for name in (
            "motion_relative_body_position_error_exp",
            "motion_relative_body_orientation_error_exp",
            "motion_global_body_lin_vel",
            "motion_global_body_ang_vel",
            "motion_global_ref_position_error_exp",
            "motion_global_ref_orientation_error_exp",
        )
    )
    assert positive == SUPERMIMIC_MAX_STEP_REWARD
    # Upstream height/tilt termination thresholds.
    assert cfg.terminations.bad_ref_z.threshold == 0.25
    assert cfg.terminations.bad_ref_ori.threshold == 0.8
    # Upstream reset noise magnitudes.
    assert cfg.sim_reset.body_pos.noise == (0.05, 0.05, 0.01)
    assert cfg.sim_reset.body_rot.noise == (0.1, 0.1, 0.2)
    assert cfg.sim_reset.body_lin_vel.noise == (0.5, 0.5, 0.2)
    assert cfg.sim_reset.body_rot_vel.noise == (0.52, 0.52, 0.78)
    assert cfg.sim_reset.body_dof_pos.noise == 0.1
    # Episode and control cadence: 500 steps at 50 Hz.
    assert cfg.ctrl_dt == _CTRL_DT
    assert cfg.max_episode_seconds == 10.0


def test_supermimic_policy_noise_matches_upstream(tmp_path) -> None:
    cfg = _cfg((str(tmp_path / "unused.npz"),))
    policy = cfg.observations.policy
    assert policy.dof_pos.noise.amplitude == 0.01
    assert policy.dof_vel.noise.amplitude == 0.5
    assert policy.base_ang_vel.noise.amplitude == 0.2
    assert policy.gravity.noise.amplitude == 0.05
    assert policy.motion_ref_ori_b.noise.amplitude == 0.05
    # The value group reuses the same terms noise-free.
    value = cfg.observations.value
    assert value.dof_pos.noise.amplitude == 0.0
    assert value.dof_vel.noise.amplitude == 0.0


def test_supermimic_for_play_switches_to_sequential_playback(tmp_path) -> None:
    cfg = _cfg((str(tmp_path / "unused.npz"),)).for_play()
    motion = cfg.commands.motion
    assert motion.supermimic_sampling is False
    assert motion.sequential_clips is True
    assert motion.hold_at_clip_end is False
    assert cfg.max_episode_seconds is None
    assert cfg.sim_reset.body_pos.noise_scale == 0.0
    assert cfg.sim_reset.body_dof_pos.noise_scale == 0.0


def test_supermimic_registry_entry_exists(tmp_path) -> None:
    import motrix_envs

    motrix_envs.register()
    assert registry.contains("g1-supermimic")


# ---------------------------------------------------------------------------
# Manager-kernel behavior (numba)
# ---------------------------------------------------------------------------


def _tiny_corpus_files(tmp_path, lengths=(6, 5)) -> tuple[str, ...]:
    """Slice per-frame fields of the bundled motion into short clips."""
    with np.load(_DANCE_MOTION, allow_pickle=False) as data:
        total = int(np.asarray(data["num_frames"]).reshape(-1)[0])
        fields = {key: data[key] for key in data.files}
    paths = []
    start = 0
    for index, length in enumerate(lengths):
        assert start + length <= total
        part = {"num_frames": np.int32(length)}
        for name, value in fields.items():
            if name == "num_frames":
                continue
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                part[name] = value[start : start + length]
            else:
                part[name] = value
        path = tmp_path / f"tiny{index}.npz"
        np.savez(path, **part)
        paths.append(str(path))
        start += length
    return tuple(paths)


@pytest.mark.numba
def test_numba_supermimic_observation_layout_and_goal(tmp_path) -> None:
    num_envs = 3
    cfg = _cfg(_tiny_corpus_files(tmp_path), num_goal_frames=3)
    env = SuperMimicManagerEnv(cfg, num_envs=num_envs, seed=5)
    state = env.init_state()

    goal_dim = 3 * _GOAL_FRAME_DIM
    assert env.observation_space.policy.shape == (_PROPRIO_DIM + goal_dim,)
    assert env.observation_space.value.shape[0] == _PROPRIO_DIM + goal_dim + 2 * _NUM_JOINTS + 3 + 3 * 14 + 3
    assert np.all(np.isfinite(state.obs.policy))

    motion = env.command_terms["motion"]
    assert isinstance(motion, SuperMimicMotionCommand)
    clip = motion.clip
    layout = env.manager_layout.observations["policy"]
    goal_slice = next(term for term in layout.terms if term.name == "goal").output_slice
    for lane in range(num_envs):
        step = int(motion.steps[lane, 0])
        end = int(clip.frame_clip_end[step])
        yaw_quat = _yaw_quaternion(clip.root_body_quat_w[step])
        for frame_id in range(3):
            future = min(step + frame_id + 1, end)
            expected = _goal_frame(clip, step, future, yaw_quat)
            got = state.obs.policy[lane][goal_slice][frame_id * _GOAL_FRAME_DIM : (frame_id + 1) * _GOAL_FRAME_DIM]
            np.testing.assert_allclose(got, expected, atol=2e-4, err_msg=f"lane {lane} frame {frame_id}")

    # The sampler is attached and the uniform mixture was published.
    assert env._supermimic_sampler is not None
    num_bins = motion.bin_cdf.shape[0]
    uniform_cdf = np.cumsum(np.full(num_bins, 1.0 / num_bins, dtype=np.float32))
    np.testing.assert_allclose(motion.bin_cdf, uniform_cdf, rtol=1e-6)

    # The dynamics randomization publishes a bounded simulation context
    # (friction scale in [0.3, 1.2], torso CoM offset within the per-axis
    # noise) and the flattened per-lane state buffer stays finite.
    context = motion.context
    assert context.shape == (num_envs, 4)
    assert np.all((context[:, 0] >= 0.3) & (context[:, 0] <= 1.2))
    assert np.all(np.abs(context[:, 1]) <= 0.025 + 1e-6)
    assert np.all(np.abs(context[:, 2:4]) <= 0.05 + 1e-6)
    assert motion.state.shape == (num_envs, 71)
    assert np.all(np.isfinite(motion.state))


def _yaw_quaternion(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    half_yaw = 0.5 * np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], np.float64)


def _goal_frame(clip, step: int, future: int, yaw_quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    heading = R.from_quat(yaw_quat)
    root_quat = lambda frame: R.from_quat(clip.root_body_quat_w[frame])  # noqa: E731
    rot6 = (heading.inv() * root_quat(future)).as_matrix()[:2, :].flatten()
    delta = clip.root_body_pos_w[future] - clip.root_body_pos_w[step]
    return np.concatenate(
        [
            heading.inv().apply(delta),
            rot6,
            heading.inv().apply(clip.root_body_lin_vel_w[future]),
            heading.inv().apply(clip.root_body_ang_vel_w[future]),
            clip.joint_pos[future],
            clip.joint_vel[future] * 0.1,
        ]
    )


@pytest.mark.numba
def test_numba_supermimic_clip_end_terminates_episode(tmp_path) -> None:
    # Two 6-frame clips: every lane reaches its clip end within six steps and
    # the episode ends there instead of resampling mid-episode.
    cfg = _cfg(_tiny_corpus_files(tmp_path, lengths=(6, 6)))
    env = SuperMimicManagerEnv(cfg, num_envs=2, seed=7)
    state = env.init_state()
    motion = env.command_terms["motion"]
    assert isinstance(motion, SuperMimicMotionCommand)
    counters_before = motion.reset_counter[:, 0].copy()

    action = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
    clip_end_fired = False
    for _ in range(8):
        state = env.step(action)
        if bool(np.any(state.metrics["motion_clip_end"])):
            clip_end_fired = True
            break
    assert clip_end_fired
    # The terminated lane rematerialized on a freshly drawn start frame
    # inside some clip, so the new step sits strictly before that clip's end.
    steps = motion.steps[:, 0].astype(np.int64)
    at_end = steps >= motion.clip.frame_clip_end[steps]
    assert not np.all(at_end)
    assert int(motion.reset_counter[:, 0].max()) > int(counters_before.max())

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG plugin contracts: registration, geometry, replay, and learner loop.

The learner is exercised through a fully third-party fake parsing head that
lives in this file and imports nothing from the plugin beyond the public
modules — the standing proof that a task can implement the ``ApgTask``
protocol without touching this package.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from motrix_rl_builtin.apg.async_train import UpdateCredit
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.contracts import require_apg_head, resolve_head
from motrix_rl_builtin.apg.framework import ApgProvider
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.networks import WorldModel
from motrix_rl_builtin.apg.replay import ReplayBuffer
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.framework import FastSacProvider

from motrix_env_core.torch.geometry import (
    INVARIANT_DIM,
    STATE_HEAD_DIM,
    TARGET_DIM,
    flat_state_dim,
    flatten_state,
    so3_exp,
    so3_log,
    unflatten_state,
)
from motrix_rl import frameworks

_NUM_JOINTS = 3
_NUM_FRAMES = 64
_OBS_DIM = 3 * _NUM_JOINTS + 12
_GOAL_FRAME_DIM = 15 + 2 * _NUM_JOINTS
_STATE_DIM = flat_state_dim(_NUM_JOINTS)
# The fake env's policy group: proprioceptive head plus the goal tail for
# the fake head's goal width (2 frames).
_OBS_FULL_DIM = _OBS_DIM + 2 * _GOAL_FRAME_DIM


def _make_cfg(**overrides) -> ApgCfg:
    cfg = ApgCfg()
    cfg.device = "cpu"
    agent = cfg.agent
    agent.buffer_steps = 32
    agent.utd = 0.5
    agent.horizon = 4
    agent.hist = 2
    agent.z_dim = 8
    agent.wm_hidden = [16, 16]
    agent.policy_hidden = [16, 16]
    agent.critic_hidden = [8, 8]
    agent.wm_window = 3
    agent.wm_batches = 1
    agent.wm_batch = 16
    agent.pi_batch = 8
    agent.lr = 1e-3
    agent.gamma = 0.99
    agent.lam = 0.95
    agent.action_noise = 0.1
    agent.critic_epochs = 1
    agent.critic_minibatches = 2
    agent.warmup_steps = 1
    agent.wm_only_iters = 0
    agent.lr_decay_iters = 100
    agent.amp = "none"
    agent.compile = False
    cfg.trainer.num_learning_iterations = 2
    for name, value in overrides.items():
        setattr(agent, name, value)
    return cfg


class _FakeHead:
    """A third-party parsing head over a synthetic corpus and fake env.

    Implements the whole ``ApgTask`` surface — dimensions, analytic torch
    heads, and the env-bound collection hooks — without importing anything
    from the plugin's internals.
    """

    num_joints = _NUM_JOINTS
    context_dim = 33
    obs_dim = _OBS_DIM
    goal_dim = 2 * _GOAL_FRAME_DIM
    max_step_reward = 5.0
    ctrl_dt = 0.02

    def __init__(self, env, device=torch.device("cpu")):
        self.device = torch.device(device)
        self.env = env
        self.clip_end = torch.full((_NUM_FRAMES,), _NUM_FRAMES - 1, dtype=torch.long, device=device)
        self.action_low = np.full(_NUM_JOINTS, -1.0, np.float32)
        self.action_high = np.full(_NUM_JOINTS, 1.0, np.float32)

    def to(self, device):
        self.device = torch.device(device)
        self.clip_end = self.clip_end.to(device)
        return self

    # -- torch heads ------------------------------------------------------

    def ref_state(self, frames):
        zeros = torch.zeros(frames.shape + (3,), device=frames.device)
        eye = torch.eye(3, device=frames.device).expand(frames.shape + (3, 3))
        q = torch.zeros(frames.shape + (_NUM_JOINTS,), device=frames.device)
        return (zeros + frames.unsqueeze(-1) * 0.01, eye, zeros, zeros, q, torch.zeros_like(q))

    def valid_imagination_start(self, frames, horizon):
        return frames + horizon <= self.clip_end[frames]

    def obs(self, state, previous_action, context, frames):
        return torch.cat(
            (state[4], state[5], state[3], state[3] * 0.0, previous_action, state[3] * 0.0, state[3] * 0.0),
            -1,
        )

    def add_noise(self, obs):
        return obs

    def goal(self, frames, context):
        return torch.zeros(frames.shape + (self.goal_dim,), device=frames.device)

    def reward(self, state, frames, action, previous_action, context):
        tracking = torch.ones(frames.shape)
        rate = ((action - previous_action) ** 2).sum(-1)
        return (tracking - 0.1 * rate) * 0.02

    def soft_termination(self, state, frames):
        return torch.zeros(frames.shape, device=frames.device)

    def critic_features(self, state, frames, normalized_context):
        return torch.cat((state[0], normalized_context), -1)

    def normalize_context(self, context):
        return context * 0.5

    # -- env-bound collection hooks ---------------------------------------

    def current_transition(self):
        motion = self.env.command_terms["motion"]
        return motion.state, motion.steps[:, 0], motion.context

    def endpoint_state(self):
        return self.env.command_terms["motion"].endpoint_state

    def split_observation(self, full):
        return full[..., : self.obs_dim], full[..., self.obs_dim :]

    def termination_counts(self, metrics):
        hard = np.asarray(metrics["bad_ref_z"], dtype=bool) | np.asarray(metrics["bad_ref_ori"], dtype=bool)
        return float(hard.sum())

    def sampling_state(self):
        return self.env.sampling_state()

    def load_sampling_state(self, state):
        self.env.load_sampling_state(state)

    def sampling_metrics(self):
        return {}


class _FakeMotion:
    def __init__(self, num_envs: int):
        rng = np.random.default_rng(0)
        self.state = rng.normal(size=(num_envs, _STATE_DIM)).astype(np.float32)
        self.endpoint_state = self.state.copy()
        self.steps = np.zeros((num_envs, 1), dtype=np.int64)
        self.context = rng.normal(size=(num_envs, 33)).astype(np.float32)


class _FakeEnv:
    """Deterministic linear-dynamics stand-in satisfying the env surface."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.command_terms = {"motion": _FakeMotion(num_envs)}
        self._rng = np.random.default_rng(1)
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)
        self._sampling_value = np.zeros(1, dtype=np.float64)

    def _draw_obs(self) -> np.ndarray:
        return self._rng.normal(size=(self.num_envs, _OBS_FULL_DIM)).astype(np.float32)

    def sampling_state(self):
        return {"value": self._sampling_value.copy()}

    def load_sampling_state(self, state):
        np.copyto(self._sampling_value, state["value"])

    @property
    def cfg(self):
        return SimpleNamespace(ctrl_dt=0.02)

    def init_state(self):
        return SimpleNamespace(
            obs=SimpleNamespace(policy=self._draw_obs()),
            reward=np.zeros(self.num_envs, np.float32),
            terminated=np.zeros(self.num_envs, bool),
        )

    def step(self, actions):
        motion = self.command_terms["motion"]
        delta = actions.astype(np.float32)
        motion.state[:, 13:16] += 0.1 * delta
        motion.state[:, 7:10] += delta
        motion.endpoint_state[:] = motion.state
        motion.steps = ((motion.steps + 1) % (_NUM_FRAMES - 8)).astype(np.int64)
        self._episode_steps += 1
        reward = 0.1 * np.ones(self.num_envs, np.float32)
        terminated = np.zeros(self.num_envs, bool)
        truncated = self._episode_steps >= 20
        self._episode_steps[truncated] = 0
        return SimpleNamespace(
            obs=SimpleNamespace(policy=self._draw_obs()),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "bad_ref_z": terminated.copy(),
                "bad_ref_ori": np.zeros(self.num_envs, bool),
            },
        )


def _make_learner(env, cfg=None, head=None) -> ApgLearner:
    return ApgLearner(
        env,
        (cfg or _make_cfg()).agent,
        head if head is not None else _FakeHead(env),
        device=torch.device("cpu"),
        seed=0,
    )


def test_plugin_registers_fastsac_and_apg():
    assert frameworks.supported_agents("motrix") == ("apg", "fastsac")
    assert frameworks.get_config_type("motrix", "fastsac") is FastSacCfg
    assert frameworks.get_config_type("motrix", "apg") is ApgCfg
    assert isinstance(frameworks.get_agent_provider("motrix", "apg", "torch"), ApgProvider)
    assert isinstance(frameworks.get_agent_provider("motrix", "fastsac", "torch"), FastSacProvider)


def test_apg_total_env_steps_counts_utd():
    cfg = _make_cfg()
    # Sync warmup covers the policy warmup, one WM window and one
    # observation history; each iteration then collects 1/utd steps with the
    # fractional part carried over.
    assert ApgProvider().total_env_steps(cfg, 10) == 7 * 10
    # The async collector earns update credit in whole blocks: strict UTD
    # over the warmup, rounded up to the block size.
    cfg.asynchronous = True
    assert ApgProvider().total_env_steps(cfg, 10) == 8 * 10


def test_resolve_head_imports_dotted_factory():
    factory = resolve_head("motrix_rl_builtin.apg.contracts:require_apg_env")
    assert callable(factory)
    with pytest.raises(ValueError, match="dotted path"):
        resolve_head("motrix_rl_builtin.apg.contracts")
    with pytest.raises(TypeError, match="callable factory"):
        resolve_head("motrix_rl_builtin.apg.contracts:not_a_factory")


def test_require_apg_head_rejects_heads_missing_protocol_members():
    require_apg_head(_FakeHead(_FakeEnv(4)))
    with pytest.raises(TypeError, match="missing ApgTask members"):
        require_apg_head(object())


def test_so3_exp_log_roundtrip():
    # Rotation vectors beyond pi wrap to another representative in the log,
    # so the roundtrip is only asserted inside the principal range.
    vectors = torch.randn(64, 3, dtype=torch.float64) * 0.7
    norm = vectors.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    vectors = vectors * torch.clamp((math.pi - 1e-4) / norm, max=1.0)
    roundtrip = so3_log(so3_exp(vectors))
    torch.testing.assert_close(roundtrip, vectors, rtol=1e-5, atol=1e-6)


def test_flatten_unflatten_state_roundtrip():
    rng = torch.randn(8, 3, dtype=torch.float64)
    eye = so3_exp(torch.randn(8, 3, dtype=torch.float64) * 0.5)
    state = (
        rng,
        eye,
        torch.randn(8, 3, dtype=torch.float64),
        torch.randn(8, 3, dtype=torch.float64),
        torch.randn(8, _NUM_JOINTS, dtype=torch.float64),
        torch.randn(8, _NUM_JOINTS, dtype=torch.float64),
    )
    flat = flatten_state(state)
    assert flat.shape == (8, _STATE_DIM)
    back = unflatten_state(flat)
    torch.testing.assert_close(back[0], state[0])
    torch.testing.assert_close(back[2], state[2])
    torch.testing.assert_close(back[4], state[4])


def test_state_layout_widths_are_the_head_contract():
    # The state layout is the interface contract between the world model and
    # the task heads; pin the widths independently so a layout edit cannot
    # desync the geometry constants, the networks, and the replay silently.
    assert STATE_HEAD_DIM == 13
    assert INVARIANT_DIM == 13
    assert TARGET_DIM == 12
    assert flat_state_dim(_NUM_JOINTS) == 13 + 2 * _NUM_JOINTS
    wm = WorldModel([8], _NUM_JOINTS, context_dim=4)
    assert wm.in_dim == 13 + 3 * _NUM_JOINTS + 4
    assert wm.out_dim == 12 + 2 * _NUM_JOINTS
    buffer = ReplayBuffer(4, 1, "cpu", num_joints=_NUM_JOINTS, context_dim=4)
    assert buffer.data["s"].shape[-1] == 13 + 2 * _NUM_JOINTS


def test_replay_windows_mask_after_done():
    buffer = ReplayBuffer(16, 2, "cpu", num_joints=_NUM_JOINTS, context_dim=4)
    env = _FakeEnv(2)
    for _ in range(16):
        transition = {
            "s": torch.tensor(env.command_terms["motion"].state),
            "s1": torch.tensor(env.command_terms["motion"].state),
            "a": torch.zeros(2, _NUM_JOINTS),
            "a_prev": torch.zeros(2, _NUM_JOINTS),
            "xi": torch.zeros(2, 4),
            "r": torch.zeros(2),
            "f": torch.tensor(env.command_terms["motion"].steps[:, 0]),
            "ep_step": torch.zeros(2, dtype=torch.int32),
            "done": torch.tensor(env.step(np.zeros((2, _NUM_JOINTS), np.float32)).truncated),
        }
        buffer.add(transition)
    assert buffer.size == 16
    window = buffer.sample_windows(32, 4)
    assert window["s"].shape == (32, 4, _STATE_DIM)
    assert window["mask"].shape == (32, 4)
    assert torch.all(window["mask"] >= 0.0)
    assert torch.all(window["mask"] <= 1.0)
    start, history = buffer.sample_starts(16, 3)
    assert start["s"].shape == (16, _STATE_DIM)
    assert history["s"].shape == (16, 3, _STATE_DIM)


def test_replay_keeps_terminal_endpoint_but_stops_at_next_episode():
    buffer = ReplayBuffer(3, 1, "cpu", num_joints=_NUM_JOINTS, context_dim=4)
    for index, done in enumerate((False, True, False)):
        row = {
            "s": torch.full((1, _STATE_DIM), float(index)),
            "s1": torch.full((1, _STATE_DIM), float(index + 1)),
            "a": torch.zeros(1, _NUM_JOINTS),
            "a_prev": torch.zeros(1, _NUM_JOINTS),
            "xi": torch.zeros(1, 4),
            "r": torch.zeros(1),
            "f": torch.zeros(1, dtype=torch.long),
            "ep_step": torch.zeros(1, dtype=torch.int32),
            "done": torch.tensor([done]),
        }
        buffer.add(row)
    window = buffer.sample_windows(8, 3)
    torch.testing.assert_close(window["mask"], torch.tensor([[1.0, 1.0, 0.0]]).expand(8, 3))
    torch.testing.assert_close(window["s1"][:, 1, 0], torch.full((8,), 2.0))


def test_learner_collects_and_updates_with_finite_losses():
    env = _FakeEnv(4)
    learner = _make_learner(env, _make_cfg())
    learner.observe(env.init_state().obs.policy)
    learner.collect(3)
    assert learner.buffer.size == 3
    wm_loss, wm_error = learner.wm_update()
    assert torch.isfinite(wm_loss) and torch.isfinite(wm_error)
    pi_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()
    assert torch.isfinite(pi_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(imagined_reward)
    assert torch.isfinite(survival)
    checkpoint = learner.state_dict()
    assert checkpoint["iteration"] == 0
    learner.iteration = 3
    env._sampling_value[0] = 0.7
    checkpoint = learner.state_dict()
    restored = _make_learner(_FakeEnv(4), _make_cfg())
    restored.load_state_dict(checkpoint)
    assert restored.iteration == 3
    assert restored.env._sampling_value[0] == pytest.approx(0.7)
    transferred = _make_learner(_FakeEnv(4), _make_cfg())
    transferred.initialize_from(checkpoint)
    assert transferred.iteration == transferred.sim_steps == transferred.buffer.size == 0
    assert transferred.env._sampling_value[0] == 0.0
    assert not transferred.opt_wm.state
    for key, value in learner.pi.state_dict().items():
        torch.testing.assert_close(transferred.pi.state_dict()[key], value)


def test_put_latest_counts_dropped_snapshots():
    import queue

    from motrix_rl_builtin.apg.async_train import put_latest

    snapshots = queue.Queue(maxsize=1)
    # First message goes straight in; a full queue drops the unpublished
    # snapshot to make room. The return feeds train/snapshots_replaced.
    assert put_latest(snapshots, b"v1") == 0
    assert put_latest(snapshots, b"v2") == 1
    assert put_latest(snapshots, b"v3") == 1
    assert snapshots.get_nowait() == b"v3"


def test_update_credit_excludes_warmup_and_tracks_fractional_utd():
    credit = UpdateCredit(utd=0.4, warmup=4)
    credit.receive(4)
    assert credit.available == 0
    credit.receive(3)
    assert credit.available == 1
    credit.consume()
    assert credit.available == 0
    credit.receive(5)
    assert credit.available == 2


def test_episode_length_counts_completed_transitions():
    env = _FakeEnv(4)
    learner = _make_learner(env)
    learner.observe(env.init_state().obs.policy)
    learner.collect(21)
    assert sum(learner.stats["done"]) == 4
    assert sum(learner.stats["ep_len_sum"]) / sum(learner.stats["done"]) == 20


def test_terminations_counted_only_through_head_classification():
    class ExtraDoneEnv(_FakeEnv):
        def step(self, actions):
            result = super().step(actions)
            # A done the head's own failure metrics do not explain.
            result.terminated[0] = True
            return result

    env = ExtraDoneEnv(4)
    learner = _make_learner(env)
    learner.observe(env.init_state().obs.policy)
    learner.collect(1)
    assert learner.stats["done"] == [1.0]
    # Only the head's reference-failure union counts as a task termination.
    assert learner.stats["term"] == [0.0]


def test_imagined_reward_is_normalized_by_step_reward():
    env = _FakeEnv(4)
    head = _FakeHead(env)
    learner = _make_learner(env, _make_cfg(), head)
    learner.observe(env.init_state().obs.policy)
    learner.collect(3)
    head.reward = lambda state, frames, action, previous, context: torch.full_like(frames, 0.1, dtype=torch.float32)
    _, _, imagined_reward, _, _ = learner.policy_update()
    assert float(imagined_reward) == pytest.approx(1.0)


def test_learner_rejects_non_contract_envs():
    with pytest.raises(TypeError, match="requires an environment exposing"):
        ApgLearner(
            object(),
            _make_cfg().agent,
            _FakeHead(_FakeEnv(4)),
            device=torch.device("cpu"),
        )


def test_learner_rejects_invalid_amp():
    env = _FakeEnv(4)
    cfg = _make_cfg(amp="fp16")
    with pytest.raises(ValueError, match="amp"):
        ApgLearner(env, cfg.agent, _FakeHead(env), device=torch.device("cpu"))


def test_update_runs_serial_wm_then_policy():
    env = _FakeEnv(4)
    learner = _make_learner(env)
    learner.observe(env.init_state().obs.policy)
    learner.collect(3)
    metrics = learner.update()
    assert learner.iteration == 1
    assert math.isfinite(metrics["wm_loss"])
    assert math.isfinite(metrics["wm_last_error"])
    # wm_only_iters defaults to 0 in _make_cfg, so policy metrics are present.
    for name in ("policy_loss", "value_loss", "imagined_reward", "policy_grad_norm", "imagined_survival"):
        assert math.isfinite(metrics[name])


def test_update_holds_policy_during_wm_only_phase():
    env = _FakeEnv(4)
    learner = _make_learner(env, _make_cfg(wm_only_iters=10))
    learner.observe(env.init_state().obs.policy)
    learner.collect(3)
    metrics = learner.update()
    assert learner.iteration == 1
    assert "policy_loss" not in metrics
    assert math.isfinite(metrics["wm_loss"])


def test_policy_update_fails_fast_without_valid_imagination_starts():
    class NoValidStartHead(_FakeHead):
        def valid_imagination_start(self, frames, horizon):
            return torch.zeros(frames.shape, dtype=torch.bool, device=frames.device)

    env = _FakeEnv(4)
    learner = _make_learner(env, _make_cfg(), NoValidStartHead(env))
    learner.observe(env.init_state().obs.policy)
    learner.collect(3)
    with pytest.raises(RuntimeError, match="no valid imagination start"):
        learner.policy_update()


def test_observe_rejects_wrong_observation_width():
    env = _FakeEnv(4)
    learner = _make_learner(env)
    with pytest.raises(ValueError, match="split the policy observation"):
        learner.observe(np.zeros((4, _OBS_DIM), np.float32))

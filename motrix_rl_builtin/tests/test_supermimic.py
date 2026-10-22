# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic plugin contracts: registration, geometry, replay, and learner loop."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.framework import FastSacProvider
from motrix_rl_builtin.supermimic.config import SuperMimicCfg
from motrix_rl_builtin.supermimic.framework import SuperMimicProvider
from motrix_rl_builtin.supermimic.geometry import (
    flatten_state,
    so3_exp,
    so3_log,
    unflatten_state,
)
from motrix_rl_builtin.supermimic.learner import SuperMimicLearner
from motrix_rl_builtin.supermimic.replay import ReplayBuffer

from motrix_rl import frameworks

_NUM_JOINTS = 3
_NUM_FRAMES = 64
_OBS_DIM = 3 * _NUM_JOINTS + 12
_GOAL_FRAME_DIM = 15 + 2 * _NUM_JOINTS
_STATE_DIM = 13 + 2 * _NUM_JOINTS


def _make_cfg(**overrides) -> SuperMimicCfg:
    cfg = SuperMimicCfg()
    cfg.device = "cpu"
    agent = cfg.agent
    agent.buffer_steps = 32
    agent.utd = 0.5
    agent.horizon = 4
    agent.goal_frames = 2
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


class _FakeMdp:
    """Minimal analytic MDP over a synthetic corpus."""

    num_joints = _NUM_JOINTS
    context_dim = 4
    obs_dim = _OBS_DIM
    goal_frame_dim = _GOAL_FRAME_DIM
    max_step_reward = 5.0
    ctrl_dt = 0.02

    def __init__(self, device):
        self.device = device
        self.clip_end = torch.full((_NUM_FRAMES,), _NUM_FRAMES - 1, dtype=torch.long, device=device)

    def to(self, device):
        self.device = torch.device(device)
        self.clip_end = self.clip_end.to(device)
        return self

    def unflatten_state(self, value):
        return tuple(value[..., i : i + 1] if i == 0 else value[..., i : i + 1] for i in range(0)) or (
            value[..., 0:3],
            torch.eye(3, device=value.device).expand(value.shape[:-1] + (3, 3)),
            value[..., 7:10],
            value[..., 10:13],
            value[..., 13:16],
            value[..., 16:19],
        )

    def flatten_state(self, state):
        identity = torch.zeros(state[0].shape[:-1] + (4,), device=state[0].device)
        identity[..., 3] = 1.0
        return torch.cat((*state[:2], identity, *state[2:]), -1)[..., :_STATE_DIM]

    def ref_state(self, frames):
        zeros = torch.zeros(frames.shape + (3,), device=frames.device)
        eye = torch.eye(3, device=frames.device).expand(frames.shape + (3, 3))
        q = torch.zeros(frames.shape + (_NUM_JOINTS,), device=frames.device)
        return (zeros + frames.unsqueeze(-1) * 0.01, eye, zeros, zeros, q, torch.zeros_like(q))

    def valid_imagination_start(self, frames, horizon):
        return frames + horizon <= self.clip_end[frames]

    def obs(self, state, previous_action, frames):
        return torch.cat(
            (state[4], state[5], state[3], state[3] * 0.0, previous_action, state[3] * 0.0, state[3] * 0.0),
            -1,
        )

    def add_noise(self, obs):
        return obs

    def goal(self, frames, num_frames):
        return torch.zeros(frames.shape + (num_frames * _GOAL_FRAME_DIM,), device=frames.device)

    def reward(self, state, frames, action, previous_action):
        tracking = torch.ones(frames.shape)
        rate = ((action - previous_action) ** 2).sum(-1)
        return (tracking - 0.1 * rate) * 0.02

    def soft_termination(self, state, frames):
        return torch.zeros(frames.shape, device=frames.device)

    def critic_features(self, state, frames, normalized_context):
        return torch.cat((state[0], normalized_context), -1)

    def normalize_context(self, context):
        return context * 0.5


class _FakeMotion:
    def __init__(self, num_envs: int, device):
        rng = np.random.default_rng(0)
        self.state = rng.normal(size=(num_envs, _STATE_DIM)).astype(np.float32)
        self.steps = np.zeros((num_envs, 1), dtype=np.int64)
        self.context = rng.normal(size=(num_envs, 4)).astype(np.float32)


class _FakeEnv:
    """Deterministic linear-dynamics stand-in satisfying the env contract."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self._mdp = _FakeMdp(torch.device("cpu"))
        self.command_terms = {"motion": _FakeMotion(num_envs, torch.device("cpu"))}
        self._rng = np.random.default_rng(1)
        self._episode_steps = np.zeros(num_envs, dtype=np.int64)

    @property
    def cfg(self):
        joints = [f"joint{i}" for i in range(_NUM_JOINTS)]
        return SimpleNamespace(
            commands=SimpleNamespace(motion=SimpleNamespace(joint_names=joints, tracked_body_names=[])),
            ctrl_dt=0.02,
        )

    def init_state(self):
        return SimpleNamespace(reward=np.zeros(self.num_envs, np.float32), terminated=np.zeros(self.num_envs, bool))

    def step(self, actions):
        motion = self.command_terms["motion"]
        delta = actions.astype(np.float32)
        motion.state[:, 13:16] += 0.1 * delta
        motion.state[:, 7:10] = delta
        motion.steps = ((motion.steps + 1) % (_NUM_FRAMES - 8)).astype(np.int64)
        self._episode_steps += 1
        reward = 0.1 * np.ones(self.num_envs, np.float32)
        terminated = np.zeros(self.num_envs, bool)
        truncated = self._episode_steps >= 20
        self._episode_steps[truncated] = 0
        return SimpleNamespace(reward=reward, terminated=terminated, truncated=truncated)


def test_plugin_registers_fastsac_and_supermimic():
    assert frameworks.supported_agents("motrix") == ("fastsac", "supermimic")
    assert frameworks.get_config_type("motrix", "fastsac") is FastSacCfg
    assert frameworks.get_config_type("motrix", "supermimic") is SuperMimicCfg
    assert isinstance(frameworks.get_agent_provider("motrix", "supermimic", "torch"), SuperMimicProvider)
    assert isinstance(frameworks.get_agent_provider("motrix", "fastsac", "torch"), FastSacProvider)


def test_supermimic_total_env_steps_counts_utd():
    cfg = _make_cfg()
    assert SuperMimicProvider().total_env_steps(cfg, 10) == 2 * 10 * 2  # iters * envs * round(1/utd)


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


def test_learner_collects_and_updates_with_finite_losses():
    env = _FakeEnv(4)
    cfg = _make_cfg()
    learner = SuperMimicLearner(
        env,
        cfg.agent,
        _FakeMdp(torch.device("cpu")),
        device=torch.device("cpu"),
        action_low=np.full(_NUM_JOINTS, -1.0, np.float32),
        action_high=np.full(_NUM_JOINTS, 1.0, np.float32),
        seed=0,
    )
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
    checkpoint = learner.state_dict()
    restored = SuperMimicLearner(
        _FakeEnv(4),
        cfg.agent,
        _FakeMdp(torch.device("cpu")),
        device=torch.device("cpu"),
        action_low=np.full(_NUM_JOINTS, -1.0, np.float32),
        action_high=np.full(_NUM_JOINTS, 1.0, np.float32),
    )
    restored.load_state_dict(checkpoint)
    assert restored.iteration == 3


def test_learner_rejects_non_contract_envs():
    with pytest.raises(TypeError, match="SuperMimic contract"):
        SuperMimicLearner(
            object(),
            _make_cfg().agent,
            _FakeMdp(torch.device("cpu")),
            device=torch.device("cpu"),
            action_low=np.zeros(1, np.float32),
            action_high=np.ones(1, np.float32),
        )


def test_learner_rejects_invalid_amp():
    env = _FakeEnv(4)
    cfg = _make_cfg(amp="fp16")
    with pytest.raises(ValueError, match="amp"):
        SuperMimicLearner(
            env,
            cfg.agent,
            _FakeMdp(torch.device("cpu")),
            device=torch.device("cpu"),
            action_low=np.zeros(_NUM_JOINTS, np.float32),
            action_high=np.ones(_NUM_JOINTS, np.float32),
        )

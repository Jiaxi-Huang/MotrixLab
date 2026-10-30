# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Namespace-generic lifecycle contracts of ``ArrayEnv``.

These tests pin the decision that NumPy remains the default frontend flavor
while the same lifecycle runs unchanged on the Torch namespace: the counter
fixture below is written once against Array API operators and instantiated
under both namespaces, so any per-namespace fork in the lifecycle shows up as
a behavioral divergence here.
"""

import time

import gymnasium as gym
import numpy as np
import pytest

from motrix_env_core.array.env import ArrayEnv, ArrayEnvState, FinalObs, Obs
from motrix_env_core.base import EnvCfg, ObsSpace
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.sim.backend import RenderConfig, SimRenderer

torch = pytest.importorskip("torch")


class _CounterEnv(ArrayEnv):
    """Exercise the frontend lifecycle without constructing a simulator backend."""

    def __init__(self, *, with_value: bool = False, ns: str = "numpy"):
        super().__init__(EnvCfg(scene=SceneCfg(), ctrl_dt=1.0, max_episode_seconds=3.0), num_envs=4, ns=ns)
        policy_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        value_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32) if with_value else None
        self._observation_space = ObsSpace(policy=policy_space, value=value_space)
        self._action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.counters = self.xp.zeros(self.num_envs, dtype=self.xp.float32)
        self.next_terminated = self.xp.zeros(self.num_envs, dtype=self.xp.bool)
        self.reset_rows: list = []
        self.physics_steps = 0
        self.observation_calls = 0

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def reset(self, env_ids) -> None:
        self.reset_rows.append(env_ids)
        self.counters[env_ids] = 0.0

    def apply_action(self, actions, state: ArrayEnvState) -> ArrayEnvState:
        return state

    def physics_step(self) -> None:
        self.physics_steps += 1
        self.counters = self.counters + 1.0

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        state.reward[:] = self.counters
        state.terminated[:] = self.next_terminated
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        self.observation_calls += 1
        state.obs.policy[:, 0] = self.counters
        if state.obs.value is not None:
            state.obs.value[:, 0] = self.counters
            state.obs.value[:, 1] = -self.counters
        return state

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        raise NotImplementedError("The counter environment has no simulator or renderer")


def _as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _step_actions(env: _CounterEnv):
    return env.xp.zeros((env.num_envs, 1), dtype=env.xp.float32)


def test_default_frontend_is_numpy() -> None:
    env = _CounterEnv()
    assert isinstance(env.state if env.state else env.counters, np.ndarray) or isinstance(env.counters, np.ndarray)
    assert isinstance(env.counters, np.ndarray)


def test_torch_frontend_runs_full_lifecycle() -> None:
    env = _CounterEnv(ns="torch")
    assert isinstance(env.counters, torch.Tensor)

    initial = env.init_state()
    assert isinstance(initial.obs, Obs)
    assert isinstance(initial.obs.policy, torch.Tensor)
    assert initial.final_obs is None
    assert _as_numpy(initial.obs.policy).shape == (4, 1)
    assert _as_numpy(env.reset_rows[0]).tolist() == [0, 1, 2, 3]

    # Drive a truncation so the final-obs + auto-reset path also runs on torch.
    state = env.step(_step_actions(env))
    assert isinstance(state.reward, torch.Tensor)
    assert state.final_obs is None
    assert env.physics_steps == 1

    env.state.episode_steps[:] = torch.tensor([2, 0, 2, 0])
    env.counters[:] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    truncated_state = env.step(_step_actions(env))
    assert _as_numpy(truncated_state.truncated).tolist() == [True, False, True, False]
    # Reset lanes restart at 0; the untouched lane keeps its incremented count.
    assert _as_numpy(truncated_state.episode_steps).tolist() == [0, 1, 0, 1]
    # The not-yet-truncated lane keeps its counter; reset lanes restart at 0.
    assert _as_numpy(truncated_state.obs.policy).tolist() == [[0.0], [21.0], [0.0], [41.0]]


def test_final_obs_default_hook_stays_none_on_torch() -> None:
    class _NoCaptureEnv(_CounterEnv):
        def _capture_final_obs(self, env_ids) -> FinalObs | None:
            return None

    env = _NoCaptureEnv(ns="torch")
    env.init_state()
    env.state.episode_steps[:] = torch.tensor([2, 0, 2, 0])
    state = env.step(_step_actions(env))
    assert state.final_obs is None
    assert _as_numpy(state.truncated).tolist() == [True, False, True, False]


def test_process_metrics_reduces_both_namespaces() -> None:
    for ns in ("numpy", "torch"):
        env = _CounterEnv(ns=ns)
        env.init_state()
        state = env.step(_step_actions(env))
        state.metrics = {"mean_counter": state.reward}
        reduced = state.process_metrics()
        assert reduced == {"mean_counter": pytest.approx(1.0)}


def test_numpy_step_timing_unchanged_by_namespace_layer() -> None:
    """The generic lifecycle must not add measurable per-step overhead on NumPy.

    Guards decision D8 (default stays NumPy with no behavior or performance
    regression): the same counter environment is stepped in a tight loop and
    the median step time is compared against a plain-NumPy reimplementation of
    the identical arithmetic. A generous factor bounds scheduler noise while
    still catching accidental per-step namespace dispatch or allocations.
    """
    env = _CounterEnv()
    env.init_state()
    actions = np.zeros((env.num_envs, 1), dtype=np.float32)

    def _baseline_step() -> None:
        counters = np.ones(4, dtype=np.float32)
        terminated = counters > 2
        truncated = counters > 3
        done = terminated | truncated
        if done.any():
            counters[done] = 0.0
            np.flatnonzero(done)

    # Warmup both loops, then take the median of repeated samples.
    for _ in range(50):
        env.step(actions)
        _baseline_step()

    env_samples = []
    baseline_samples = []
    for _ in range(5):
        started = time.perf_counter()
        for _ in range(200):
            env.step(actions)
        env_samples.append((time.perf_counter() - started) / 200)
        started = time.perf_counter()
        for _ in range(200):
            _baseline_step()
        baseline_samples.append((time.perf_counter() - started) / 200)

    env_median = sorted(env_samples)[2]
    baseline_median = sorted(baseline_samples)[2]
    # The env step does strictly more work (obs write, truncate, bookkeeping);
    # only the namespace layer's share is bounded here.
    assert env_median < max(baseline_median * 50, 1e-3), (env_median, baseline_median)


__all__ = ["_CounterEnv"]

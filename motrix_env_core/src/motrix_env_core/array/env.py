# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from dataclasses import dataclass
from typing import Generic, TypeVar

import gymnasium as gym
import numpy as np

from motrix_env_core.array_api import Array, array_namespace, is_array, resolve_namespace
from motrix_env_core.base import ABEnv, EnvCfg, ObsSpace
from motrix_env_core.perf import Perf, perf_root
from motrix_env_core.sim.backend import (
    RenderConfig,
    SimRenderer,
)

# The concrete config type an ArrayEnv subclass consumes. Subclasses parametrize
# via ``ArrayEnv[MyCfg]`` so ``self.cfg`` is typed as ``MyCfg`` (IDE hint +
# pyright narrowing). Unparametrized subclasses fall back to ``EnvCfg``
# (back-compat).
EnvCfgType = TypeVar("EnvCfgType", bound=EnvCfg)


@dataclass
class Obs:
    """Observation pair with an optional value branch.

    Arrays belong to the frontend namespace the environment was constructed
    with (NumPy by default, Torch when ``ns="torch"``).
    """

    policy: Array
    value: Array | None = None

    @property
    def value_or_policy(self) -> Array:
        return self.policy if self.value is None else self.value


@dataclass
class FinalObs:
    """Owned pre-reset observations of the current transition's truncated rows.

    Rows in ``obs`` align with ``env_ids``. Arrays remain valid across steps;
    consumers must treat them as read-only. ``obs.value`` stays optional.
    """

    env_ids: Array
    obs: Obs

    def __len__(self) -> int:
        return len(self.env_ids)


@dataclass
class ArrayEnvState:
    obs: Obs
    reward: Array
    terminated: Array
    truncated: Array
    episode_steps: Array
    # Per-term reward breakdown for the latest transition: term name to a
    # ``(num_envs,)`` array. Freshly written every transition; empty when the
    # environment does not decompose its reward.
    reward_terms: dict[str, Array] = dataclasses.field(default_factory=dict)
    # Live diagnostics view: per-environment quantities stay unreduced as
    # ``(num_envs,)`` arrays and are views into manager buffers that kernels
    # overwrite every step; batch-level gauges are scalars. Values always
    # reflect the latest step: retain values across steps through
    # :meth:`process_metrics`, which returns a fresh dict of reduced scalars.
    metrics: dict[str, float | Array] = dataclasses.field(default_factory=dict)
    # None when no row truncated or the implementation does not capture finals.
    final_obs: FinalObs | None = None

    @property
    def done(self) -> Array:
        """
        Check if the environment is done.
        """
        return self.terminated | self.truncated

    def replace(self, **updates) -> "ArrayEnvState":
        return dataclasses.replace(self, **updates)

    def process_metrics(self) -> dict[str, float]:
        """Reduce per-environment metrics into batch-level scalar floats.

        Per-env arrays are reduced with their mean (bool arrays become trigger
        rates); scalars pass through. The result is a fresh, small dict that is
        safe to retain across steps, unlike the raw ``metrics`` arrays.
        """
        return {
            name: float(array_namespace(value).mean(value)) if is_array(value) else float(value)
            for name, value in self.metrics.items()
        }

    def validate(self):
        num_envs = self.obs.policy.shape[0]
        assert self.obs.policy.shape[0] == num_envs, self.obs.policy.shape
        if self.obs.value is not None:
            assert self.obs.value.shape[0] == num_envs, self.obs.value.shape
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.episode_steps.shape == (num_envs,), self.episode_steps.shape


def _as_obs(obs: Obs | Array) -> Obs:
    if isinstance(obs, Obs):
        return obs
    return Obs(policy=obs)


def _as_obs_space(space: gym.Space | ObsSpace) -> ObsSpace:
    if isinstance(space, ObsSpace):
        return space
    assert isinstance(space, gym.spaces.Box)
    return ObsSpace(policy=space)


class ArrayEnv(ABEnv, Generic[EnvCfgType]):
    """Namespace-generic frontend environment lifecycle.

    ``ArrayEnv`` owns the step / auto-reset / truncation lifecycle shared by
    NumPy- and Torch-frontend environments. It is deliberately free of
    simulator-state concepts: subclasses address simulator rows themselves
    inside the reset hooks, so the frontend only ever speaks in environment
    indices.

    The lifecycle speaks the Array API: subclass implementations read arrays
    from ``self.xp`` and use operators/indexing so the same code runs on both
    namespaces. Namespace-specific functions (``zeros``, ``nonzero``, dtypes)
    are bound once in ``__init__`` so the per-step path never resolves the
    namespace again.
    """

    _cfg: EnvCfgType
    _state: ArrayEnvState = None
    _render_spacing: float

    def __init__(self, cfg: EnvCfgType, num_envs: int = 1, ns: str = "numpy"):
        self._cfg = cfg
        self._num_envs = num_envs
        self.perf = Perf()
        if cfg.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        self._render_spacing = cfg.render_spacing
        self.xp = resolve_namespace(ns)
        # Pre-bind the namespace functions the per-step lifecycle uses so the
        # hot path holds direct function references instead of attribute
        # lookups on the namespace module.
        self._zeros = self.xp.zeros
        self._ones = self.xp.ones
        # Array API standard: nonzero returns a tuple of index arrays; the
        # done mask is 1-D so its indices are the first element.
        self._nonzero = self.xp.nonzero
        self._bool_dtype = self.xp.bool
        self._f32_dtype = self.xp.float32
        # int64 rather than uint64: torch implements index assignment for
        # signed counters only, and episode lengths stay far below 2**63.
        self._i64_dtype = self.xp.int64

    @property
    def state(self) -> ArrayEnvState:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfgType:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def render_spacing(self) -> float:
        """
        Get the render spacing, with which the multi-envs will be rendered seperately in grid
        """
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> ArrayEnvState:
        """
        Create a new environment state
        """
        obs = self._zeros_obs(self.observation_space)
        reward = self._zeros((self._num_envs,), dtype=self._f32_dtype)
        # All lanes start terminated so the first reset pass covers the batch.
        terminated = self._ones((self._num_envs,), dtype=self._bool_dtype)
        truncated = self._zeros((self._num_envs,), dtype=self._bool_dtype)
        episode_steps = self._zeros((self._num_envs,), dtype=self._i64_dtype)
        self._state = self._new_state(obs, reward, terminated, truncated, episode_steps)
        self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        self._state.validate()
        return self._state

    def _new_state(
        self,
        obs: Obs,
        reward: Array,
        terminated: Array,
        truncated: Array,
        episode_steps: Array,
    ) -> ArrayEnvState:
        """Assemble the environment state; subclasses add simulator-owned fields."""
        return ArrayEnvState(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_steps=episode_steps,
        )

    @property
    def policy_observation_space(self) -> gym.spaces.Box:
        return _as_obs_space(self.observation_space).policy

    @property
    def value_observation_space(self) -> gym.spaces.Box:
        return _as_obs_space(self.observation_space).value_or_policy

    @property
    def has_value_observation(self) -> bool:
        return _as_obs_space(self.observation_space).value is not None

    def _zeros_box(self, space: gym.spaces.Box) -> Array:
        # Gym spaces carry NumPy dtypes; map through the namespace so Torch
        # allocators receive a torch dtype.
        dtype = self._map_dtype(space.dtype)
        return self._zeros((self._num_envs, *space.shape), dtype=dtype)

    def _map_dtype(self, dtype) -> "Array":
        """Translate a NumPy/gym dtype into the frontend namespace's dtype."""
        return getattr(self.xp, np.dtype(dtype).name, dtype)

    def _zeros_obs(self, space: gym.Space | ObsSpace) -> Obs:
        obs_space = _as_obs_space(space)
        value = None if obs_space.value is None else self._zeros_box(obs_space.value)
        return Obs(policy=self._zeros_box(obs_space.policy), value=value)

    def _assign_obs(self, dst: Obs, mask: Array, src: Obs | Array):
        src = _as_obs(src)
        dst.policy[mask] = src.policy
        if dst.value is not None:
            assert src.value is not None
            dst.value[mask] = src.value
        else:
            assert src.value is None

    def _reset_done_envs(self) -> None:
        """
        Reset the environments that are done
        """
        state = self._state
        with self.perf.scope("done_mask"):
            done = state.done
            assert done.shape == (self._num_envs,)
            has_done = bool(done.any())
        if not has_done:
            return

        with self.perf.scope("select_done"):
            # Index assignment (not masked_fill): torch does not implement
            # masked assignment for signed step counters.
            env_ids = self._nonzero(done)[0]
            state.episode_steps[env_ids] = 0
        with self.perf.scope("reset_envs"):
            self.reset(env_ids)

    @abc.abstractmethod
    def reset(self, env_ids: Array) -> None:
        """Write reset rows for the selected environments.

        Subclasses address their simulator's rows themselves. Observation
        generation is deferred until :meth:`compute_observation`, after reset
        writes have completed.
        """

    def _capture_final_obs(self, env_ids: Array) -> FinalObs | None:
        """Capture truncated rows before reset; default implementation is unsupported.

        Called after transition and truncation updates, before any reset write.
        Implementations must leave normal observations and task state unchanged
        and return owned row arrays, without advancing observation history.
        """
        return None

    def _max_episode_steps(self) -> int | None:
        return self._cfg.max_episode_steps

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        max_episode_steps = self._max_episode_steps()
        if not max_episode_steps:
            return
        self._state.truncated = self._state.episode_steps >= max_episode_steps

    @abc.abstractmethod
    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        """Create the backend-owned renderer bound to this environment."""

    @abc.abstractmethod
    def apply_action(self, actions: Array, state: ArrayEnvState) -> ArrayEnvState:
        """
        Apply the action to the environment

        Args:
            actions: The actions to apply
            state: The environment state to apply the actions.
        """

    @abc.abstractmethod
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        """
        Update the environment state after physics step

        This is the only full simulator-data refresh point of a step: concrete
        environments execute their whole read program here before deriving
        reward, termination, and metrics. It must not write ``state.obs`` —
        observations belong to :meth:`compute_observation`.

        Args:
            state: The environment state to update
        """

    @abc.abstractmethod
    def physics_step(self):
        """Advance the simulator by the configured substeps."""

    def _prev_physics_step(self):
        if self._state.final_obs is not None:
            self._state = self._state.replace(final_obs=None)
        state = self._state
        # Broadcast assignment instead of ``fill``/``fill_``: NumPy and Torch
        # agree on ``arr[:] = scalar``.
        state.reward[:] = 0.0
        state.terminated[:] = False
        state.truncated[:] = False

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        """Compute observations for all environments after reset.

        Called once after the first reset of :meth:`init_state` and once per
        :meth:`step` after episode bookkeeping and auto-reset. Implementations
        must purely construct the observation from the simulator data already
        refreshed by :meth:`compute_transition` (whole batch) and reset
        (selected rows) without executing further reads. The default preserves
        observations already present in ``state``.
        """
        return state

    @perf_root("step")
    def step(self, actions: Array) -> ArrayEnvState:
        """Advance one control step.

        Timing: ``apply_action`` -> ``physics_step`` -> ``compute_transition``
        (the only full simulator-data refresh) -> episode bookkeeping ->
        optional pre-reset ``final_obs`` capture for truncated rows ->
        ``reset(env_ids)`` for done rows (may refresh selected rows) ->
        ``compute_observation`` (whole batch, pure construction).
        """
        if self._state is None:
            with self.perf.scope("init_state"):
                self.init_state()

        self._prev_physics_step()
        with self.perf.scope("apply_action"):
            self._state = self.apply_action(actions, self._state)
            assert self._state is not None, "apply_action must return a valid environment state"
        with self.perf.scope("physics"):
            self.physics_step()
        with self.perf.scope("transition"):
            self._state = self.compute_transition(self._state)
        self._state.episode_steps += 1
        self._update_truncate()
        trunc_ids = self._nonzero(self._state.truncated)[0]
        if len(trunc_ids):
            with self.perf.scope("final_obs"):
                self._state = self._state.replace(final_obs=self._capture_final_obs(trunc_ids))
        with self.perf.scope("reset"):
            self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        return self._state


__all__ = ["ArrayEnv", "ArrayEnvState", "EnvCfgType", "FinalObs", "Obs"]

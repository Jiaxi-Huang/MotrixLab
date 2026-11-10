# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dual-path modeling derivation contracts (P2.3).

The fixtures build a physics-less manager env whose command term carries one
channel per exogenous rule. The dual-path tests capture the exact evaluate
inputs of a real step and replay them through the derived modeling program:
the Phase 2 acceptance line is that the same terms produce the same numbers
through both context bindings.
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.config import configclass  # noqa: E402
from motrix_env_core.config.scene import SceneCfg  # noqa: E402
from motrix_env_core.manager import (  # noqa: E402
    ActionCfg,
    ActionState,
    ActionTerm,
    CommandCfg,
    CommandTerm,
    ExogenousKind,
    ExogenousRule,
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerCommandsCfg,
    ManagerContext,
    ManagerEnv,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    ModelingDerivationError,
    ObservationTermCfg,
    ObsTerm,
    RewardTerm,
    RewardTermCfg,
    TerminationTerm,
    TerminationTermCfg,
    TruncationTerm,
    TruncationTermCfg,
    metric,
)

# ---------------------------------------------------------------------------
# Fixture: one action term and one command term with a channel per rule.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TestActionState(ActionState):
    source: torch.Tensor


class _TestActionTerm(ActionTerm):
    def _process(self, actions: torch.Tensor) -> None:
        self.state.source.copy_(actions)

    def _reset(self, env_ids: torch.Tensor) -> None:
        del env_ids


def _test_action_space(_: ManagerEnv, __: object) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)


@configclass(kw_only=True)
class _TestActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] = ()

    def __call__(self, env: ManagerEnv, actuator_indices: object) -> ActionTerm:
        return _TestActionTerm(
            _test_action_space(env, actuator_indices),
            _TestActionState(
                torch.from_numpy(np.zeros((env.num_envs, 2, 1), dtype=np.float32)),
                torch.from_numpy(np.zeros(1, dtype=np.int64)),
                torch.from_numpy(np.zeros((env.num_envs, 1), dtype=np.float32)),
            ),
        )


@dataclass(frozen=True, slots=True)
class _PhaseCommand(CommandTerm):
    command: torch.Tensor  # increment channel (phase-like)
    goal: torch.Tensor  # copy_action channel (velocity-command-like)
    bias: torch.Tensor  # constant channel (reset-resampled)
    derived: torch.Tensor  # recomputed by update from the channels
    table: torch.Tensor  # shared episode-independent constant
    gauge: torch.Tensor = metric()

    def update(self: "_PhaseCommand", ctx: ManagerContext) -> None:
        self.derived[:, 0] = self.command[:, 0] + self.goal[:, 0]
        self.gauge[:] = self.derived[:, 0]

    def advance(self: "_PhaseCommand", ctx: ManagerContext) -> None:
        # The real counterpart of the annotated rules: command increments by
        # dt, goal tracks the current action, bias holds, the gauge folds.
        self.command[:, 0] += ctx.dt
        self.goal[:, 0] = ctx.actions["test"].source[:, 0]
        self.gauge.add_(1.0)

    def reset(self: "_PhaseCommand", ctx: ManagerContext) -> None:
        self.command[ctx.reset_ids, 0] = -1.0
        self.bias[ctx.reset_ids, 0] = 0.0


@configclass(kw_only=True)
class _PhaseCommandCfg(CommandCfg):
    def __call__(self, env: ManagerEnv) -> _PhaseCommand:
        shape = (env.num_envs, 1)
        return _PhaseCommand(
            command=torch.zeros(shape, dtype=torch.float32),
            goal=torch.zeros(shape, dtype=torch.float32),
            bias=torch.ones(shape, dtype=torch.float32),
            derived=torch.zeros(shape, dtype=torch.float32),
            table=torch.arange(4, dtype=torch.float32),
        )


def _phase_reward(ctx: ManagerContext) -> torch.Tensor:
    return ctx.actions["test"].source[:, 0] + ctx.commands["phase"].derived[:, 0]


@configclass(kw_only=True)
class _PhaseRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(_phase_reward)


def _threshold_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    return ctx.actions["test"].source[:, 0] >= threshold


@configclass(kw_only=True)
class _ThresholdTerminationCfg(TerminationTermCfg):
    threshold: float

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(_threshold_termination, float(self.threshold))


def _threshold_truncation(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    return ctx.actions["test"].source[:, 0] >= threshold


@configclass(kw_only=True)
class _ThresholdTruncationCfg(TruncationTermCfg):
    threshold: float

    def __call__(self, env: ManagerEnv) -> TruncationTerm:
        del env
        return TruncationTerm(_threshold_truncation, float(self.threshold))


def _phase_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
    out[:, 0] = ctx.actions["test"].source[:, 0]
    out[:, 1] = ctx.commands["phase"].command[:, 0]


@configclass(kw_only=True)
class _PhaseObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(2, _phase_observation)


def _noisy_observation(ctx: ManagerContext, out: torch.Tensor) -> None:
    out[:, 0] = ctx.commands["phase"].command[:, 0] + 0.5 * ctx.rand.next_uniform(1)[:, 0]


@configclass(kw_only=True)
class _NoisyObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(1, _noisy_observation)


@configclass
class _ModelingCommandsCfg(ManagerCommandsCfg):
    phase: _PhaseCommandCfg = _PhaseCommandCfg()


@configclass
class _ModelingActionsCfg(ManagerActionsCfg):
    test: _TestActionCfg = _TestActionCfg()


@configclass
class _ModelingRewardsCfg(ManagerRewardsCfg):
    track: _PhaseRewardCfg = _PhaseRewardCfg(weight=2.0)


@configclass
class _ModelingTerminationsCfg(ManagerTerminationsCfg):
    limit: _ThresholdTerminationCfg = _ThresholdTerminationCfg(threshold=0.9)


@configclass
class _ModelingPolicyObsCfg(ManagerObservationGroupCfg):
    phase: _PhaseObservationCfg = _PhaseObservationCfg()


@configclass
class _ModelingObservationsCfg(ManagerObservationsCfg):
    policy: _ModelingPolicyObsCfg = _ModelingPolicyObsCfg()


# The increment step must equal the env's ctrl_dt (the real advance adds
# ctx.dt); the default EnvCfg ctrl_dt is 0.01.
_INCREMENT_STEP = 0.01


@configclass
class _ModelingEnvCfg(ManagerBasedEnvCfg):
    scene: SceneCfg = SceneCfg()
    commands: _ModelingCommandsCfg = _ModelingCommandsCfg(
        phase=_PhaseCommandCfg(
            exogenous={
                "command": ExogenousRule(kind=ExogenousKind.INCREMENT, step=_INCREMENT_STEP),
                "goal": ExogenousRule(kind=ExogenousKind.COPY_ACTION, action_term="test"),
                "bias": ExogenousRule(kind=ExogenousKind.CONSTANT),
            },
        ),
    )
    actions: _ModelingActionsCfg = _ModelingActionsCfg()
    observations: _ModelingObservationsCfg = _ModelingObservationsCfg()
    rewards: _ModelingRewardsCfg = _ModelingRewardsCfg()
    terminations: _ModelingTerminationsCfg = _ModelingTerminationsCfg()
    truncations: dict[str, TruncationTermCfg] = dataclass_field(default_factory=dict)


def _make_env(num_envs: int = 2) -> ManagerEnv:
    env = ManagerEnv(_ModelingEnvCfg(truncations={"time_style": _ThresholdTruncationCfg(threshold=0.8)}), num_envs)
    env.physics_step = lambda: None
    env.init_state()
    return env


def _capture_evaluate_inputs(env: ManagerEnv) -> dict[str, torch.Tensor]:
    """Record the sim arena exactly as the next real evaluate consumes it.

    The wrapper delegates to the real evaluate kernel, so the step proceeds
    normally; the env instance is test-local and discarded afterwards.
    """
    captured: dict[str, torch.Tensor] = {}
    original = env._execute_evaluate_kernel

    def capture() -> None:
        captured.clear()
        captured.update({key: env.sim_data.view(key).clone() for key in env.sim_data.keys})
        original()

    env._execute_evaluate_kernel = capture  # type: ignore[method-assign]
    return captured


# ---------------------------------------------------------------------------
# Derivation surface
# ---------------------------------------------------------------------------


def test_modeling_view_derives_fields_exogenous_and_caches() -> None:
    env = _make_env(num_envs=3)
    view = env.modeling_view()

    assert set(view.fields) == set(env.sim_data.keys)
    assert set(view.exogenous) == {"phase.command", "phase.goal", "phase.bias"}
    assert view.exogenous["phase.command"].kind is ExogenousKind.INCREMENT
    assert view.exogenous["phase.goal"].kind is ExogenousKind.COPY_ACTION
    assert view.exogenous["phase.bias"].kind is ExogenousKind.CONSTANT
    # The accessor derives once; later calls return the same view.
    assert env.modeling_view() is view
    # Same function objects as the real path, not re-implementations.
    assert view.action_process["test"] is type(env.action_terms["test"]).process


def test_modeling_view_requires_resolved_program() -> None:
    env = ManagerEnv(_ModelingEnvCfg(truncations={}), num_envs=2)
    with pytest.raises(RuntimeError, match="init_state"):
        env.modeling_view()


def test_shadow_rules_copy_rows_share_constants_zero_metrics() -> None:
    env = _make_env(num_envs=3)
    view = env.modeling_view()
    real = env.command_terms["phase"]
    shadow = view.program._host.commands["phase"]
    assert isinstance(shadow, _PhaseCommand)

    # Per-environment state rows clone into independent storage.
    assert shadow.command is not real.command
    assert shadow.goal is not real.goal
    assert shadow.derived is not real.derived
    assert shadow.bias is not real.bias
    # Shared episode-independent constants are the same objects, read-only.
    assert shadow.table is real.table
    # Declared metric gauges become fresh writable sinks: independent storage
    # (the dry run already exercised writes through it), while the real gauge
    # stays untouched.
    assert shadow.gauge is not real.gauge
    torch.testing.assert_close(real.gauge, torch.zeros_like(real.gauge))
    # Action working memory copies entirely — including the shared cursor,
    # which imagination must not advance for the collector.
    real_action = env.action_terms["test"].state
    shadow_action = view.program._host.actions["test"].term.state
    assert shadow_action.action_queue is not real_action.action_queue
    assert shadow_action.source is not real_action.source
    assert shadow_action.action_ptr is not real_action.action_ptr


# ---------------------------------------------------------------------------
# Dual-path numerical consistency (Phase 2 acceptance line)
# ---------------------------------------------------------------------------


def test_dual_path_outputs_match_real_kernel_on_alive_step() -> None:
    env = _make_env(num_envs=2)
    program = env.modeling_view().program
    env.command_terms["phase"].command[:] = 2.0

    for action_value in (0.25, 0.5):
        action = torch.full((2, 1), action_value, dtype=torch.float32)
        program.sync_from_env()
        captured = _capture_evaluate_inputs(env)
        state = env.step(action)

        reward, terminated, truncated = program.evaluate(captured, action)
        np.testing.assert_allclose(reward.numpy(), state.reward.numpy())
        np.testing.assert_array_equal(terminated.numpy(), state.terminated.numpy())
        np.testing.assert_array_equal(truncated.numpy(), state.truncated.numpy())

        # Observations compare only on steps where no lane is done (auto-reset
        # rewrites the real side); imagination observes after the exogenous
        # advance, mirroring the real observe's post-advance view.
        program.advance_exogenous(action)
        obs = program.observe(captured)
        np.testing.assert_allclose(obs["policy"].numpy(), state.obs.policy.numpy())


def test_dual_path_outputs_match_real_kernel_on_done_rows() -> None:
    env = _make_env(num_envs=2)
    program = env.modeling_view().program

    program.sync_from_env()
    action = torch.tensor([[0.25], [0.95]], dtype=torch.float32)
    captured = _capture_evaluate_inputs(env)
    state = env.step(action)

    # Lane 1 terminates (>= 0.9) and truncates (>= 0.8): the evaluate outputs
    # are pre-reset truth on both paths, so they still compare.
    np.testing.assert_array_equal(state.terminated.numpy(), [False, True])
    np.testing.assert_array_equal(state.truncated.numpy(), [False, True])
    reward, terminated, truncated = program.evaluate(captured, action)
    np.testing.assert_allclose(reward.numpy(), state.reward.numpy())
    np.testing.assert_array_equal(terminated.numpy(), state.terminated.numpy())
    np.testing.assert_array_equal(truncated.numpy(), state.truncated.numpy())


def test_advance_exogenous_matches_real_advance() -> None:
    env = _make_env(num_envs=2)
    program = env.modeling_view().program
    real = env.command_terms["phase"]
    shadow = program._host.commands["phase"]
    program.sync_from_env()

    for step in range(5):
        action_value = 0.1 * (step + 1)
        action = torch.full((2, 1), action_value, dtype=torch.float32)
        env.step(action)
        program.advance_exogenous(action)
        # The increment and copy_action channels track the real advance
        # exactly; the constant channel holds its episode value on both
        # sides (the init reset resampled it to zero).
        np.testing.assert_allclose(shadow.command.numpy(), real.command.numpy())
        np.testing.assert_allclose(shadow.goal.numpy(), real.goal.numpy())
        np.testing.assert_allclose(shadow.bias.numpy(), real.bias.numpy())


def test_shadow_isolation_after_imagination() -> None:
    env = _make_env(num_envs=2)
    program = env.modeling_view().program
    program.sync_from_env()
    real = env.command_terms["phase"]
    saved = {name: getattr(real, name).clone() for name in ("command", "goal", "bias", "derived", "gauge")}
    saved_table = real.table.clone()
    saved_queue = env.action_terms["test"].state.action_queue.clone()
    saved_ptr = env.action_terms["test"].state.action_ptr.clone()

    for step in range(3):
        action = torch.full((2, 1), 0.3 + 0.1 * step, dtype=torch.float32)
        program.evaluate({}, action)
        program.advance_exogenous(action)
        program.observe({})

    # The real resident state is bit-identical after imagination rounds.
    for name, value in saved.items():
        torch.testing.assert_close(getattr(real, name), value)
    torch.testing.assert_close(real.table, saved_table)
    torch.testing.assert_close(env.action_terms["test"].state.action_queue, saved_queue)
    torch.testing.assert_close(env.action_terms["test"].state.action_ptr, saved_ptr)
    # Imagination actually progressed its own shadow history (the cursor
    # wraps around the queue width W == 2).
    shadow_action = program._host.actions["test"].term.state
    assert int(shadow_action.action_ptr[0]) == (int(saved_ptr[0]) + 3) % 2


# ---------------------------------------------------------------------------
# Determinism, binding, and enforcement
# ---------------------------------------------------------------------------


def test_observe_noise_is_deterministic() -> None:
    cfg = _ModelingEnvCfg(truncations={})
    cfg.observations = {"policy": {"noisy": _NoisyObservationCfg()}}
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    env.init_state()
    program = env.modeling_view().program

    first = program.observe({})
    second = program.observe({})
    np.testing.assert_allclose(first["policy"].numpy(), second["policy"].numpy())
    # Zero-fill semantics: the noise contribution vanishes, so the column is
    # exactly the shadow command value the term anchors on.
    shadow_command = program._host.commands["phase"].command
    np.testing.assert_allclose(first["policy"][:, 0].numpy(), shadow_command[:, 0].numpy())


def test_evaluate_rejects_mismatched_state_fields() -> None:
    env = _make_env(num_envs=2)
    program = env.modeling_view().program
    action = torch.zeros((2, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="undeclared fields"):
        program.evaluate({"ghost": torch.zeros((2, 1))}, action)


def test_dry_run_error_names_term_and_field() -> None:
    def _ghost_termination(ctx: ManagerContext) -> torch.Tensor:
        return ctx.sim["ghost"] >= 0.0

    @configclass(kw_only=True)
    class _GhostTerminationCfg(TerminationTermCfg):
        def __call__(self, env: ManagerEnv) -> TerminationTerm:
            del env
            return TerminationTerm(_ghost_termination)

    cfg = _ModelingEnvCfg(truncations={})
    cfg.terminations = {"ghost": _GhostTerminationCfg()}
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    env.init_state()  # terminations first run at step(); construction passes

    with pytest.raises(ModelingDerivationError, match=r"termination\.ghost.*ghost"):
        env.modeling_view()


def test_query_types_must_declare_modelability() -> None:
    from motrix_env_core.sim.read import SimDataQuery

    with pytest.raises(TypeError, match="modelable"):

        @dataclass(frozen=True, kw_only=True)
        class _Undeclared(SimDataQuery):
            def compile_with(self, compiler, *, key: str) -> None:
                del compiler, key


def test_derivation_rejects_non_modelable_queries() -> None:
    from test_manager_sim_backend import _FakeManagerCfg

    from motrix_env_core.sim import LinkNetContactForceQuery

    cfg = _FakeManagerCfg()
    cfg.queries.data["contact"] = LinkNetContactForceQuery(link="robot")
    env = ManagerEnv(cfg, num_envs=2, backend="fake-manager")
    env.init_state()

    with pytest.raises(ModelingDerivationError, match=r"contact.*LinkNetContactForceQuery.*not modelable"):
        env.modeling_view()


def test_exogenous_annotation_validation() -> None:
    cfg = _ModelingEnvCfg(truncations={})
    cfg.commands.phase.exogenous = {"nope": ExogenousRule(kind=ExogenousKind.INCREMENT)}
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    env.init_state()
    with pytest.raises(ModelingDerivationError, match="not a per-environment state leaf"):
        env.modeling_view()

    cfg = _ModelingEnvCfg(truncations={})
    cfg.commands.phase.exogenous = {"command": ExogenousRule(kind=ExogenousKind.COPY_ACTION, action_term="missing")}
    env = ManagerEnv(cfg, num_envs=2)
    env.physics_step = lambda: None
    env.init_state()
    with pytest.raises(ModelingDerivationError, match="unknown action term"):
        env.modeling_view()


# ---------------------------------------------------------------------------
# Sim-field addressing on the fake backend (declared read views)
# ---------------------------------------------------------------------------


def test_dual_path_matches_on_fake_backend_sim_fields() -> None:
    from test_manager_sim_backend import _FakeManagerCfg

    cfg = _FakeManagerCfg()
    env = ManagerEnv(cfg, num_envs=2, backend="fake-manager")
    env.init_state()
    program = env.modeling_view().program
    assert set(env.modeling_view().fields) == {"dof_pos"}

    program.sync_from_env()
    action = torch.full((2, 1), 0.5, dtype=torch.float32)
    captured = _capture_evaluate_inputs(env)
    state = env.step(action)

    reward, terminated, truncated = program.evaluate(captured, action)
    np.testing.assert_allclose(reward.numpy(), state.reward.numpy())
    np.testing.assert_array_equal(terminated.numpy(), state.terminated.numpy())
    np.testing.assert_array_equal(truncated.numpy(), state.truncated.numpy())
    program.advance_exogenous(action)
    obs = program.observe(captured)
    np.testing.assert_allclose(obs["policy"].numpy(), state.obs.policy.numpy())

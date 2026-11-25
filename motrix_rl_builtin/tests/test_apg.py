# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG numerical, inference and episode-boundary contracts without simulation."""

from types import SimpleNamespace

import pytest
import torch
from motrix_rl_builtin.apg.bridge import TransitionAssembler, matrix_quaternion
from motrix_rl_builtin.apg.config import NetworkCfg
from motrix_rl_builtin.apg.dynamics import integrate, represent
from motrix_rl_builtin.apg.learner import lambda_returns
from motrix_rl_builtin.apg.networks import History, MotionActor, Normalizer, Policy
from motrix_rl_builtin.apg.observations import ApgObservationLayout
from motrix_rl_builtin.apg.train import restore_policy

from motrix_env_core.math import quaternion


def test_normalizer_streaming_variance_and_input_gradient():
    normalizer = Normalizer(2)
    values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [10.0, -2.0]])
    normalizer.update(values[:2])
    normalizer.update(values[2:])
    torch.testing.assert_close(normalizer._mean[0], values.mean(0))
    torch.testing.assert_close(normalizer._var[0], values.var(0, unbiased=False))
    inputs = values.clone().requires_grad_()
    normalizer(inputs).sum().backward()
    assert torch.all(inputs.grad != 0)
    assert normalizer.count == len(values)


def test_gencoord_zero_delta_constant_twist_and_gradient():
    aux = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    rep = represent(
        aux, torch.tensor([[2.0, 0.0, 0.0]]), torch.zeros(1, 3), torch.tensor([[0.2]]), torch.tensor([[0.3]])
    )
    prediction = torch.zeros(1, 7, requires_grad=True)
    next_aux, next_rep = integrate(aux, rep, prediction, 0.1, 4)
    torch.testing.assert_close(next_aux[:, :3], torch.tensor([[0.2, 0.0, 1.0]]))
    torch.testing.assert_close(next_rep[:, 10:11], torch.tensor([[0.23]]))
    (next_aux.sum() + next_rep.sum()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_gencoord_rotation_exponential_and_gradients():
    # Cover stationary, tiny, both sides of the Taylor boundary, and large
    # rotations. The independent CPU sinc reference needs no CUDA compiler.
    angular = torch.tensor(
        [[0, 0, 0], [1e-8, -2e-8, 3e-8], [0.009, 0, 0], [0.011, 0, 0], [1, -2, 3]],
        dtype=torch.float64,
    )
    aux = torch.tensor([[0, 0, 1, 0, 0, 0, 1]], dtype=torch.float64).expand(len(angular), -1)
    rep = represent(aux, torch.zeros_like(angular), angular, aux[:, :1], aux[:, :1])
    prediction = torch.zeros(len(angular), 7, dtype=torch.float64, requires_grad=True)
    next_aux, _ = integrate(aux, rep, prediction, 1.0, 4)
    angle = angular.norm(dim=-1, keepdim=True)
    expected = torch.cat([angular * (0.5 * torch.sinc(angle / (2 * torch.pi))), torch.cos(angle / 2)], -1)
    torch.testing.assert_close(next_aux[:, 3:7], expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda delta: integrate(aux, rep, delta, 1.0, 4), (prediction,))


def test_fk_rotation_roundtrip_and_coincident_reward_gradient():
    q = torch.nn.functional.normalize(torch.randn(20, 4), dim=-1).requires_grad_()
    matrix = quaternion.to_matrix(q)
    recovered = matrix_quaternion(matrix)
    torch.testing.assert_close(quaternion.to_matrix(recovered), matrix, atol=1e-6, rtol=1e-6)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True)
    quaternion.rotation_distance(identity, identity.detach()).square().sum().backward()
    assert torch.isfinite(identity.grad).all()


def test_lambda_return_bootstrap_distinguishes_truncation_and_failure():
    rewards = torch.ones(2, 3)
    next_values = torch.full_like(rewards, 10.0)
    terminated = torch.tensor([[True, False, False], [False, False, False]])
    truncated = torch.tensor([[False, True, False], [False, False, False]])
    returns = lambda_returns(rewards, next_values, terminated, truncated, 0.9, 1.0)
    torch.testing.assert_close(returns[0], torch.tensor([1.0, 10.0, 10.0]))
    # The continuing lane uses future rewards; vary the second reward to
    # distinguish this recursion from a one-step bootstrap.
    rewards[1] = 2
    returns = lambda_returns(rewards, next_values, terminated, truncated, 0.9, 1.0)
    torch.testing.assert_close(returns[0], torch.tensor([1.0, 10.0, 10.9]))


def test_history_reset_and_policy_checkpoint_parity():
    from dataclasses import asdict

    cfg = NetworkCfg(actor_hidden_dims=(8,), latent_dim=3, history_length=3)
    frame = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    history = History(frame, 3)
    history.advance(frame + 10, torch.tensor([True, False]))
    torch.testing.assert_close(history.value[0], (frame[0] + 10).expand(3, -1))
    torch.testing.assert_close(history.value[1, :2], frame[1].expand(2, -1))
    actor = MotionActor(2, 4, 1, cfg)
    policy = Policy(actor, [-2.0], [2.0])
    obs = torch.cat([history.value.flatten(1), torch.randn(2, 4)], -1)
    checkpoint = {
        "policy": policy.state_dict(),
        "network": asdict(cfg),
        "proprio_dim": 2,
        "goal_dim": 4,
        "action_dim": 1,
    }
    restored = restore_policy(checkpoint)
    torch.testing.assert_close(restored(obs), policy(obs))
    torch.testing.assert_close(restored.low, policy.low)
    torch.testing.assert_close(restored.high, policy.high)
    count = restored.actor.obs_normalizer.count.clone()
    restored(obs).sum().backward()
    assert restored.actor.encoder[0].weight.grad.abs().sum() > 0
    assert restored.actor.obs_normalizer.count == count


def test_assembler_keeps_owned_true_terminal_successor():
    sim = torch.tensor([[1.0], [2.0]])
    phase = torch.tensor([[3], [4]])
    env = SimpleNamespace(
        sim_data=SimpleNamespace(view=lambda key: sim),
        command_terms={"goal": SimpleNamespace(phase=phase)},
        action_space=SimpleNamespace(shape=(1,)),
    )
    view = SimpleNamespace(fields={"position": SimpleNamespace(shape=(1,))}, exogenous={"goal.phase": None})
    assembler = TransitionAssembler(env, view, 1)
    before = assembler.snapshot()
    sim[:] = torch.tensor([[0.0], [20.0]])  # first row auto-reset
    phase[:] = torch.tensor([[0], [5]])
    final = SimpleNamespace(
        env_ids=torch.tensor([0]),
        sim={"position": torch.tensor([[10.0]])},
        commands={"goal.phase": torch.tensor([[4]])},
    )
    state = SimpleNamespace(
        final_state=final,
        reward=torch.ones(2),
        terminated=torch.tensor([True, False]),
        truncated=torch.zeros(2, dtype=torch.bool),
    )
    batch = assembler.assemble(before, torch.zeros(2, 1), torch.zeros(2, 1), state)
    sim.zero_()
    torch.testing.assert_close(batch["position"], torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(batch["next_position"], torch.tensor([[10.0], [20.0]]))
    torch.testing.assert_close(batch["next_command/goal.phase"], torch.tensor([[4], [5]]))


@pytest.mark.integration
def test_onnx_history_goal_inference_parity(tmp_path):
    from dataclasses import asdict

    from motrix_rl_builtin.apg.export import ApgOnnxExporter

    from motrix_rl.deploy.api import OnnxExportRequest

    cfg = NetworkCfg(actor_hidden_dims=(8,), latent_dim=3, history_length=2)
    policy = Policy(MotionActor(3, 4, 2, cfg), [-1.0, -2.0], [1.0, 2.0])
    path = tmp_path / "policy.pt"
    torch.save(
        {"policy": policy.state_dict(), "network": asdict(cfg), "proprio_dim": 3, "goal_dim": 4, "action_dim": 2}, path
    )
    request = OnnxExportRequest(
        run=SimpleNamespace(metadata=SimpleNamespace(env_name="test")), checkpoint=path, task_config=None
    )
    artifact = ApgOnnxExporter().export(request)
    assert artifact.report.input_spec.shape == (None, 10)
    assert artifact.report.parity.max_abs_error < 1e-5


@pytest.mark.parametrize("velocity_slice", [slice(4, 7), slice(7, 10)])
def test_world_model_loss_supervises_root_velocity(velocity_slice):
    from motrix_rl_builtin.apg.config import WorldModelCfg
    from motrix_rl_builtin.apg.dynamics import state_loss

    aux = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    target = represent(aux, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 1), torch.zeros(1, 1))
    prediction = target.clone()
    prediction[:, velocity_slice] = 1
    prediction.requires_grad_()
    # Pose and joints match exactly; velocity errors must still train the model.
    loss = state_loss(aux, prediction, aux, target, WorldModelCfg()).sum()
    assert loss > 0
    loss.backward()
    assert (prediction.grad[:, velocity_slice] > 0).all()


def test_velocity_increment_roundtrip_and_normalized_critic_gradient():
    from motrix_rl_builtin.apg.dynamics import velocity_increment
    from motrix_rl_builtin.apg.networks import Critic

    aux = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    rep = represent(aux, torch.ones(1, 3), torch.ones(1, 3), torch.zeros(1, 2), torch.ones(1, 2))
    delta = torch.randn(1, 8) * 0.1
    next_aux, next_rep = integrate(aux, rep, delta, 0.02, 4)
    torch.testing.assert_close(velocity_increment(aux, rep, next_aux, next_rep), delta, atol=1e-6, rtol=1e-5)
    critic = Critic(4, (8,), False)
    inputs = torch.randn(3, 4, requires_grad=True)
    critic.input_normalizer.update(inputs.detach())
    critic.return_normalizer.update(torch.tensor([[2.0], [4.0], [6.0]]))
    values = critic(inputs)
    torch.testing.assert_close(critic.return_normalizer(values), critic.normalized(inputs))
    values.sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0


def test_lambda_returns_supports_fractional_failure_probability():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[10.0], [20.0]])
    failure = torch.tensor([[0.25], [0.5]])
    truncated = torch.tensor([[False], [True]])
    result = lambda_returns(rewards, values, failure, truncated, 0.9, 1.0)
    torch.testing.assert_close(result, torch.tensor([[8.425], [11.0]]))


def test_muon_adam_training_state_restores_next_update():
    import copy

    from motrix_rl_builtin.apg.learner import optimize
    from motrix_rl_builtin.apg.networks import mlp
    from motrix_rl_builtin.apg.optim import MuonAdam

    net = mlp(3, (8, 8), 2)
    optimizer = MuonAdam(net, (net,), 1e-3, (0.9, 0.95), False)
    inputs = torch.randn(5, 3)
    optimize(optimizer, net(inputs).square().mean(), net.parameters())
    restored = copy.deepcopy(net)
    restored_optimizer = MuonAdam(restored, (restored,), 1e-3, (0.9, 0.95), False)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, opt in ((net, optimizer), (restored, restored_optimizer)):
        optimize(opt, model(inputs).square().mean(), model.parameters())
    torch.testing.assert_close(net(inputs), restored(inputs))


@pytest.mark.parametrize("ratio, steps, expected", [(0.4, 100, 40), (0.3, 100, 30), (2.5, 10, 25)])
def test_update_quota_fractional_ratio_and_resume(ratio, steps, expected):
    from motrix_rl_builtin.apg.collector import UpdateSchedule
    from motrix_rl_builtin.apg.config import AgentCfg

    cfg = AgentCfg(history_length=7, learning_starts=2, num_updates=ratio)
    cfg.world_model.horizon = 3
    schedule = UpdateSchedule(cfg)
    assert all(schedule.advance(i) == 0 for i in range(1, 8))
    split = 3
    count = sum(schedule.advance(i) for i in range(8, 8 + split))
    resumed = UpdateSchedule(cfg, remainder=schedule.remainder)
    assert all(resumed.advance(i) == 0 for i in range(1, 8))
    count += sum(resumed.advance(i) for i in range(8, 8 + steps - split))
    assert count == expected


def test_learning_rate_uses_completed_updates_and_decays_all_optimizer_groups():
    from motrix_rl_builtin.apg.config import ApgCfg
    from motrix_rl_builtin.apg.learner import Learner

    learner = Learner.__new__(Learner)
    learner.cfg = ApgCfg()
    learner.device = torch.device("cpu")
    learner.cfg.agent.learning_rate_decay_steps = 10
    learner.cfg.agent.actor_learning_rate = 0.1
    learner.cfg.agent.critic_learning_rate = 0.2
    learner.cfg.agent.world_model.learning_rate = 0.3
    # Include both Muon and Adam groups, as well as an ordinary Adam optimizer.
    learner.actor_optimizer = SimpleNamespace(param_groups=[{}, {}])
    learner.critic_optimizer = SimpleNamespace(param_groups=[{}])
    learner.wm_optimizer = SimpleNamespace(param_groups=[{}, {}])
    learner.update_world_model = lambda replay: {}
    learner.update_policy = lambda replay: {}
    for completed, scale in ((0, 1.0), (5, 0.515), (10, 0.03), (15, 0.03)):
        learner.world_model_iterations = completed
        learner.update(None)
        for optimizer, base in (
            (learner.actor_optimizer, 0.1),
            (learner.critic_optimizer, 0.2),
            (learner.wm_optimizer, 0.3),
        ):
            assert all(group["lr"] == pytest.approx(base * scale) for group in optimizer.param_groups)


def test_policy_ema_artifact_and_training_state_keep_separate_weights():
    from dataclasses import asdict

    from motrix_rl_builtin.apg.learner import Learner
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    cfg = NetworkCfg(actor_hidden_dims=(8,), history_length=2)
    learner = Learner.__new__(Learner)
    learner.actor = MotionActor(3, 4, 2, cfg)
    learner.policy = Policy(learner.actor, [-1, -1], [1, 1])
    learner.policy_ema = AveragedModel(learner.policy, multi_avg_fn=get_ema_multi_avg_fn(0.5), use_buffers=True)
    learner.proprio_dim = 3
    learner.layout = SimpleNamespace(joint_count=2)
    learner.policy_ema.update_parameters(learner.policy)
    with torch.no_grad():
        learner.actor.decoder[-1].bias.add_(0.2)
    learner.policy_ema.update_parameters(learner.policy)
    artifact = {**learner.policy_state(), "network": asdict(cfg)}
    inputs = torch.randn(4, 10)
    restored = restore_policy(artifact)
    torch.testing.assert_close(restored(inputs), learner.policy_ema(inputs))
    assert not torch.allclose(restored(inputs), learner.policy(inputs))
    # Playback of a full training checkpoint must select the same averaged weights.
    training = {**artifact, "policy": learner.policy.state_dict(), "policy_ema": learner.policy_ema.state_dict()}
    torch.testing.assert_close(restore_policy(training)(inputs), restored(inputs))
    # Existing policy artifacts retain their inference compatibility.
    legacy = {**artifact, "network": {**artifact["network"], "mixed_precision": artifact["network"]["amp"]}}
    del legacy["network"]["amp"]
    torch.testing.assert_close(restore_policy(legacy)(inputs), restored(inputs))


class _CollectorEnvCfg:
    def __init__(self, fail_step=0, swapped=False):
        self.fail_step = fail_step
        self.swapped = swapped

    def validate(self):
        pass


class _CollectorEnv:
    """Tiny process fixture exercising the real APG assembler and transport."""

    def __init__(self, cfg, num_envs, **_):
        self.cfg, self.num_envs = cfg, num_envs
        self.action_space = SimpleNamespace(shape=(1,), low=[-1.0], high=[1.0])
        from gymnasium.spaces import Box

        from motrix_env_core.manager.resolver import ObservationTermLayout
        from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec

        self.observation_spec = ObservationSpec(
            {
                "policy": ObservationGroupSpec(Box(-float("inf"), float("inf"), (10,))),
                "value": ObservationGroupSpec(Box(-float("inf"), float("inf"), (2,))),
            }
        )
        terms = (
            (ObservationTermLayout("goal", slice(0, 1)), ObservationTermLayout("proprio", slice(1, 10)))
            if cfg.swapped
            else (ObservationTermLayout("proprio", slice(0, 9)), ObservationTermLayout("goal", slice(9, 10)))
        )
        self.manager_layout = SimpleNamespace(observations={"policy": SimpleNamespace(terms=terms)})
        self.counter = torch.zeros(num_envs, 1)
        self.sim_data = self
        self.command_terms = {"counter": self}

    def view(self, key):
        return self.counter

    def modeling_view(self):
        return SimpleNamespace(fields={"counter": SimpleNamespace(shape=(1,))}, exogenous=())

    def training_state_dict(self):
        return {"counter": self.counter.clone()}

    def load_training_state_dict(self, state):
        self.counter.copy_(state["counter"])

    def observation(self):
        proprio, goal = self.counter.expand(-1, 9), self.counter + 100
        return torch.cat((goal, proprio) if self.cfg.swapped else (proprio, goal), -1)

    def init_state(self):
        return SimpleNamespace(obs=SimpleNamespace(policy=self.observation()))

    def process_metrics(self):
        return {"counter": float(self.counter.mean())}

    def step(self, actions):
        self.counter.add_(1)
        if self.counter[0, 0] == self.cfg.fail_step:
            raise ValueError("injected collector failure")
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return SimpleNamespace(
            obs=SimpleNamespace(policy=self.observation()),
            reward=self.counter[:, 0].clone(),
            reward_terms={},
            terminated=done,
            truncated=done,
            done=done,
            final_state=None,
            process_metrics=self.process_metrics,
        )


@pytest.mark.integration
@pytest.mark.parametrize("fail_step,stop_after", [(0, 5), (2, 5), (0, 1)])
@pytest.mark.parametrize("swapped", [False, True])
def test_async_collector_order_checkpoint_and_failure(fail_step, stop_after, swapped):
    from motrix_rl_builtin.apg.collector import AsyncCollector
    from motrix_rl_builtin.apg.config import ApgCfg

    from motrix_rl.transport.schema import TransitionField, wm_transition_schema

    cfg = ApgCfg(device="cpu")
    cfg.agent.actor_hidden_dims = (4,)
    cfg.agent.history_length = 2
    cfg.agent.amp = False
    cfg.trainer.num_learning_iterations = 5
    cfg.trainer.async_options.ring_capacity = 1
    context = SimpleNamespace(
        num_envs=2, seed=1, sim=None, logging=SimpleNamespace(interval=1), checkpoint=SimpleNamespace(interval=3)
    )
    actor = MotionActor(9, 1, 1, cfg.agent)
    schema = wm_transition_schema([TransitionField("counter", (1,), torch.float32)], 1, obs_dim=9)
    env_cfg = _CollectorEnvCfg(fail_step, swapped)
    layout = ApgObservationLayout.from_env(_CollectorEnv(env_cfg, 2))
    collector = AsyncCollector(
        SimpleNamespace(env_cls=_CollectorEnv, env_cfg=env_cfg),
        cfg,
        context,
        actor,
        schema,
        observation_layout=layout,
    )
    try:
        for step in range(1, stop_after + 1):
            if step == fail_step:
                with pytest.raises(RuntimeError, match="injected collector failure"):
                    collector.collect()
                break
            result = collector.collect()
            torch.testing.assert_close(result.transition["counter"], torch.full((2, 1), float(step - 1)))
            torch.testing.assert_close(result.transition["next_counter"], torch.full((2, 1), float(step)))
            torch.testing.assert_close(result.transition["obs"], torch.full((2, 9), float(step - 1)))
            torch.testing.assert_close(layout.goal(result.observation), torch.full((2, 1), float(step - 1 + 100)))
            assert result.metrics["counter"] == step
            if step in (3, 5):
                assert result.rng["torch"].dtype == torch.uint8
                torch.testing.assert_close(result.commands["counter"]["counter"], result.transition["next_counter"])
            else:
                assert result.commands is None
            actor.obs_normalizer.update(torch.full((2, 9), float(step)))
            collector.publish()
    finally:
        collector.close()
    assert not collector.process.is_alive()


@pytest.mark.parametrize("kind", ["world_model", "critic"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
def test_compiled_backward_preserves_repeated_updates_and_owned_metrics(kind, device):
    """Backward regions must match eager updates and retain metrics across graph generations."""
    import copy

    from motrix_rl_builtin.apg.config import ApgCfg
    from motrix_rl_builtin.apg.dynamics import Dynamics
    from motrix_rl_builtin.apg.learner import Learner, finish_update, optimize
    from motrix_rl_builtin.apg.networks import Critic

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(92)
    learner = Learner.__new__(Learner)
    learner.cfg = ApgCfg()
    learner.cfg.agent.world_model.horizon = 2
    learner.layout = SimpleNamespace(control=lambda action: action)
    if kind == "world_model":
        learner.dynamics = Dynamics(1, (8, 8), 0.02, 4, False).to(device)
        model = learner.dynamics
        learner.wm_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = learner.wm_optimizer
        backward = learner._world_model_backward
        aux = torch.zeros(4, 7, device=device)
        aux[:, -1] = 1
        rep = torch.zeros(4, 12, device=device)
        args = (
            aux,
            rep,
            aux[:, None].expand(-1, 2, -1).clone(),
            rep[:, None].expand(-1, 2, -1).clone(),
            torch.randn(4, 2, 1, device=device),
            torch.tensor([[True, True], [True, False], [False, False], [True, True]], device=device),
        )
        reference = copy.deepcopy(learner)
        reference_model, reference_optimizer = reference.dynamics, reference.wm_optimizer

        def loss_fn():
            return reference._world_model_loss(*args)[0]
    else:
        learner.critic = Critic(5, (8, 8), False).to(device)
        model = learner.critic
        learner.critic_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = learner.critic_optimizer
        backward = learner._critic_backward
        args = (
            torch.randn(8, 5, device=device),
            torch.randn(8, device=device),
            torch.tensor([1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0], device=device),
        )
        reference = copy.deepcopy(learner)
        reference_model, reference_optimizer = reference.critic, reference.critic_optimizer

        def loss_fn():
            error = (reference.critic.normalized(args[0]).squeeze(-1) - args[1]).square()
            return (error * args[2]).sum() / args[2].sum()

    runtime = (
        torch.compile(backward, mode="reduce-overhead")
        if device == "cuda"
        else torch.compile(backward, backend="aot_eager")
    )
    metrics, expected_metrics = [], []
    for _ in range(3):
        expected_loss = loss_fn()
        expected_metrics.append(expected_loss.detach().clone())
        optimize(reference_optimizer, expected_loss, reference_model.parameters())
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        result = runtime(*args)
        metrics.append((result[0] if kind == "world_model" else result).clone())
        finish_update(optimizer, model.parameters())
        for actual, expected in zip(model.parameters(), reference_model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(torch.stack(metrics), torch.stack(expected_metrics), atol=2e-5, rtol=2e-4)


def test_archived_run_config_remains_restorable_after_worker_budget_migration(tmp_path):
    """Existing metadata-backed runs must still load for play/export, without rewriting history."""
    from motrix_rl_builtin.apg.config import ApgCfg
    from omegaconf import OmegaConf

    from motrix_rl import runs
    from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskConfig, TaskMeta

    cfg = TaskConfig(
        task=TaskMeta(env="g1-wbt-supermimic", rllib="motrix", algo="apg"),
        num_envs=2,
        play_num_envs=1,
        seed=1,
        logging=LoggingConfig(backend="tensorboard", interval=4),
        checkpoint=CheckpointConfig(interval=0),
        algo=ApgCfg(),
    )
    snapshot = runs.task_config_snapshot(cfg)
    # This field existed in real run snapshots before CPU budgets moved to workers.
    snapshot["algo"]["trainer"]["torch_threads"] = 4
    path = runs.task_config_path(tmp_path)
    OmegaConf.save(OmegaConf.create(snapshot), path)
    original = path.read_bytes()
    restored = runs.read_task_config(tmp_path, ApgCfg, cfg_override={"trainer": {"num_learning_iterations": 20}})
    restored.algo.validate()
    assert runs.task_config_snapshot(restored)["algo"]["agent"] == snapshot["algo"]["agent"]
    assert restored.algo.trainer.num_learning_iterations == 20
    assert path.read_bytes() == original


class _ActorTrajectoryFixture:
    """Small differentiable recurrence with explicit noise and distinct episode ends."""

    def _imagination_step(self, aux, rep, history, observation, workspace):
        step, rng = workspace
        action = self.actor(history, observation["policy"][:, 1:]).clamp(-1, 1)
        rep = rep + self.dynamics(action) + (rng % 17).to(rep.dtype) * 0.001
        reward = -(rep[:, 0] - 0.4).square()
        step, rng = step + 1, (rng * 7 + 3) % 101
        observation = {"policy": torch.cat([rep, aux], -1), "value": aux}
        terminated = ((aux[:, 0] == 2) & (step[:, 0] == 2)).to(rep.dtype)
        truncated = (aux[:, 0] == 1) & (step[:, 0] == 1)
        history = torch.cat([history[:, 1:], rep[:, None]], 1)
        return aux, rep, history, observation, reward, terminated, truncated, (step, rng)


def _actor_fixture(device, horizon):
    from motrix_rl_builtin.apg.config import ApgCfg
    from motrix_rl_builtin.apg.learner import Learner

    class Fixture(_ActorTrajectoryFixture, Learner):
        pass

    learner = Fixture.__new__(Fixture)
    learner.cfg = ApgCfg()
    cfg = learner.cfg.agent
    cfg.horizon, cfg.gamma, cfg.amp = horizon, 0.9, False
    cfg.actor_hidden_dims, cfg.latent_dim, cfg.history_length = (8, 8), 2, 2
    learner.actor = MotionActor(1, 1, 1, cfg).to(device)
    learner.dynamics = torch.nn.Linear(1, 1, bias=False).to(device).requires_grad_(False)
    learner.critic = torch.nn.Linear(2, 1).to(device).requires_grad_(False)
    learner.actor_optimizer = torch.optim.Adam(learner.actor.parameters(), lr=1e-3)
    learner.value_max = 100.0
    learner._critic_runtime = learner.critic
    learner._imagination_step_runtime = learner._imagination_step
    aux = torch.arange(4, device=device, dtype=torch.float32)[:, None]
    rep = torch.full((4, 1), 0.1, device=device)
    history = rep[:, None].expand(-1, 2, -1).clone()
    observation = {"policy": torch.cat([rep, aux], -1), "value": aux}
    workspace = (torch.zeros_like(aux, dtype=torch.long), torch.arange(4, device=device)[:, None])
    accepted = torch.tensor([True, True, True, False], device=device)
    return learner, (aux, rep, history, observation, workspace, accepted)


def _actor_reference_loss(learner, args):
    """Per-lane oracle stops at the true terminal state instead of masking a padded rollout."""
    aux, rep, history, _, (_, rng), accepted = args
    returns = []
    for lane in torch.nonzero(accepted).flatten().tolist():
        x, h, noise = rep[lane : lane + 1], history[lane : lane + 1], rng[lane : lane + 1]
        goal = aux[lane : lane + 1]
        total = 0
        for step in range(learner.cfg.agent.horizon):
            action = learner.actor(h, goal).clamp(-1, 1)
            x = x + learner.dynamics(action) + (noise % 17).to(x.dtype) * 0.001
            noise = (noise * 7 + 3) % 101
            total = total - learner.cfg.agent.gamma**step * (x[:, 0] - 0.4).square()
            h = torch.cat([h[:, 1:], x[:, None]], 1)
            if lane == 2 and step == 1:  # Hard termination has no bootstrap.
                break
            if (lane == 1 and step == 0) or step == learner.cfg.agent.horizon - 1:
                total = total + learner.cfg.agent.gamma ** (step + 1) * learner.critic(torch.cat([x, goal], -1))[:, 0]
                break
        returns.append(total)
    return -torch.stack(returns).sum() / accepted.sum()


@pytest.mark.parametrize("horizon", [3, 9])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
def test_actor_backward_preserves_episode_gradients_and_owned_outputs(horizon, device):
    import copy

    from motrix_rl_builtin.apg.learner import finish_update, optimize

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(41)
    learner, args = _actor_fixture(device, horizon)
    reference = copy.deepcopy(learner)
    learner._imagination_step_runtime = (
        torch.compile(learner._imagination_step, fullgraph=True)
        if device == "cuda"
        else torch.compile(learner._imagination_step, fullgraph=True, backend="aot_eager")
    )
    if device == "cuda":
        learner.device = torch.device(device)
        learner._actor_graph = None

        def make_workspace_buffer():
            buffers = tuple(value.clone() for value in args[4])

            def stage(state):
                for target, source in zip(buffers, state, strict=True):
                    target.copy_(source)
                return buffers

            return stage

        learner.program = SimpleNamespace(working_state_buffer=make_workspace_buffer)
        # Other learner stages record automatic graphs between actor replays.
        # Their cuBLAS workspace cleanup must not invalidate actor's graph.
        other_stage = torch.compile(lambda x: (x @ x).sin(), mode="reduce-overhead")
        other_input = torch.randn(32, 32, device=device)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            other_stage(other_input)
    runtime = learner._captured_actor_backward if device == "cuda" else learner._actor_backward
    # Small numerical trajectories remain traceable without data-dependent breaks.
    if device == "cpu":
        fullgraph = torch.compile(learner._actor_rollout_loss, fullgraph=True, backend="aot_eager")
        torch.testing.assert_close(fullgraph(*args)[0], _actor_reference_loss(reference, args))
    kept = []
    for iteration in range(3):
        args[1].add_(0.01)
        args[2].add_(0.02)
        args[3]["policy"][:, :1].copy_(args[1])
        args[4][1].add_(1)
        args[5].copy_(torch.tensor([True, iteration != 1, True, iteration == 1], device=device))
        expected = _actor_reference_loss(reference, args)
        _, expected_trajectory, expected_workspace = reference._actor_rollout_loss(*args)
        optimize(reference.actor_optimizer, expected, reference.actor.parameters())
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        loss, trajectory, workspace = runtime(*args)
        owned = (loss.clone(), *(v.detach().clone() for v in trajectory), *(v.detach().clone() for v in workspace))
        kept.append((owned, tuple(v.clone() for v in owned)))
        for actual, wanted in zip(trajectory + workspace, expected_trajectory + expected_workspace, strict=True):
            torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(loss, expected.detach(), atol=2e-5, rtol=2e-4)
        finish_update(learner.actor_optimizer, learner.actor.parameters())
        if device == "cuda":
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                torch.testing.assert_close(other_stage(other_input).clone(), (other_input @ other_input).sin())
        for actual, wanted in zip(learner.actor.parameters(), reference.actor.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-4)
        if iteration == 1:
            learner.actor.load_state_dict(copy.deepcopy(learner.actor.state_dict()))
            learner.actor_optimizer.load_state_dict(copy.deepcopy(learner.actor_optimizer.state_dict()))
        assert all(p.grad is None for model in (learner.dynamics, learner.critic) for p in model.parameters())
        assert torch.count_nonzero(args[4][0]) == 0  # Borrowed inputs are not advanced.
        torch.testing.assert_close(workspace[0], torch.full_like(workspace[0], horizon))
        for saved, snapshot in kept:
            for actual, wanted in zip(saved, snapshot, strict=True):
                torch.testing.assert_close(actual, wanted)


def test_actor_empty_start_batch_does_not_advance_training_state():
    learner, _ = _actor_fixture("cpu", 3)
    replay = SimpleNamespace(sample_starts=lambda batch: {"intact": torch.zeros(4, dtype=torch.bool)})
    assert learner.update_policy(replay) == {}
    assert not learner.actor_optimizer.state


@pytest.mark.integration
def test_compiled_normalization_preserves_cumulative_statistics():
    """Fused preparation must keep every running mean, variance and sample count."""
    import copy

    from motrix_rl_builtin.apg.bridge import GenCoordLayout
    from motrix_rl_builtin.apg.config import AgentCfg
    from motrix_rl_builtin.apg.dynamics import Dynamics
    from motrix_rl_builtin.apg.learner import Learner
    from motrix_rl_builtin.apg.networks import MotionActor

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(71)
    layout = GenCoordLayout.__new__(GenCoordLayout)
    layout.root_keys = ("p", "q", "v", "w")
    layout.joint_keys = ("j", "jd")
    layout.control = lambda actions: actions * 0.5
    eager = Learner.__new__(Learner)
    eager.layout, eager.device, eager.proprio_dim = layout, torch.device("cuda"), 12
    eager.observation_layout = ApgObservationLayout(slice(0, 12), slice(12, 17), 5, 2)
    eager.dynamics = Dynamics(2, (16, 16), 0.02, 4).cuda()
    eager.actor = MotionActor(12, 5, 2, AgentCfg(actor_hidden_dims=(16, 16))).cuda()
    eager._normalization_runtime = eager._update_normalization
    compiled = copy.deepcopy(eager)
    compiled._normalization_runtime = torch.compile(compiled._update_normalization, fullgraph=True)
    for _ in range(4):
        state = {
            key: torch.randn(64, width)
            for key, width in zip(layout.root_keys + layout.joint_keys, (3, 4, 3, 3, 2, 2), strict=True)
        }
        state["q"] = torch.nn.functional.normalize(state["q"], dim=-1)
        observation, actions = torch.randn(64, 17), torch.randn(64, 2)
        transition = {"next_" + key: value + torch.randn_like(value) * 0.01 for key, value in state.items()}
        transition["next_q"] = torch.nn.functional.normalize(transition["next_q"], dim=-1)
        transition["obs"] = observation[:, :12]
        for learner in (eager, compiled):
            learner.update_normalization(state, actions, observation, transition)
        for name in ("dynamics", "actor"):
            expected = dict(getattr(eager, name).named_buffers())
            actual = dict(getattr(compiled, name).named_buffers())
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.integration
def test_nonfinite_cuda_gradients_abort_isolated_process():
    """A device-side finite check must fail training; isolate its poisoned CUDA context."""
    import subprocess
    import sys
    import textwrap

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    code = textwrap.dedent("""
        import torch
        from motrix_rl_builtin.apg.learner import finish_update
        parameter = torch.nn.Parameter(torch.ones(2, device="cuda"))
        parameter.grad = torch.full_like(parameter, float("nan"))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        try:
            finish_update(optimizer, [parameter])
            torch.cuda.synchronize()
        except RuntimeError as error:
            if "device-side assert" not in str(error):
                raise
            print("nonfinite gradients rejected")
        else:
            raise AssertionError("nonfinite gradients did not abort CUDA work")
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "nonfinite gradients rejected" in result.stdout


def test_borrowed_collection_keeps_slot_until_consumed_and_releases_on_error():
    """A staging consumer owns the lease even when ingestion exits with an error."""
    import pickle

    from motrix_rl_builtin.apg.collector import AsyncCollector, Collection

    from motrix_rl.transport.ring import HostTransitionRing
    from motrix_rl.transport.schema import TransitionField, TransitionSchema

    schema = TransitionSchema(
        (TransitionField("value", (1,), torch.float32), TransitionField("collector_observation", (2,), torch.float32))
    )
    collector = AsyncCollector.__new__(AsyncCollector)
    collector.ring = HostTransitionRing(schema, 1, 2)
    metadata = pickle.dumps(Collection({}, torch.empty(0), {}, {}, None, 0.0))
    collector.messages = SimpleNamespace(poll=lambda timeout: True, recv_bytes=lambda: metadata)
    collector.ring.push({"value": torch.ones(2, 1), "collector_observation": torch.ones(2, 2)})
    with pytest.raises(RuntimeError, match="ingestion failed"):
        with collector.borrow() as result:
            assert collector.ring.is_full()
            owned = result.transition["value"].clone()
            torch.testing.assert_close(owned, torch.ones(2, 1))
            raise RuntimeError("ingestion failed")
    assert not collector.ring.is_full()
    collector.ring.push({"value": torch.full((2, 1), 2.0), "collector_observation": torch.full((2, 2), 2.0)})
    torch.testing.assert_close(owned, torch.ones(2, 1))
    result = collector.collect()
    collector.ring.push({"value": torch.full((2, 1), 3.0), "collector_observation": torch.full((2, 2), 3.0)})
    torch.testing.assert_close(result.transition["value"], torch.full((2, 1), 2.0))

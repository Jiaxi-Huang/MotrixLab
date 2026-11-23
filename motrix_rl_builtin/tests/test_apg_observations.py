# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Named APG input semantics across collection, imagination and checkpoint loading."""

from types import SimpleNamespace

import pytest
import torch
from gymnasium.spaces import Box
from motrix_rl_builtin.apg.collector import Collector
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.learner import Learner
from motrix_rl_builtin.apg.networks import MotionActor, Policy
from motrix_rl_builtin.apg.observations import ApgObservationLayout

from motrix_env_core.manager.resolver import ObservationTermLayout
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec


def observation_env(proprio_dim=3, *, swapped=False, names=None, history=1, value=True):
    widths = {"proprio": proprio_dim, "goal": 2, "extra": 1}
    names = names or (["goal", "proprio"] if swapped else ["proprio", "goal"])
    terms, offset = [], 0
    for name in names:
        terms.append(ObservationTermLayout(name, slice(offset, offset + widths[name])))
        offset += widths[name]
    groups = {"policy": ObservationGroupSpec(Box(-float("inf"), float("inf"), (offset,)), history)}
    if value:
        groups["value"] = ObservationGroupSpec(Box(-float("inf"), float("inf"), (7,)))
    return SimpleNamespace(
        observation_spec=ObservationSpec(groups),
        manager_layout=SimpleNamespace(observations={"policy": SimpleNamespace(terms=terms)}),
        action_space=Box(-1.0, 1.0, (1,)),
    )


@pytest.mark.parametrize("width", [3, 8])
@pytest.mark.parametrize("compiled", [False, True])
def test_named_layout_preserves_collection_and_imagination_when_terms_reorder(width, compiled):
    cfg = ApgCfg()
    cfg.agent.actor_hidden_dims = (8,)
    cfg.agent.history_length = 2
    cfg.agent.amp = False
    cfg.agent.exploration_noise = 0
    actor = MotionActor(width, 2, 1, cfg.agent)
    policy = Policy(actor, [-1.0], [1.0])
    proprio = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    goal = torch.tensor([[23.0, 24.0], [25.0, 26.0]])
    results = []
    for swapped in (False, True):
        env = observation_env(width, swapped=swapped)
        layout = ApgObservationLayout.from_env(env)
        assert (layout.proprio_dim, layout.goal_dim, layout.value_dim, layout.action_dim) == (width, 2, 7, 1)
        obs = torch.cat((goal, proprio) if swapped else (proprio, goal), -1)
        state = SimpleNamespace(obs=SimpleNamespace(policy=obs))
        env.step = lambda action: SimpleNamespace(
            obs=SimpleNamespace(policy=obs + 1), done=torch.tensor([True, False]), reward_terms={}
        )
        assembler = SimpleNamespace(
            snapshot=lambda: {},
            assemble=lambda before, actions, frame, state: {"obs": frame.clone(), "actions": actions.clone()},
        )
        collector = Collector(env, state, actor, policy, assembler, cfg, "cpu", observation_layout=layout)
        result = collector.collect()
        torch.testing.assert_close(result.transition["obs"], proprio)
        torch.testing.assert_close(collector.history.value[0], (proprio[0] + 1).expand(2, -1))
        torch.testing.assert_close(collector.history.value[1, 0], proprio[1])

        # Exercise the real imagination step, including history gradients, with
        # a small differentiable state transition instead of a simulator.
        learner = Learner.__new__(Learner)
        learner.actor, learner.policy, learner.observation_layout = actor, policy, layout
        learner.device = torch.device("cpu")
        learner.layout = SimpleNamespace(root_keys=[], joint_keys=[])
        normalized = []
        learner._normalization_runtime = lambda before, after, actions, frame, target: normalized.append(
            (frame, target)
        )
        learner.update_normalization({}, result.transition["actions"], result.observation, result.transition)
        torch.testing.assert_close(normalized[0][0], proprio)
        torch.testing.assert_close(normalized[0][1], goal)
        learner.layout = SimpleNamespace(control=lambda action: action, decode=lambda aux, rep: rep)
        learner.dynamics = lambda aux, rep, action: (aux, rep)
        learner.program = SimpleNamespace(
            transition=lambda rep, action, workspace, **kwargs: (
                {"policy": rep},
                action[:, 0],
                torch.zeros(2),
                torch.zeros(2, dtype=torch.bool),
                workspace,
            )
        )
        history = proprio[:, None].repeat(1, 2, 1).requires_grad_()
        next_obs = (obs + 2).requires_grad_()
        step = (
            torch.compile(learner._imagination_step, backend="aot_eager", fullgraph=True)
            if compiled
            else learner._imagination_step
        )
        output = step(None, next_obs, history, {"policy": obs}, ())
        output[2].sum().backward()
        torch.testing.assert_close(history.grad[:, 0], torch.zeros_like(proprio))
        torch.testing.assert_close(history.grad[:, 1], torch.ones_like(proprio))
        torch.testing.assert_close(layout.proprio(next_obs.grad), torch.ones_like(proprio))
        torch.testing.assert_close(layout.goal(next_obs.grad), torch.zeros_like(goal))
        results.append((result.transition["actions"], output[2].detach(), output[4].detach()))
    for original, reordered in zip(*results, strict=True):
        torch.testing.assert_close(original, reordered)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"names": ["proprio"]}, "exactly 'proprio' and 'goal'"),
        ({"names": ["proprio", "goal", "extra"]}, "exactly 'proprio' and 'goal'"),
        ({"history": 2}, "history_length=1"),
        ({"value": False}, "privileged 'value'"),
    ],
)
def test_apg_rejects_unsupported_observation_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ApgObservationLayout.from_env(observation_env(**kwargs))


@pytest.mark.parametrize("dimension", ["proprio_dim", "goal_dim", "action_dim"])
def test_resume_rejects_checkpoint_dimensions_before_loading_weights(dimension):
    learner = Learner.__new__(Learner)
    learner.observation_layout = ApgObservationLayout.from_env(observation_env())
    checkpoint = {"proprio_dim": 3, "goal_dim": 2, "action_dim": 1}
    learner.observation_layout.validate_policy_dimensions(**checkpoint)
    checkpoint[dimension] += 1
    with pytest.raises(ValueError, match="do not match environment"):
        learner.load_state_dict(checkpoint)


@pytest.mark.parametrize("swapped,mismatch", [(False, False), (True, False), (True, True)])
def test_play_legacy_policy_uses_named_layout(tmp_path, monkeypatch, swapped, mismatch):
    from dataclasses import asdict

    from motrix_rl_builtin.apg.train import Trainer

    cfg = ApgCfg().agent
    cfg.actor_hidden_dims, cfg.history_length, cfg.amp = (8,), 2, False
    policy = Policy(MotionActor(3, 2, 1, cfg), [-0.3], [0.7])
    from motrix_rl_builtin.apg.config import NetworkCfg

    network = {name: getattr(cfg, name) for name in asdict(NetworkCfg())}
    path = tmp_path / "legacy-policy.pt"
    torch.save(
        {"policy": policy.state_dict(), "network": network, "proprio_dim": 3, "goal_dim": 2, "action_dim": 1},
        path,
    )
    env = observation_env(4 if mismatch else 3, swapped=swapped)
    proprio, goal = torch.randn(2, 3), torch.randn(2, 2)
    obs = torch.cat((goal, proprio) if swapped else (proprio, goal), -1)
    state = SimpleNamespace(
        obs=SimpleNamespace(policy=obs),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
    )
    env.init_state = lambda: state
    actions = []

    def step(action):
        actions.append(action.clone())
        return state

    env.step = step
    renderer = SimpleNamespace(render=lambda: False, close=lambda: None)
    monkeypatch.setattr("motrix_rl_builtin.apg.train.create_renderer", lambda env, render: renderer)
    trainer = Trainer.__new__(Trainer)
    trainer.device, trainer.context = torch.device("cpu"), SimpleNamespace(render=True)
    trainer._env = lambda play: env
    if mismatch:
        with pytest.raises(ValueError, match="do not match environment"):
            trainer.play(path)
        assert not actions
    else:
        trainer.play(path)
        expected = policy(torch.cat([proprio.repeat(1, 2), goal], -1))
        torch.testing.assert_close(actions[0], expected)

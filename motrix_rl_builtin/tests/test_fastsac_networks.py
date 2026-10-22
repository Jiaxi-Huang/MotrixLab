# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hidden-width configuration contract for the FastSAC actor and critic.

``actor_hidden_dims``/``critic_hidden_dims`` set every layer's width
explicitly (e.g. ``[1024, 1024, 1024]``); unset (``null``) they fall back to
the scalar ``actor_hidden_dim``/``critic_hidden_dim``, which keeps expanding
to the holosoma taper ``(dim, dim // 2, dim // 4)`` so every existing task
recipe and checkpoint keeps its exact network shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from motrix_rl_builtin.fastsac.config import resolve_hidden_dims
from torch import nn

from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.plugins import load_plugins

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    load_plugins()
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(task: str, overrides: list[str]) -> TrainConfig:
    with initialize_config_module(version_base=None, config_module="motrix_rl.configs"):
        config = compose(
            config_name="train",
            overrides=[f"hydra.searchpath=[file://{CONFIG_DIR}]", f"task={task}", *overrides],
        )
    return to_typed_config(config, TrainConfig)


def _linear_widths(module: nn.Module) -> list[int]:
    return [layer.out_features for layer in module.modules() if isinstance(layer, nn.Linear)]


def test_resolve_hidden_dims_expands_scalar_taper_and_validates_lists() -> None:
    assert resolve_hidden_dims(None, 512) == (512, 256, 128)
    assert resolve_hidden_dims([1024, 1024, 1024], 512) == (1024, 1024, 1024)
    assert resolve_hidden_dims([64], 512) == (64,)
    for invalid in ([], [0], [-8, 16]):
        with pytest.raises(ValueError, match="non-empty list of positive widths"):
            resolve_hidden_dims(invalid, 512)


def test_default_task_actor_and_critic_follow_explicit_hidden_dims() -> None:
    from motrix_rl_builtin.fastsac.agent import FastSacAgent
    from motrix_rl_builtin.fastsac.factory import make_actor

    config = _compose(
        "g1-wbt-dance/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dims=[64,32]",
            "algo.agent.critic_hidden_dims=[48,24]",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=2",
            "algo.agent.num_updates=1",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
        ],
    )
    agent = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=5,
        act_dim=1,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )

    # The actor trunk honors the explicit widths, the critic ensemble follows
    # its own list, and the actor head reads the last actor width.
    actor = agent.actor
    assert _linear_widths(actor.net) == [64, 32]
    assert actor.fc_mu.in_features == 32
    assert actor.fc_logstd.in_features == 32
    assert _linear_widths(agent.qnet.qnets[0].net) == [48, 24, 5]  # trunk + atom head

    # make_actor builds the same shapes for the default policy variant.
    mirror = make_actor(
        config.algo,
        dims=(5, 1),
        action_scale=torch.ones(1),
        action_bias=torch.zeros(1),
        device="cpu",
    )
    assert _linear_widths(mirror.net) == [64, 32]


def test_scalar_hidden_dim_keeps_holosoma_taper() -> None:
    from motrix_rl_builtin.fastsac.agent import FastSacAgent

    config = _compose(
        "g1-wbt-dance/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=16",
            "algo.agent.critic_hidden_dim=24",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.buffer_size=2",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
        ],
    )
    agent = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=5,
        act_dim=1,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )

    # Unset lists fall back to the scalar taper, byte-identical to the
    # pre-list networks (16 -> 16/8/4, 24 -> 24/12/6).
    assert _linear_widths(agent.actor.net) == [16, 8, 4]
    assert _linear_widths(agent.qnet.qnets[0].net) == [24, 12, 6, 5]


def test_sonic_policy_head_follows_explicit_hidden_dims() -> None:
    from motrix_rl_builtin.fastsac.factory import resolve_policy_variant

    config = _compose(
        "g1-sonic/motrix.fastsac",
        ["algo.agent.actor_hidden_dims=[32,16]"],
    )
    variant = resolve_policy_variant(config.algo.policy_variant)
    model = config.algo.variant["model"]
    act_dim = model["action_dim"]
    obs_dim = (
        model["num_history_frames"] * (3 * act_dim + 6)
        + model["num_future_frames"] * ((2 * act_dim + 6) + (24 * 3 + 6 + 2 * 3))
        + (model["num_future_frames"] * 24 + 27)
        + 3
    )

    actor = variant.build_actor(
        config.algo,
        (obs_dim, act_dim),
        torch.ones(act_dim),
        torch.zeros(act_dim),
        device="cpu",
    )

    assert _linear_widths(actor.policy_head.net) == [32, 16]
    assert actor.policy_head.fc_mu.in_features == 16

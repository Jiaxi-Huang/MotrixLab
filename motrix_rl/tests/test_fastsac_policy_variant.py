# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.fastsac.factory import configure_env_spec, make_actor, resolve_policy_variant
from motrix_rl.fastsac.sonic import SONIC_ENCODER_COUNT

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(task: str, overrides: list[str]) -> TrainConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        config = compose(config_name="train", overrides=[f"task={task}", *overrides])
    return to_typed_config(config, TrainConfig)


def test_policy_variant_registry_resolves_names() -> None:
    default = resolve_policy_variant("default")
    sonic = resolve_policy_variant("sonic")

    assert default.name == "default"
    assert sonic.name == "sonic"
    with pytest.raises(ValueError, match=r"unknown FastSAC policy variant 'missing';.*default, sonic"):
        resolve_policy_variant("missing")


def test_sonic_actor_geometry_validates_independent_temporal_dims() -> None:
    from motrix_rl.fastsac.sonic import SonicModelConfig

    config = _compose("g1-sonic/motrix.fastsac", [])
    variant = resolve_policy_variant(config.algo.policy_variant)
    model = config.algo.variant["model"]
    env_obs_dim = _sonic_obs_dim(config)

    actor = variant.build_actor(
        config.algo,
        (env_obs_dim, 29),
        torch.ones(29),
        torch.zeros(29),
        device="cpu",
    )

    assert actor.n_obs == env_obs_dim
    assert actor.config.num_future_frames == model["num_future_frames"]
    assert actor.config.num_history_frames == model["num_history_frames"]
    assert actor.config.action_dim == 29

    mismatched = SonicModelConfig(
        num_future_frames=1,
        num_history_frames=1,
        num_tokens=2,
        token_dim=4,
        fsq_levels=8,
    )
    with pytest.raises(ValueError, match="does not match configured geometry"):
        mismatched.with_env_dims(env_obs_dim - 1, 29)


def test_sonic_temporal_dims_configure_environment_independently() -> None:
    import motrix_envs  # noqa: F401 registers built-in environments
    from motrix_env_core import registry

    config = _compose(
        "g1-sonic/motrix.fastsac",
        ["algo.variant.model.num_future_frames=3", "algo.variant.model.num_history_frames=7"],
    )
    spec = configure_env_spec(config.algo, registry.resolve("g1-sonic"))

    assert spec.env_cfg.commands.motion.num_future_frames == 3
    assert spec.env_cfg.observations.policy.obs.num_history_frames == 7
    assert spec.env_cfg.observations.value.obs.num_history_frames == 7


def test_sonic_canonical_variant_builds_actor() -> None:
    config = _compose("g1-sonic/motrix.fastsac", [])
    variant = resolve_policy_variant(config.algo.policy_variant)
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    act_dim = model["action_dim"]

    actor = variant.build_actor(
        config.algo,
        (obs_dim, act_dim),
        torch.ones(act_dim),
        torch.zeros(act_dim),
        device="cpu",
    )

    assert actor.n_obs == obs_dim
    assert actor.n_act == act_dim
    assert variant.passthrough_dims(config.algo) == SONIC_ENCODER_COUNT
    torch.testing.assert_close(actor.policy_head.action_scale, torch.ones(act_dim))
    torch.testing.assert_close(actor.policy_head.action_bias, torch.zeros(act_dim))
    observations = torch.zeros(2, obs_dim)
    observations[:, -2] = 1.0
    actions, log_probs, variant_loss, metrics = variant.policy_update(actor, observations, config.algo)
    assert actions.shape == (2, act_dim)
    assert log_probs.shape == (2,)
    assert torch.isfinite(variant_loss)
    assert set(metrics) == {
        "aux_reconstruction",
        "aux_latent_alignment",
        "aux_cycle_consistency",
        "aux_loss",
    }
    mirror = make_actor(
        config.algo,
        dims=(obs_dim, act_dim),
        action_scale=torch.ones(act_dim),
        action_bias=torch.zeros(act_dim),
        device="cpu",
    )
    assert [(name, parameter.shape) for name, parameter in mirror.named_parameters()] == [
        (name, parameter.shape) for name, parameter in actor.named_parameters()
    ]


def test_sonic_variant_forwards_action_scale_and_bias() -> None:
    config = _compose(
        "g1-sonic/motrix.fastsac",
        ["algo.agent.actor_hidden_dim=16", "algo.variant.model.g1_encoder_hidden_dims=[32,16]"],
    )
    variant = resolve_policy_variant(config.algo.policy_variant)
    obs_dim = _sonic_obs_dim(config)
    act_dim = config.algo.variant["model"]["action_dim"]
    scale = torch.linspace(0.5, 2.5, act_dim)
    bias = torch.linspace(-0.25, 0.25, act_dim)

    actor = variant.build_actor(config.algo, (obs_dim, act_dim), scale, bias, device="cpu")

    # The sonic head must honor the env-derived bounds like the default actor:
    # the env declares a limit-derived Box, so dropping scale/bias would
    # silently shrink the policy's reachable joint range.
    torch.testing.assert_close(actor.action_scale, scale)
    torch.testing.assert_close(actor.action_bias, bias)


def test_sonic_split_reproduces_upstream_tokenizer_flat_order() -> None:
    """``_split`` re-chunks the feature-major terms into the upstream layout.

    Upstream builds ``command_multi_future`` as [all joint positions, all joint
    velocities] and reshapes to (F, 2J) — a documented flattening bug its
    decoders depend on. The G1 term here is emitted feature-major, so the
    reshape in ``_split`` must reproduce that exact scrambled flat order (and
    the SMPL body/wrist re-chunk) for the encoder inputs to match upstream.
    """
    from motrix_rl.fastsac.sonic import SonicActor, SonicModelConfig

    future_frames, history_frames, act_dim = 3, 2, 7
    proprioceptive = 3 * act_dim + 6
    g1_frame = 2 * act_dim + 6
    smpl_frame = 24 * 3 + 6 + 2 * 3
    obs_dim = history_frames * proprioceptive + future_frames * (g1_frame + smpl_frame) + SONIC_ENCODER_COUNT
    config = SonicModelConfig(
        num_future_frames=future_frames,
        num_history_frames=history_frames,
        num_tokens=2,
        token_dim=4,
        fsq_levels=8,
        action_dim=act_dim,
        g1_encoder_hidden_dims=(32,),
        smpl_encoder_hidden_dims=(32,),
        g1_motion_decoder_hidden_dims=(32,),
    )
    resolved = config.with_env_dims(obs_dim, act_dim)
    assert resolved.num_future_frames == future_frames
    assert resolved.num_history_frames == history_frames
    actor = SonicActor(config, hidden_dim=16, log_std_min=-5.0, log_std_max=0.0, device="cpu")

    batch = 2
    obs = torch.randn(batch, obs_dim)
    g1_start = history_frames * proprioceptive
    smpl_start = g1_start + future_frames * g1_frame
    positions = torch.arange(future_frames * act_dim, dtype=torch.float32)
    velocities = torch.arange(future_frames * act_dim, dtype=torch.float32) + 100.0
    rotations = torch.arange(future_frames * 6, dtype=torch.float32) + 200.0
    obs[:, g1_start : g1_start + future_frames * act_dim] = positions
    obs[:, g1_start + future_frames * act_dim : smpl_start - future_frames * 6] = velocities
    obs[:, smpl_start - future_frames * 6 : smpl_start] = rotations

    _, g1, _, _ = actor._split(obs)

    pv = torch.cat((positions, velocities)).reshape(future_frames, 2 * act_dim)
    rot = rotations.reshape(future_frames, 6)
    expected = torch.cat((pv, rot), dim=-1).flatten()
    torch.testing.assert_close(g1[0].flatten(), expected)
    torch.testing.assert_close(g1[1].flatten(), expected)


def _tiny_sonic_config() -> TrainConfig:
    return _compose(
        "g1-sonic/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=16",
            "algo.agent.critic_hidden_dim=16",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.v_min=-5.0",
            "algo.agent.v_max=5.0",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=2",
            "algo.agent.num_updates=1",
            "algo.agent.policy_frequency=1",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
            "algo.variant.model.num_future_frames=1",
            "algo.variant.model.num_history_frames=2",
            "algo.variant.model.num_tokens=1",
            "algo.variant.model.token_dim=4",
            "algo.variant.model.fsq_levels=8",
            "algo.variant.model.action_dim=7",
            "algo.variant.model.g1_encoder_hidden_dims=[32,16]",
            "algo.variant.model.smpl_encoder_hidden_dims=[32,16]",
            "algo.variant.model.g1_motion_decoder_hidden_dims=[32,16]",
        ],
    )


def _sonic_obs_dim(config: TrainConfig) -> int:
    model = config.algo.variant["model"]
    action_dim = model["action_dim"]
    proprioceptive = 3 * action_dim + 6
    g1_reference = 2 * action_dim + 6
    smpl_reference = 24 * 3 + 6 + 2 * 3
    return (
        model["num_history_frames"] * proprioceptive + model["num_future_frames"] * (g1_reference + smpl_reference) + 2
    )


def _packed(obs_dim: int, count: int) -> torch.Tensor:
    observations = torch.randn(count, obs_dim)
    observations[:, -2] = 1.0
    observations[:, -1] = 0.0
    return observations


def test_agent_weights_and_logs_sonic_auxiliary_losses() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _tiny_sonic_config()
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    agent = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    agent.rb.extend(
        _packed(obs_dim, 1),
        torch.zeros(1, 3),
        torch.zeros(1, model["action_dim"]),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
    )
    agent.rb.extend(
        _packed(obs_dim, 1),
        torch.zeros(1, 3),
        torch.zeros(1, model["action_dim"]),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
    )

    metrics = agent.update(1)

    assert set(metrics) >= {"aux_reconstruction", "aux_latent_alignment", "aux_cycle_consistency", "aux_loss"}
    assert torch.isfinite(metrics["aux_reconstruction"])
    assert torch.isfinite(metrics["aux_latent_alignment"])
    assert torch.isfinite(metrics["aux_cycle_consistency"])
    assert torch.isfinite(metrics["aux_loss"])


def test_default_variant_has_no_auxiliary_metrics() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _compose(
        "g1-wbt-dance/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=8",
            "algo.agent.critic_hidden_dim=8",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.v_min=-5.0",
            "algo.agent.v_max=5.0",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=2",
            "algo.agent.num_updates=1",
            "algo.agent.policy_frequency=1",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
        ],
    )
    assert config.algo.policy_variant == "default"
    assert config.algo.variant == {}
    agent = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=7,
        act_dim=3,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    agent.rb.extend(
        torch.randn(1, 5),
        torch.randn(1, 7),
        torch.zeros(1, 3),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
    )
    agent.rb.extend(
        torch.randn(1, 5),
        torch.randn(1, 7),
        torch.zeros(1, 3),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
    )

    metrics = agent.update(1)

    assert not any(name.startswith("aux_") for name in metrics)

    legacy_checkpoint = agent.state_dict()
    legacy_checkpoint.pop("policy_variant")
    legacy_checkpoint.pop("policy_variant_metadata")
    restored = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=7,
        act_dim=3,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(legacy_checkpoint, load_optimizers=False)


def test_sonic_checkpoint_records_variant_metadata() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _tiny_sonic_config()
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    agent = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    checkpoint = agent.state_dict()
    assert checkpoint["policy_variant"] == "sonic"
    assert checkpoint["policy_variant_metadata"] == {
        **config.algo.variant,
        "actor_hidden_dim": config.algo.agent.actor_hidden_dim,
        "log_std_min": config.algo.agent.log_std_min,
        "log_std_max": config.algo.agent.log_std_max,
        "use_layer_norm": config.algo.agent.use_layer_norm,
    }


def test_sonic_checkpoint_load_validates_policy_variant() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _tiny_sonic_config()
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    agent = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    checkpoint = agent.state_dict()
    restored = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )

    restored.load_state_dict(checkpoint, load_optimizers=False)

    mismatched = dict(checkpoint, policy_variant="default")
    with pytest.raises(ValueError, match="does not match configured variant"):
        restored.load_state_dict(mismatched, load_optimizers=False)
    legacy_sonic = dict(checkpoint, sonic={"enabled": True})
    legacy_sonic.pop("policy_variant")
    legacy_sonic.pop("policy_variant_metadata")
    with pytest.raises(ValueError, match="pre-PolicyVariant SONIC format"):
        restored.load_state_dict(legacy_sonic, load_optimizers=False)

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import torch
from motrix_rl_builtin.fastsac.buffer import EmpiricalNormalization
from motrix_rl_builtin.fastsac.sonic import SonicActor, SonicModelConfig


def _packed(cfg: SonicModelConfig, batch: int = 4) -> torch.Tensor:
    g1 = torch.randn(batch, cfg.g1_input_dim)
    smpl = torch.randn(batch, cfg.smpl_input_dim)
    teleop = torch.randn(batch, cfg.teleop_input_dim)
    obs = torch.randn(batch, cfg.actor_obs_dim)
    # Reference terms use the manager's term-major layout: command/body terms
    # first, then the 6D orientation/wrist terms.
    index = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])[:batch]
    return torch.cat((obs, g1, smpl, teleop, index), dim=-1)


def test_sonic_actor_observation_width_is_derived_from_action_width() -> None:
    cfg = SonicModelConfig(
        num_future_frames=2,
        num_history_frames=3,
        num_tokens=1,
        token_dim=4,
        fsq_levels=8,
        action_dim=7,
    )
    assert cfg.actor_obs_frame_dim == 27
    assert cfg.actor_obs_dim == 81
    assert cfg.g1_frame_dim == 20
    assert cfg.g1_input_dim == 40
    assert cfg.smpl_frame_dim == 84
    # Upstream teleop layout: 12 leg positions + velocities per future frame,
    # plus the static current-frame tail (9 + 12 + 6).
    assert cfg.teleop_input_dim == 24 * 2 + 27
    assert cfg.packed_obs_dim == 81 + 40 + 168 + 75 + 3

    more_future = replace(cfg, num_future_frames=4)
    assert more_future.actor_obs_dim == cfg.actor_obs_dim
    assert more_future.g1_input_dim == 2 * cfg.g1_input_dim
    assert more_future.smpl_input_dim == 2 * cfg.smpl_input_dim
    assert more_future.teleop_input_dim == 24 * 4 + 27

    more_history = replace(cfg, num_history_frames=5)
    assert more_history.actor_obs_dim == 5 * cfg.actor_obs_frame_dim
    assert more_history.g1_input_dim == cfg.g1_input_dim
    assert more_history.smpl_input_dim == cfg.smpl_input_dim
    assert more_history.teleop_input_dim == cfg.teleop_input_dim


def test_sonic_actor_forward_and_auxiliary_backward() -> None:
    cfg = SonicModelConfig(
        num_future_frames=4,
        num_history_frames=3,
        num_tokens=1,
        token_dim=8,
        fsq_levels=16,
        action_dim=29,
        g1_encoder_hidden_dims=(256, 128),
        smpl_encoder_hidden_dims=(256, 128),
        teleop_encoder_hidden_dims=(256, 128),
        g1_motion_decoder_hidden_dims=(256, 128),
    )
    actor = SonicActor(cfg, hidden_dims=(128, 64, 32), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    observations = _packed(cfg)
    # Multi-hot rows like the sampler emits: a tri-hot row (smpl co-activates
    # g1 and teleop) drives the g1-teleop / teleop-smpl alignment losses.
    observations[:, -3:] = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    actions, logp, aux = actor.get_actions_and_log_probs_with_aux(observations)
    assert actions.shape == (4, cfg.action_dim)
    assert logp.shape == (4,)
    assert torch.isfinite(actions).all() and torch.isfinite(logp).all()
    assert set(aux) == {
        "g1_recon",
        "g1_smpl_latent",
        "reencoded_smpl_g1_latent",
        "g1_teleop_latent",
        "teleop_smpl_latent",
    }
    assert all(torch.isfinite(loss) for loss in aux.values())
    assert aux["g1_teleop_latent"] > 0.0
    assert aux["teleop_smpl_latent"] > 0.0
    (actions.square().mean() + sum(aux.values())).backward()
    assert actor.backbone.encoders["g1"].module[0].weight.grad is not None
    assert actor.backbone.encoders["smpl"].module[0].weight.grad is not None
    assert actor.backbone.encoders["teleop"].module[0].weight.grad is not None


def test_sonic_backbone_routes_multi_hot_with_smpl_priority() -> None:
    from motrix_rl_builtin.fastsac.sonic import SonicBackbone

    cfg = SonicModelConfig(
        num_future_frames=2, num_history_frames=3, num_tokens=1, token_dim=4, fsq_levels=8, action_dim=7
    )
    backbone = SonicBackbone(cfg, device="cpu")
    batch = 5
    actor_obs = torch.zeros(batch, cfg.actor_obs_dim)
    g1 = torch.randn(batch, cfg.num_future_frames, cfg.g1_frame_dim)
    smpl = torch.randn(batch, cfg.num_future_frames, cfg.smpl_frame_dim)
    teleop = torch.randn(batch, cfg.teleop_input_dim)
    # Column order [g1, teleop, smpl]: g1-only, teleop-only, smpl-native
    # (multi-hot with g1), tri-hot, and g1+teleop without smpl.
    index = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    output = backbone(actor_obs, g1, smpl, teleop, index, compute_auxiliary=False)

    expected_g1 = backbone.quantizer(backbone._encode("g1", g1) * index[:, 0, None, None])[0]
    expected_teleop = backbone.quantizer(backbone._encode("teleop", teleop) * index[:, 1, None, None])[0]
    expected_smpl = backbone.quantizer(backbone._encode("smpl", smpl) * index[:, 2, None, None])[0]
    assert torch.equal(output.selected_tokens[0], expected_g1[0])
    assert torch.equal(output.selected_tokens[1], expected_teleop[1])
    # smpl overwrites teleop overwrites g1 on multi-hot rows.
    assert torch.equal(output.selected_tokens[2], expected_smpl[2])
    assert torch.equal(output.selected_tokens[3], expected_smpl[3])
    assert torch.equal(output.selected_tokens[4], expected_teleop[4])


def _maskless_packed(cfg: SonicModelConfig, batch: int = 4) -> torch.Tensor:
    return torch.cat(
        (
            torch.randn(batch, cfg.actor_obs_dim),
            torch.randn(batch, cfg.g1_input_dim),
            torch.randn(batch, cfg.smpl_input_dim),
            torch.randn(batch, cfg.teleop_input_dim),
        ),
        dim=-1,
    )


def test_sonic_maskless_model_routes_every_row_through_g1() -> None:
    """``encoder_index=False`` drops the mask input and defaults to g1 routing.

    The maskless forward must behave exactly like a masked batch of g1-native
    rows: the packed observation ends after the teleop block, the mask slice
    is absent, and the smpl/teleop encoders stay zeroed out.
    """
    from motrix_rl_builtin.fastsac.sonic import SonicBackbone

    cfg = SonicModelConfig(
        num_future_frames=2,
        num_history_frames=3,
        num_tokens=1,
        token_dim=4,
        fsq_levels=8,
        action_dim=7,
        encoder_index=False,
    )
    assert cfg.packed_obs_dim == (cfg.actor_obs_dim + cfg.g1_input_dim + cfg.smpl_input_dim + cfg.teleop_input_dim)

    actor = SonicActor(cfg, hidden_dims=(32, 16, 8), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    actor_obs, g1, smpl, teleop, index = actor._split(_maskless_packed(cfg, 4))
    assert index is None
    actions, logp, aux = actor.get_actions_and_log_probs_with_aux(_maskless_packed(cfg, 4))
    assert actions.shape == (4, cfg.action_dim)
    assert torch.isfinite(actions).all() and torch.isfinite(logp).all()
    # The cross-encoder alignment losses reduce to zero rows; g1
    # reconstruction still trains.
    assert aux["g1_smpl_latent"].item() == 0.0
    assert aux["g1_teleop_latent"].item() == 0.0
    assert aux["teleop_smpl_latent"].item() == 0.0
    assert aux["reencoded_smpl_g1_latent"].item() == 0.0

    # Maskless routing equals an explicit all-g1 mask on the same batch.
    torch.manual_seed(0)
    backbone = SonicBackbone(cfg, device="cpu")
    backbone.eval()
    observations = _maskless_packed(cfg, 4)
    actor_obs, g1, smpl, teleop, _ = actor._split(observations)
    with torch.no_grad():
        maskless = backbone(actor_obs, g1, smpl, teleop, None)
        explicit = backbone(actor_obs, g1, smpl, teleop, torch.tensor([[1.0, 0.0, 0.0]] * 4))
    torch.testing.assert_close(maskless.selected_tokens, explicit.selected_tokens)


def test_sonic_deterministic_action_is_bounded() -> None:
    cfg = SonicModelConfig(
        num_future_frames=2, num_history_frames=3, num_tokens=1, token_dim=4, fsq_levels=8, action_dim=7
    )
    actor = SonicActor(cfg, hidden_dims=(32, 16, 8), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    actions = actor.explore(_packed(cfg, 2), deterministic=True)
    assert actions.shape == (2, cfg.action_dim)
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)


def test_sonic_normalization_preserves_encoder_selector() -> None:
    cfg = SonicModelConfig(
        num_future_frames=2, num_history_frames=3, num_tokens=1, token_dim=4, fsq_levels=8, action_dim=7
    )
    observations = _packed(cfg)
    normalizer = EmpiricalNormalization(cfg.packed_obs_dim, device="cpu", passthrough_dims=2)

    normalized = normalizer(observations, update=True)

    torch.testing.assert_close(normalized[:, -2:], observations[:, -2:])
    actor = SonicActor(cfg, hidden_dims=(32, 16, 8), log_std_min=-5.0, log_std_max=0.0, device="cpu")
    assert actor.explore(normalized).shape == (4, cfg.action_dim)


def test_sonic_ignores_non_finite_inactive_reference() -> None:
    cfg = SonicModelConfig(
        num_future_frames=2, num_history_frames=3, num_tokens=1, token_dim=4, fsq_levels=8, action_dim=7
    )
    observations = _packed(cfg)
    smpl_start = cfg.actor_obs_dim + cfg.g1_input_dim
    observations[:, smpl_start : smpl_start + cfg.smpl_input_dim] = float("nan")
    observations[:, -3:] = torch.tensor((1.0, 0.0, 0.0))
    actor = SonicActor(cfg, hidden_dims=(32, 16, 8), log_std_min=-5.0, log_std_max=0.0, device="cpu")

    actions, logp, auxiliary = actor.get_actions_and_log_probs_with_aux(observations)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(logp).all()
    assert all(torch.isfinite(loss) for loss in auxiliary.values())


def test_sonic_backbone_is_excluded_from_actor_weight_decay() -> None:
    """The tokenizer backbone trains without weight decay; the head keeps it.

    The SONIC backbone mirrors the upstream ATM, which is trained with Adam
    without weight decay: decaying encoder/decoder weights shrinks latents over
    long runs and surfaces as late-training aux-loss spikes. The actor
    optimizer must therefore split its parameter groups instead of decaying
    every actor parameter.
    """
    from types import SimpleNamespace

    from motrix_rl_builtin.fastsac.agent import FastSacAgent

    model = dict(
        num_future_frames=2,
        num_history_frames=3,
        num_tokens=1,
        token_dim=4,
        fsq_levels=8,
        action_dim=7,
    )
    cfg = SonicModelConfig(**model)
    agent_cfg = SimpleNamespace(
        actor_hidden_dim=16,
        actor_hidden_dims=None,
        critic_hidden_dim=16,
        critic_hidden_dims=None,
        num_q_networks=2,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        alpha_learning_rate=1e-3,
        weight_decay=1e-3,
        max_grad_norm=0.0,
        use_layer_norm=False,
        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,
        num_atoms=5,
        v_min=-20.0,
        v_max=20.0,
        gamma=0.97,
        tau=0.125,
        alpha_init=0.001,
        use_autotune=True,
        target_entropy_ratio=0.0,
        buffer_size=8,
        num_steps=1,
        batch_size=4,
        learning_starts=1,
        policy_frequency=2,
        num_updates=1,
        obs_normalization=True,
        compile=False,
        amp=False,
        amp_dtype="bf16",
        device=None,
    )
    agent = FastSacAgent(
        obs_dim=cfg.packed_obs_dim,
        critic_obs_dim=9,
        act_dim=cfg.action_dim,
        num_envs=4,
        cfg=SimpleNamespace(
            agent=agent_cfg,
            policy_variant="sonic",
            variant={"model": model, "auxiliary": {"g1_recon": 0.01}},
        ),
        device=torch.device("cpu"),
    )

    groups = agent.actor_optimizer.param_groups

    assert len(groups) == 2
    decayed, backbone = groups
    assert backbone["weight_decay"] == 0.0
    assert decayed["weight_decay"] == 1e-3
    backbone_ids = {id(p) for p in agent.actor.backbone.parameters()}
    assert {id(p) for p in backbone["params"]} == backbone_ids
    assert not backbone_ids & {id(p) for p in decayed["params"]}
    all_ids = {id(p) for p in agent.actor.parameters()}
    assert all_ids == backbone_ids | {id(p) for p in decayed["params"]}

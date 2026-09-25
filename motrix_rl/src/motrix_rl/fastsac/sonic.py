# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SONIC actor used by the Motrix FastSAC provider.

The module deliberately keeps the environment-facing contract small: a SONIC
actor consumes the packed policy observation emitted by the manager and
returns the same ``(actions, log_probs)`` pair as the generic FastSAC actor.
The G1/SMPL tokenizer mirrors the upstream SONIC graph while the training policy
head is the standard FastSAC actor sized by ``algo.agent.actor_hidden_dim``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from motrix_rl.fastsac.api import policy_variant_registry
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.networks import Actor

SONIC_VECTOR_DIM = 3
SONIC_ROTATION_REPRESENTATION_DIM = 2 * SONIC_VECTOR_DIM
SONIC_ACTOR_JOINT_FEATURES = 3
SONIC_ENCODER_COUNT = 2
SONIC_SMPL_JOINT_COUNT = 24
SONIC_SMPL_END_EFFECTOR_COUNT = 2


@dataclass(frozen=True)
class SonicModelConfig:
    num_future_frames: int
    num_history_frames: int
    num_tokens: int
    token_dim: int
    fsq_levels: int
    action_dim: int = 29
    g1_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    smpl_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)
    g1_motion_decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 512)

    @property
    def actor_obs_dim(self) -> int:
        """Proprioceptive history width implied by this model configuration."""
        return self.num_history_frames * self.actor_obs_frame_dim

    @property
    def actor_obs_frame_dim(self) -> int:
        """Width of one proprioceptive history frame."""
        return SONIC_ACTOR_JOINT_FEATURES * self.action_dim + 2 * SONIC_VECTOR_DIM

    @property
    def g1_frame_dim(self) -> int:
        return 2 * self.action_dim + SONIC_ROTATION_REPRESENTATION_DIM

    @property
    def smpl_end_effector_dim(self) -> int:
        return SONIC_SMPL_END_EFFECTOR_COUNT * SONIC_VECTOR_DIM

    @property
    def smpl_frame_dim(self) -> int:
        return (
            SONIC_SMPL_JOINT_COUNT * SONIC_VECTOR_DIM + SONIC_ROTATION_REPRESENTATION_DIM + self.smpl_end_effector_dim
        )

    @property
    def g1_input_dim(self) -> int:
        return self.num_future_frames * self.g1_frame_dim

    @property
    def smpl_input_dim(self) -> int:
        return self.num_future_frames * self.smpl_frame_dim

    @property
    def packed_obs_dim(self) -> int:
        return self.actor_obs_dim + self.g1_input_dim + self.smpl_input_dim + SONIC_ENCODER_COUNT

    @property
    def token_total_dim(self) -> int:
        return self.num_tokens * self.token_dim

    @classmethod
    def from_mapping(cls, values: dict) -> SonicModelConfig:
        """Build from the PolicyVariant mapping while preserving tuples."""
        allowed = {
            "num_future_frames",
            "num_history_frames",
            "num_tokens",
            "token_dim",
            "fsq_levels",
            "action_dim",
            "g1_encoder_hidden_dims",
            "smpl_encoder_hidden_dims",
            "g1_motion_decoder_hidden_dims",
        }
        data = {key: values[key] for key in allowed if key in values}
        for key in (
            "g1_encoder_hidden_dims",
            "smpl_encoder_hidden_dims",
            "g1_motion_decoder_hidden_dims",
        ):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    def with_env_dims(self, obs_dim: int, act_dim: int) -> SonicModelConfig:
        """Bind the action width and validate the configured temporal geometry."""
        model = replace(self, action_dim=act_dim)
        if model.packed_obs_dim != obs_dim:
            raise ValueError(
                f"SONIC packed actor width {obs_dim} does not match configured geometry "
                f"{model.packed_obs_dim} (num_history_frames={model.num_history_frames}, "
                f"num_future_frames={model.num_future_frames}, action_dim={act_dim})"
            )
        return model


class SonicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, device=None):
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for i, (source, target) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(source, target, device=device))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.module = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


@dataclass
class SonicBackboneOutput:
    selected_tokens: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor]


class SonicBackbone(nn.Module):
    """G1/SMPL encoders, FSQ, and motion decoder used by the FastSAC actor."""

    def __init__(self, config: SonicModelConfig, device=None):
        super().__init__()
        from vector_quantize_pytorch import FSQ

        self.config = config
        self.encoders = nn.ModuleDict(
            {
                "g1": SonicMLP(
                    config.g1_input_dim, config.g1_encoder_hidden_dims, config.token_total_dim, device=device
                ),
                "smpl": SonicMLP(
                    config.smpl_input_dim, config.smpl_encoder_hidden_dims, config.token_total_dim, device=device
                ),
            }
        )
        self.quantizer = FSQ(levels=[config.fsq_levels] * config.token_dim, return_indices=False).to(device)

        self.g1_kin_decoder = SonicMLP(
            config.token_total_dim,
            config.g1_motion_decoder_hidden_dims,
            config.g1_input_dim,
            device=device,
        )

    def _encode(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        batch = reference.shape[0]
        latent = self.encoders[name](reference.flatten(start_dim=1))
        return latent.reshape(batch, self.config.num_tokens, self.config.token_dim)

    def forward(
        self,
        actor_obs: torch.Tensor,
        g1_reference: torch.Tensor,
        smpl_reference: torch.Tensor,
        encoder_index: torch.Tensor,
        *,
        compute_auxiliary: bool = False,
    ) -> SonicBackboneOutput:
        cfg = self.config
        if actor_obs.ndim != 2 or actor_obs.shape[-1] != cfg.actor_obs_dim:
            raise ValueError(f"SONIC actor observation must have width {cfg.actor_obs_dim}")
        if tuple(g1_reference.shape[1:]) != (cfg.num_future_frames, cfg.g1_frame_dim):
            raise ValueError("invalid SONIC G1 reference shape")
        if tuple(smpl_reference.shape[1:]) != (cfg.num_future_frames, cfg.smpl_frame_dim):
            raise ValueError("invalid SONIC SMPL reference shape")
        if encoder_index.shape != (actor_obs.shape[0], SONIC_ENCODER_COUNT):
            raise ValueError(f"SONIC encoder_index must have shape (B, {SONIC_ENCODER_COUNT})")
        g1_mask = encoder_index[:, 0].bool()
        smpl_mask = encoder_index[:, 1].bool()
        g1_required = torch.ones_like(g1_mask) if compute_auxiliary else g1_mask
        g1_input = torch.where(g1_required[:, None, None], g1_reference, torch.zeros_like(g1_reference))
        smpl_input = torch.where(smpl_mask[:, None, None], smpl_reference, torch.zeros_like(smpl_reference))
        g1_encoded = self._encode("g1", g1_input)
        smpl_encoded = self._encode("smpl", smpl_input)
        g1_latent = g1_encoded * g1_mask[:, None, None]
        smpl_latent = smpl_encoded * smpl_mask[:, None, None]
        g1_tokens = self.quantizer(g1_latent)[0].contiguous()
        smpl_tokens = self.quantizer(smpl_latent)[0].contiguous()
        selected = torch.where(smpl_mask[:, None, None], smpl_tokens, g1_tokens)
        losses: dict[str, torch.Tensor] = {}
        if compute_auxiliary:
            reconstruction = self.g1_kin_decoder(selected.flatten(start_dim=1)).reshape(
                -1, cfg.num_future_frames, cfg.g1_frame_dim
            )
            recon_loss = F.mse_loss(reconstruction, g1_reference)
            smpl_weight = smpl_mask.to(g1_encoded.dtype)
            smpl_count = smpl_weight.sum().clamp_min(1.0)
            align_per_sample = (smpl_encoded - g1_encoded).square().mean(dim=(1, 2))
            cycle_per_sample = (self._encode("g1", reconstruction) - g1_encoded).square().mean(dim=(1, 2))
            align = (align_per_sample * smpl_weight).sum() / smpl_count
            cycle = (cycle_per_sample * smpl_weight).sum() / smpl_count
            losses = {
                "reconstruction": recon_loss,
                "latent_alignment": align,
                "cycle_consistency": cycle,
            }
        return SonicBackboneOutput(selected, losses)


class SonicActor(nn.Module):
    """SONIC tokenizer with the standard FastSAC policy head."""

    def __init__(
        self,
        config: SonicModelConfig,
        *,
        hidden_dim: int,
        log_std_min,
        log_std_max,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        action_scale=None,
        action_bias=None,
        device,
    ):
        super().__init__()
        self.config = config
        self.backbone = SonicBackbone(config, device=device)
        self.policy_head = Actor(
            n_obs=config.token_total_dim + config.actor_obs_dim,
            n_act=config.action_dim,
            hidden_dim=hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            device=device,
        )
        self.n_obs = config.packed_obs_dim
        self.n_act = config.action_dim

    @property
    def action_scale(self) -> torch.Tensor:
        return self.policy_head.action_scale

    @property
    def action_bias(self) -> torch.Tensor:
        return self.policy_head.action_bias

    def _split(self, obs):
        cfg = self.config
        a = obs[:, : cfg.actor_obs_dim]
        g1_start = cfg.actor_obs_dim
        g1_end = g1_start + cfg.g1_input_dim
        smpl_end = g1_end + cfg.smpl_input_dim
        g1_flat = obs[:, g1_start:g1_end]
        smpl_flat = obs[:, g1_end:smpl_end]
        # The G1 command term is emitted feature-major ([all joint positions, all
        # joint velocities, all rotations]); this reshape re-chunks it into the
        # per-frame (F, 64) layout the upstream tokenizer consumes. That layout
        # itself interleaves frames — upstream builds command_multi_future as
        # [all P, all V] and reshapes to (F, 2J), a flattening bug it documents
        # in trl/losses/token_losses.py ("temporal axis is incorrectly
        # flattened") and that its decoders depend on. The permutation here
        # reproduces the upstream flat order exactly (G1 frames stride 5 = 0.1
        # s, SMPL stride 1 = 0.02 s), so do NOT "fix" this reshape without
        # porting upstream with it. The SMPL term is per-frame by construction;
        # only its wrist tail needs the same re-chunking.
        g1_command_frame_dim = cfg.g1_frame_dim - SONIC_ROTATION_REPRESENTATION_DIM
        g1_command_width = cfg.num_future_frames * g1_command_frame_dim
        g1 = torch.cat(
            (
                g1_flat[:, :g1_command_width].reshape(-1, cfg.num_future_frames, g1_command_frame_dim),
                g1_flat[:, g1_command_width:].reshape(-1, cfg.num_future_frames, SONIC_ROTATION_REPRESENTATION_DIM),
            ),
            dim=-1,
        )
        smpl_body_frame_dim = cfg.smpl_frame_dim - cfg.smpl_end_effector_dim
        smpl_body_width = cfg.num_future_frames * smpl_body_frame_dim
        smpl = torch.cat(
            (
                smpl_flat[:, :smpl_body_width].reshape(-1, cfg.num_future_frames, smpl_body_frame_dim),
                smpl_flat[:, smpl_body_width:].reshape(-1, cfg.num_future_frames, cfg.smpl_end_effector_dim),
            ),
            dim=-1,
        )
        index = obs[:, smpl_end : smpl_end + SONIC_ENCODER_COUNT]
        return a, g1, smpl, index

    def forward(self, obs):
        control_input, _ = self._policy_input(obs, compute_auxiliary=self.training)
        actions, mean, _log_std = self.policy_head(control_input)
        return actions, mean, _log_std

    def get_actions_and_log_probs(self, obs):
        control_input, _ = self._policy_input(obs, compute_auxiliary=False)
        return self.policy_head.get_actions_and_log_probs(control_input)

    def get_actions_and_log_probs_with_aux(self, obs):
        control_input, auxiliary = self._policy_input(obs, compute_auxiliary=True)
        actions, logp = self.policy_head.get_actions_and_log_probs(control_input)
        return actions, logp, auxiliary

    @torch.no_grad()
    def explore(self, obs, deterministic=False):
        control_input, _ = self._policy_input(obs, compute_auxiliary=False)
        return self.policy_head.explore(control_input, deterministic=deterministic)

    def _policy_input(self, obs, *, compute_auxiliary: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actor_obs, g1, smpl, index = self._split(obs)
        output = self.backbone(actor_obs, g1, smpl, index, compute_auxiliary=compute_auxiliary)
        control_input = torch.cat((actor_obs, output.selected_tokens.flatten(start_dim=1)), dim=-1)
        return control_input, output.auxiliary_losses


class SonicPolicyVariant:
    name = "sonic"

    def configure_env_cfg(self, cfg: FastSacCfg, env_cfg: Any) -> None:
        model = self._model(cfg.variant)
        env_cfg.commands.motion.num_future_frames = model.num_future_frames
        env_cfg.observations.policy.obs.num_history_frames = model.num_history_frames
        env_cfg.observations.value.obs.num_history_frames = model.num_history_frames

    def _model(self, values: Mapping[str, Any]) -> SonicModelConfig:
        model_values = values.get("model")
        if not isinstance(model_values, Mapping):
            raise ValueError("SONIC policy variant requires a 'model' mapping in algo.variant")
        return SonicModelConfig.from_mapping(dict(model_values))

    def build_actor(
        self,
        cfg: FastSacCfg,
        dims: tuple[int, int],
        action_scale,
        action_bias,
        device,
    ):
        values = cfg.variant
        obs_dim, act_dim = dims
        model = self._model(values).with_env_dims(obs_dim, act_dim)
        actor = SonicActor(
            model,
            hidden_dim=cfg.agent.actor_hidden_dim,
            log_std_min=cfg.agent.log_std_min,
            log_std_max=cfg.agent.log_std_max,
            use_tanh=cfg.agent.use_tanh,
            use_layer_norm=cfg.agent.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            device=device,
        )
        return actor

    def passthrough_dims(self, cfg: FastSacCfg) -> int:
        return SONIC_ENCODER_COUNT

    def _aux_loss_weights(self, cfg: FastSacCfg) -> dict[str, float]:
        values = cfg.variant.get("auxiliary", {})
        if not isinstance(values, Mapping):
            raise ValueError("SONIC policy variant 'auxiliary' must be a mapping")
        return {name: float(value) for name, value in values.items()}

    def policy_update(self, actor, observations, cfg: FastSacCfg):
        actions, log_probs, auxiliary = actor.get_actions_and_log_probs_with_aux(observations)
        weights = self._aux_loss_weights(cfg)
        auxiliary_loss = log_probs.new_zeros(())
        for name, value in auxiliary.items():
            if name not in weights:
                raise ValueError(f"SONIC actor returned unweighted auxiliary loss {name!r}")
            auxiliary_loss = auxiliary_loss + weights[name] * value
        metrics = {f"aux_{name}": value.detach().float() for name, value in auxiliary.items() if value.numel() == 1}
        if auxiliary:
            metrics["aux_loss"] = auxiliary_loss.detach().float()
        return actions, log_probs, auxiliary_loss, metrics

    def checkpoint_metadata(self, cfg: FastSacCfg) -> dict[str, Any]:
        return {
            **cfg.variant,
            "actor_hidden_dim": cfg.agent.actor_hidden_dim,
            "log_std_min": cfg.agent.log_std_min,
            "log_std_max": cfg.agent.log_std_max,
            "use_layer_norm": cfg.agent.use_layer_norm,
        }

    def validate_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if isinstance(checkpoint.get("sonic"), Mapping):
            raise ValueError(
                "FastSAC checkpoint uses the pre-PolicyVariant SONIC format; "
                "retrain it with the current FastSAC policy variant"
            )


policy_variant_registry.register(SonicPolicyVariant.name, SonicPolicyVariant())


__all__ = [
    "SonicActor",
    "SonicBackbone",
    "SonicModelConfig",
    "SonicPolicyVariant",
]

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Neutral policy-variant extension contract for FastSAC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from motrix_rl.fastsac.config import FastSacCfg

if TYPE_CHECKING:
    import torch
    from torch import nn


class PolicyVariant(Protocol):
    """Build and describe an actor that follows the frozen FastSAC actor protocol."""

    name: str

    def configure_env_cfg(self, cfg: FastSacCfg, env_cfg: Any) -> None:
        """Apply variant-owned observation contract values to one environment config."""

    def build_actor(
        self,
        cfg: FastSacCfg,
        dims: tuple[int, int],
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        device: torch.device | str,
    ) -> nn.Module:
        """Return an actor for ``(obs_dim, act_dim)``.

        Parameter names, order and shapes must be identical across supported
        devices so learner and collector can exchange a flat parameter vector.
        """

    def passthrough_dims(self, cfg: FastSacCfg) -> int:
        """Return trailing observation dimensions bypassing normalization."""

    def policy_update(
        self,
        actor: nn.Module,
        observations: torch.Tensor,
        cfg: FastSacCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return actions, log-probabilities, extra actor loss and metrics."""

    def checkpoint_metadata(self, cfg: FastSacCfg) -> dict[str, Any]:
        """Return variant metadata recorded with the checkpoint."""

    def validate_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Validate variant-owned checkpoint compatibility before loading."""


class PolicyVariantRegistry:
    def __init__(self) -> None:
        self._variants: dict[str, PolicyVariant] = {}

    def register(self, name: str, variant: PolicyVariant) -> None:
        self._variants[name] = variant

    def resolve(self, name: str) -> PolicyVariant:
        try:
            return self._variants[name]
        except KeyError as exc:
            expected = ", ".join(sorted(self._variants)) or "none"
            raise ValueError(f"unknown FastSAC policy variant {name!r}; expected one of: {expected}") from exc


policy_variant_registry = PolicyVariantRegistry()


class DefaultPolicyVariant:
    name = "default"

    def configure_env_cfg(self, cfg: FastSacCfg, env_cfg: Any) -> None:
        del cfg, env_cfg

    def build_actor(
        self,
        cfg: FastSacCfg,
        dims: tuple[int, int],
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        device: torch.device | str,
    ) -> nn.Module:
        from motrix_rl.fastsac.networks import Actor

        obs_dim, act_dim = dims
        return Actor(
            n_obs=obs_dim,
            n_act=act_dim,
            hidden_dim=cfg.agent.actor_hidden_dim,
            log_std_max=cfg.agent.log_std_max,
            log_std_min=cfg.agent.log_std_min,
            use_tanh=cfg.agent.use_tanh,
            use_layer_norm=cfg.agent.use_layer_norm,
            action_scale=action_scale,
            action_bias=action_bias,
            device=device,
        )

    def passthrough_dims(self, cfg: FastSacCfg) -> int:
        return 0

    def policy_update(
        self,
        actor: nn.Module,
        observations: torch.Tensor,
        cfg: FastSacCfg,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del cfg
        actions, log_probs = actor.get_actions_and_log_probs(observations)
        return actions, log_probs, log_probs.new_zeros(()), {}

    def checkpoint_metadata(self, cfg: FastSacCfg) -> dict[str, Any]:
        return {}

    def validate_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        del checkpoint


policy_variant_registry.register(DefaultPolicyVariant.name, DefaultPolicyVariant())

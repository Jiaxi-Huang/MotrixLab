# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic world-model agent (motrix.supermimic)."""

from motrix_rl.contracts import AgentProvider, TrainerContext
from motrix_rl_builtin.supermimic.config import SuperMimicCfg


class SuperMimicProvider(AgentProvider[SuperMimicCfg]):
    config_type = SuperMimicCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "supermimic"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(self, context: TrainerContext[SuperMimicCfg]) -> "Trainer":  # noqa: F821
        from motrix_rl_builtin.supermimic.trainer import Trainer

        return Trainer(context=context)

    def total_env_steps(self, config: SuperMimicCfg, num_envs: int) -> int:
        steps_per_iteration = num_envs * max(round(1.0 / config.agent.utd), 1)
        return config.trainer.num_learning_iterations * steps_per_iteration


__all__ = ["SuperMimicProvider"]

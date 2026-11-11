# Copyright Motrix Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Agent provider for the APG world-model/BPTT algorithm (``motrix.apg``)."""

from motrix_rl.contracts import AgentProvider, TrainerBase, TrainerContext
from motrix_rl_builtin.apg.config import ApgCfg


class ApgProvider(AgentProvider[ApgCfg]):
    config_type = ApgCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "apg"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(self, context: TrainerContext[ApgCfg]) -> TrainerBase:
        from motrix_rl_builtin.apg.train import Trainer

        return Trainer(context=context)

    def total_env_steps(self, config: ApgCfg, num_envs: int) -> int:
        return int(config.trainer.iterations / config.trainer.utd * num_envs)

    def create_policy_exporter(self):
        from motrix_rl_builtin.apg.export import ApgOnnxExporter

        return ApgOnnxExporter()

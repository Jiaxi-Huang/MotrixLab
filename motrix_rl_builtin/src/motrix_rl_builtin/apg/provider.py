# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl.contracts import AgentProvider
from motrix_rl_builtin.apg.config import ApgCfg


class ApgProvider(AgentProvider[ApgCfg]):
    config_type = ApgCfg

    @property
    def train_backend(self):
        return "torch"

    @property
    def agent_name(self):
        return "apg"

    @property
    def checkpoint_format(self):
        return "pt"

    def create_trainer(self, context):
        from motrix_rl_builtin.apg.train import Trainer

        return Trainer(context=context)

    def total_env_steps(self, config, num_envs):
        return config.trainer.num_learning_iterations * num_envs

    def create_policy_exporter(self):
        from motrix_rl_builtin.apg.export import ApgOnnxExporter

        return ApgOnnxExporter()

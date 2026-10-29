# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG world-model agent (motrix.apg)."""

import math

from motrix_rl.contracts import AgentProvider, TrainerContext
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

    def create_trainer(self, context: TrainerContext[ApgCfg]) -> "Trainer":  # noqa: F821
        if context.rl_cfg.asynchronous:
            from motrix_rl_builtin.apg.trainer import Trainer
        else:
            from motrix_rl_builtin.apg.sync.train import Trainer

        return Trainer(context=context)

    def total_env_steps(self, config: ApgCfg, num_envs: int) -> int:
        if config.asynchronous:
            # The collector earns update credit in whole blocks: strict UTD
            # over the warmup, rounded up to the block size.
            warmup = max(config.agent.warmup_steps, config.agent.wm_window)
            collected = warmup + math.ceil(config.trainer.num_learning_iterations / config.agent.utd)
            blocks = math.ceil(collected / config.trainer.block_steps) * config.trainer.block_steps
            return num_envs * blocks
        # Sync collects max(warmup, wm_window, hist) up front, then 1/utd
        # steps per iteration with the fractional part carried over.
        warmup = max(config.agent.warmup_steps, config.agent.wm_window, config.agent.hist)
        return num_envs * (warmup + int(config.trainer.num_learning_iterations / config.agent.utd))


__all__ = ["ApgProvider"]

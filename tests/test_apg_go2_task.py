# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""go2-walk-flat binds APG through its own task-side head, provider unchanged.

This task was added after the APG provider was finalized: its entire
definition is the quadruped head module in ``motrix_envs`` plus the task
yaml — the zero-provider-change contract this file pins from the wiring
side.
"""

from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from motrix_rl_builtin.apg.contracts import resolve_head
from omegaconf import OmegaConf

from motrix_rl import runner, runs
from motrix_rl.plugins import load_plugins


def test_apg_go2_task_composes_and_selects_sync_trainer(tmp_path):
    from motrix_rl_builtin.apg.sync.train import Trainer as SyncTrainer

    load_plugins()
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_module(version_base=None, config_module="motrix_rl.configs"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"hydra.searchpath=[file://{config_dir}]",
                "task=go2-walk-flat/motrix.apg",
            ],
        )
    task_cfg = OmegaConf.to_object(cfg)
    assert task_cfg.task.env == "go2-walk-flat"
    assert task_cfg.task.algo == "apg"
    assert callable(resolve_head(task_cfg.algo.head))

    run = runs.create_run_context(
        env_name=task_cfg.task.env,
        rllib=task_cfg.task.rllib,
        train_backend="torch",
        algo=task_cfg.task.algo,
        seed=task_cfg.seed,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    runs.write_task_config(run.run_dir, task_cfg)
    assert isinstance(runner.create_run_handle(run).trainer, SyncTrainer)

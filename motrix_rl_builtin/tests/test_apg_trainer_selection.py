# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG trainer selection: the ``asynchronous`` switch picks the trainer."""

from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from motrix_rl_builtin.apg.contracts import resolve_head
from omegaconf import OmegaConf

from motrix_rl import runner, runs
from motrix_rl.plugins import load_plugins


def _compose_task_option(*, task: str = "g1-supermimic/motrix.apg", overrides=()):
    load_plugins()
    GlobalHydra.instance().clear()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_module(version_base=None, config_module="motrix_rl.configs"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"hydra.searchpath=[file://{config_dir}]",
                f"task={task}",
                *overrides,
            ],
        )
    return OmegaConf.to_object(cfg)


def test_apg_asynchronous_switches_trainer(tmp_path, monkeypatch):
    from motrix_rl_builtin.apg.sync.train import Trainer as SyncTrainer
    from motrix_rl_builtin.apg.trainer import Trainer as AsyncTrainer

    # The g1-supermimic cfg factory resolves the corpus directory at
    # construction; the trainers only need the spec, so an empty directory
    # satisfies them without building an environment.
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", str(tmp_path))

    sync_cfg = _compose_task_option(overrides=["algo.asynchronous=false"])
    async_cfg = _compose_task_option(overrides=["algo.asynchronous=true"])
    assert sync_cfg.algo.asynchronous is False
    assert async_cfg.algo.asynchronous is True
    # The task binding survives the compose round-trip as a resolvable factory.
    assert callable(resolve_head(sync_cfg.algo.head))

    for index, (task_cfg, trainer_type) in enumerate(((sync_cfg, SyncTrainer), (async_cfg, AsyncTrainer))):
        run = runs.create_run_context(
            env_name=task_cfg.task.env,
            rllib=task_cfg.task.rllib,
            train_backend="torch",
            algo=task_cfg.task.algo,
            seed=task_cfg.seed,
            checkpoint_format="pt",
            runs_root=tmp_path / str(index),
        )
        runs.write_task_config(run.run_dir, task_cfg)

        assert isinstance(runner.create_run_handle(run).trainer, trainer_type)

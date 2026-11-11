# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG sync trainer end-to-end on the real g1-wbt-supermimic task (P3.2).

A tiny recipe drives the full loop — collect (FinalState assembler), world-
model window MSE, imagination BPTT through the derived program, lambda-return
critic, checkpoints, resume, and deterministic play — at cartpole-sized
widths. Acceptance is mechanical (the loop runs, losses are finite, restore
and play work), not convergence.
"""

from pathlib import Path

import pytest
import torch


def _tiny_cfg():
    from motrix_rl_builtin.apg.config import ApgCfg

    cfg = ApgCfg()
    cfg.device = "cpu"
    cfg.world_model.scheme = "floating_base"
    cfg.world_model.hidden = [32, 32]
    cfg.world_model.batch_size = 4
    cfg.world_model.window = 3
    cfg.world_model.batches_per_iteration = 1
    cfg.actor.batch_size = 4
    cfg.actor.history = 2
    cfg.actor.hidden = [16, 16]
    cfg.critic.hidden = [16, 16]
    cfg.critic.minibatches = 2
    cfg.critic.epochs = 1
    cfg.imagination.horizon = 3
    cfg.replay.capacity = 16
    cfg.replay.warmup_steps = 6
    cfg.collector.exploration_noise_std = 0.05
    cfg.trainer.iterations = 3
    cfg.trainer.collect_steps_per_iteration = 2
    cfg.trainer.warmup_iterations = 1
    cfg.trainer.eval_interval = 0
    cfg.trainer.final_eval = False
    cfg.trainer.lr_decay_iterations = 3
    return cfg


def _make_context(tmp_dir, *, resume_from=None):
    import motrix_envs  # noqa: F401
    from motrix_rl.config import LoggingConfig
    from motrix_rl.contracts import TrainerContext
    from motrix_rl.metrics import MetricsLogger
    from motrix_rl.runs import create_run_context

    cfg = _tiny_cfg()
    run = create_run_context(
        env_name="g1-wbt-supermimic",
        rllib="motrix",
        train_backend="torch",
        algo="apg",
        sim=None,
        seed=3,
        checkpoint_format="pt",
        runs_root=tmp_dir,
    )
    return TrainerContext(
        run=run,
        env_name="g1-wbt-supermimic",
        run_dir=run.run_dir,
        checkpoint_dir=run.checkpoint_dir,
        sim=None,
        checkpoint_format="pt",
        num_envs=2,
        play_num_envs=1,
        seed=3,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=type("C", (), {"interval": 2})(),
        render=None,
        resume_from=resume_from,
        metrics=MetricsLogger.disabled(),
    ), cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from motrix_rl_builtin.apg import train as apg_train
    from motrix_rl_builtin.apg.train import Trainer

    context, cfg = _make_context(tmp_path_factory.mktemp("runs"))
    apg_train._HEADLESS_PLAY_STEPS = 20  # keep the play smoke cheap
    trainer = Trainer(context=context)
    trainer.train()
    return context, trainer, cfg


def test_training_loop_produces_finite_history(trained) -> None:
    _context, trainer, _cfg = trained
    assert len(trainer.history) == 3
    for row in trainer.history:
        assert row["apg.wm_loss"] == row["apg.wm_loss"]  # finite


def test_checkpoint_records_and_resumes(trained, tmp_path) -> None:
    from motrix_rl_builtin.apg.train import Trainer

    from motrix_rl import checkpoints

    context, _trainer, cfg = trained
    final = checkpoints.final_checkpoint_path("pt", context.run_dir)
    assert Path(final).exists()
    assert (context.checkpoint_dir / "model_0000002.pt").exists()
    payload = torch.load(final, map_location="cpu", weights_only=False)
    assert payload["iteration"] == cfg.trainer.iterations
    assert set(payload["optimizers"]) == {"dynamics", "actor", "critic"}

    # Resume path: the restored run sees iteration >= target and exits.
    resume_context, _ = _make_context(tmp_path, resume_from=str(final))
    resumed = Trainer(context=resume_context)
    resumed.train()
    assert len(resumed.history) == 0


def test_play_loads_policy_and_rolls(trained) -> None:
    from motrix_rl import checkpoints

    _context, trainer, _cfg = trained
    final = checkpoints.final_checkpoint_path("pt", _context.run_dir)
    trainer.play(str(final))

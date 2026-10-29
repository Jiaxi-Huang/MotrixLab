# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end APG world-model integration: real env, task head, learner loop.

Covers the cross-package wiring — the task-declared parsing head resolved
from the task yaml's dotted path, the learner over the real environment,
both trainer topologies with checkpointing, and the per-clip evaluator
wired through the provider's policy rebuilding. Task-side head parity with
the manager kernels lives in ``motrix_envs/tests``.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.evaluate import load_policy
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.sync.train import Trainer as SyncTrainer
from motrix_rl_builtin.apg.trainer import Trainer

from motrix_envs.locomotion.supermimic.apg_head import build_head
from motrix_envs.locomotion.supermimic.evaluate import evaluate
from motrix_envs.locomotion.supermimic.g1 import SuperMimicManagerEnv, make_g1_supermimic_cfg
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerContext
from motrix_rl.runs import RunMetadata

_DANCE_MOTION = MOTION_DIR / "dance" / "dance1_subject2.npz"

pytestmark = [pytest.mark.numba, pytest.mark.integration]


def _seed_run_metadata(run_dir: Path, env_name: str = "g1-supermimic") -> None:
    """Mirror the runner: trainers annotate metadata.json, so create it first.

    Real runs go through ``runs.create_run_context`` before the trainer sees
    the run directory; these tests build ``TrainerContext`` by hand and need
    the same seed record for the trainers' ``annotate_run`` calls.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = RunMetadata(
        env_name=env_name,
        rllib="motrix",
        train_backend="torch",
        algo="apg",
        seed=1,
        created_at="1970-01-01T00:00:00",
        checkpoint_format="pt",
    )
    (run_dir / "metadata.json").write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")


def _corpus(tmp_path, lengths=(160, 140)) -> tuple[str, ...]:
    with np.load(_DANCE_MOTION, allow_pickle=False) as data:
        total = int(np.asarray(data["num_frames"]).reshape(-1)[0])
        fields = {key: data[key] for key in data.files}
    paths = []
    start = 0
    for index, length in enumerate(lengths):
        part = {"num_frames": np.int32(length)}
        for name, value in fields.items():
            if name == "num_frames":
                continue
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                part[name] = value[start : start + length]
            else:
                part[name] = value
        path = tmp_path / f"corpus{index}.npz"
        np.savez(path, **part)
        paths.append(str(path))
        start += length
    return tuple(paths)


def _learner_cfg(num_envs: int) -> ApgCfg:
    cfg = ApgCfg()
    cfg.head = "motrix_envs.locomotion.supermimic.apg_head:build_head"
    cfg.device = "cpu"
    agent = cfg.agent
    agent.buffer_steps = 24
    agent.utd = 0.5
    agent.horizon = 4
    agent.hist = 2
    agent.z_dim = 8
    agent.wm_hidden = [32, 32]
    agent.policy_hidden = [32, 32]
    agent.critic_hidden = [16, 16]
    agent.wm_window = 2
    agent.wm_batches = 1
    agent.wm_batch = 16
    agent.pi_batch = 8
    agent.lr = 1e-3
    agent.gamma = 0.99
    agent.lam = 0.95
    agent.action_noise = 0.1
    agent.critic_epochs = 1
    agent.critic_minibatches = 2
    agent.warmup_steps = 1
    agent.wm_only_iters = 0
    agent.lr_decay_iters = 100
    agent.amp = "none"
    agent.compile = False
    return cfg


def test_apg_learner_runs_on_real_env(tmp_path):
    cfg = _learner_cfg(4)
    env_cfg = make_g1_supermimic_cfg(_corpus(tmp_path))
    env_cfg.commands.motion.num_goal_frames = 2
    env = SuperMimicManagerEnv(env_cfg, num_envs=4, seed=1)
    learner = ApgLearner(
        env,
        cfg.agent,
        build_head(env),
        device=torch.device("cpu"),
        seed=0,
    )
    learner.observe(env.init_state().obs.policy)
    learner.collect(6)
    assert learner.buffer.size == 6
    assert np.all(np.isfinite(learner.head.motion.state))
    wm_loss, wm_error = learner.wm_update()
    assert torch.isfinite(wm_loss)
    assert torch.isfinite(wm_error)
    pi_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()
    assert torch.isfinite(pi_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(imagined_reward)
    assert torch.isfinite(survival)

    checkpoint = learner.state_dict()
    assert set(checkpoint) >= {"wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v"}


def test_apg_trainer_runs_two_iterations(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, lengths=(200, 180))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 2
    # Publish a policy snapshot every iteration: the snapshot path is part of
    # the async loop's contract, not an after-iteration-10 corner.
    cfg.trainer.snapshot_every = 1
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "run" / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )

    _seed_run_metadata(context.run_dir, "g1-supermimic")
    Trainer(context=context).train()

    final = tmp_path / "run" / "checkpoints" / "latest.pt"
    assert final.is_file()
    assert (tmp_path / "run" / "checkpoints" / "manifest.json").is_file()
    checkpoint = torch.load(final, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert set(checkpoint) >= {"wm", "pi", "critic"}
    assert {"opt_wm", "opt_pi", "opt_v", "sampling"} <= set(checkpoint)
    # Two warmup vector steps earn no update credit; each subsequent
    # two-step block permits one update at UTD=0.5.
    assert checkpoint["sim_steps"] == 6 * context.num_envs

    # A new corpus stage imports learned networks while starting its own clock
    # and optimizer state, so its one-iteration target must execute once.
    cfg.trainer.initialize_from = str(final)
    cfg.trainer.num_learning_iterations = 1
    next_run = tmp_path / "initialized"
    _seed_run_metadata(next_run, "g1-supermimic")
    Trainer(context=replace(context, run_dir=next_run, checkpoint_dir=next_run / "checkpoints")).train()
    initialized = torch.load(next_run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert initialized["iteration"] == 1
    assert initialized["sim_steps"] < checkpoint["sim_steps"]


def test_apg_sync_trainer_runs_two_iterations(tmp_path, monkeypatch):
    """The sync trainer collects in-process and serializes WM -> policy updates."""
    corpus = _corpus(tmp_path, lengths=(200, 180))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 2
    run_dir = tmp_path / "sync_run"
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )

    _seed_run_metadata(run_dir, "g1-supermimic")
    SyncTrainer(context=context).train()

    final = run_dir / "checkpoints" / "latest.pt"
    assert final.is_file()
    assert (run_dir / "checkpoints" / "manifest.json").is_file()
    checkpoint = torch.load(final, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert {"opt_wm", "opt_pi", "opt_v", "sampling"} <= set(checkpoint)
    # Warmup collects max(warmup_steps, wm_window, hist) = 2 steps; each of
    # the two iterations then collects 1 / utd = 2 steps before its serial
    # update, all on the in-process simulator.
    assert checkpoint["sim_steps"] == 6 * context.num_envs

    # Trainer provenance lands in metadata.json: the run's device record
    # (gpus_used is empty on a CPU-only host) and the end-of-run summary.
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "system" in metadata and isinstance(metadata["system"]["gpus_used"], list)
    assert metadata["performance"]["iterations"] == 2
    assert metadata["performance"]["total_env_steps"] == 6 * context.num_envs
    assert metadata["performance"]["mean_env_steps_per_s"] > 0


def test_apg_quadruped_sync_trainer_runs_two_iterations(tmp_path):
    """A command-driven quadruped task trains through the same provider.

    The go2 head, its task yaml, and this run exercise the whole APG stack
    over a second task family: everything task-specific lives in
    ``motrix_envs`` and the task yaml.
    """
    cfg = _learner_cfg(2)
    cfg.head = "motrix_envs.locomotion.quadruped.apg_head:build_head"
    cfg.trainer.num_learning_iterations = 2
    run_dir = tmp_path / "go2_sync_run"
    context = TrainerContext(
        run=None,
        env_name="go2-walk-flat",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=2,
        play_num_envs=1,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )

    _seed_run_metadata(run_dir, "go2-walk-flat")
    SyncTrainer(context=context).train()

    final = run_dir / "checkpoints" / "latest.pt"
    assert final.is_file()
    checkpoint = torch.load(final, map_location="cpu", weights_only=False)
    assert checkpoint["iteration"] == 2
    assert {"opt_wm", "opt_pi", "opt_v"} <= set(checkpoint)
    # Warmup 2 steps plus two iterations at 1 / utd = 2 steps each.
    assert checkpoint["sim_steps"] == 6 * context.num_envs


def test_apg_per_clip_evaluation(tmp_path, monkeypatch):
    """Train briefly, then run deterministic per-clip evaluation twice.

    The trainer checkpoint carries the policy hyperparameters; the evaluator
    pins one lane per clip to its first frame, and repeated evaluation of the
    same checkpoint is bit-stable.
    """
    corpus = _corpus(tmp_path, lengths=(120, 100))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 1
    run_dir = tmp_path / "run"
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )
    _seed_run_metadata(context.run_dir, "g1-supermimic")
    Trainer(context=context).train()
    checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)

    env = SuperMimicManagerEnv(make_g1_supermimic_cfg(corpus).for_eval(), num_envs=2, seed=3)
    head = build_head(env)
    policy = load_policy(head, checkpoint, torch.device("cpu"))
    names = ["corpus0", "corpus1"]
    first = evaluate(
        env,
        policy,
        head,
        hist=cfg.agent.hist,
        device=torch.device("cpu"),
        clip_names=names,
    )
    second = evaluate(
        env,
        policy,
        head,
        hist=cfg.agent.hist,
        device=torch.device("cpu"),
        clip_names=names,
    )

    assert len(first["per_clip"]) == 2
    assert {entry["clip"] for entry in first["per_clip"].values()} == {"corpus0", "corpus1"}
    for entry in first["per_clip"].values():
        assert entry["length"] in (120, 100)
        assert 0 <= entry["steps"] <= entry["length"]
    assert first["norm_return"] <= 1.0 + 1e-6
    assert 0 <= first["n_failed"] <= 2
    # Physics-level warm-start differences across repeated rollouts keep the
    # comparison at solver tolerance rather than bit equality.
    for key in ("norm_return", "norm_return_framew"):
        assert first[key] == pytest.approx(second[key], abs=2e-3)
    for index in first["per_clip"]:
        assert first["per_clip"][index]["steps"] == second["per_clip"][index]["steps"]
        assert first["per_clip"][index]["failed"] == second["per_clip"][index]["failed"]


def test_apg_eval_script_runs(tmp_path, monkeypatch, capsys):
    """The repo eval script resolves a run's checkpoint and writes json/csv."""
    import importlib.util

    corpus = _corpus(tmp_path, lengths=(80, 70))
    monkeypatch.setenv("SUPERMIMIC_MOTION_DIR", corpus[0] + ":" + corpus[1])
    cfg = _learner_cfg(4)
    cfg.trainer.num_learning_iterations = 1
    run_dir = tmp_path / "run"
    context = TrainerContext(
        run=None,
        env_name="g1-supermimic",
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=2,
        seed=1,
        rl_cfg=cfg,
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=1),
    )
    _seed_run_metadata(context.run_dir, "g1-supermimic")
    Trainer(context=context).train()

    spec = importlib.util.spec_from_file_location(
        "eval_supermimic", Path(__file__).resolve().parents[1] / "scripts" / "eval_supermimic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--run_dir", str(run_dir), "--device", "cpu"])

    out = capsys.readouterr().out
    assert "norm_return" in out
    assert (run_dir / "checkpoints" / "latest_eval.json").is_file()
    assert (run_dir / "checkpoints" / "eval.csv").is_file()
    with (run_dir / "checkpoints" / "latest_eval.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    assert set(result["failed"]) <= {"corpus0", "corpus1"}
    assert len(result["per_clip"]) == 2

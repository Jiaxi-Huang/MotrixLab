# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Asynchronous APG trainer over the MotrixLab control plane."""

from __future__ import annotations

import json
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.multiprocessing as mp

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import create_renderer
from motrix_rl import checkpoints
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.training_logger import TrainingLogger
from motrix_rl_builtin.apg.async_train import (
    UpdateCredit,
    collect_blocks,
    ingest_block,
    pack_owned,
    policy_snapshot,
    put_latest,
    receive_block,
)
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.contracts import build_head, require_apg_env
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.rollout import shift_history
from motrix_rl_builtin.apg.wm_client import WMClient

# Deliberately NOT torch.set_float32_matmul_precision("high"): on ROCm the
# TF32 matmul route makes the small 3x3 bmms of the analytic parsing head pay
# tens of milliseconds of host-side time per call, while the fp32 default is
# fast there (bf16 autocast still covers the large MLP matmuls).


class Trainer(TrainerBase):
    """Reference APG async trainer with an optional dual-GPU WM worker.

    The dual-GPU topology (``trainer.wm_device``) is default-off and has no
    automated coverage; verify it manually on a two-GPU host before relying
    on it.
    """

    def __init__(self, *, context: TrainerContext[ApgCfg]) -> None:
        self._context = context
        self._env_name = context.env_name
        self._rlcfg = context.rl_cfg
        self._env_spec = env_registry.resolve(context.env_name, sim=context.sim)
        self._render = context.render

    # ------------------------------------------------------------------ setup

    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return resolve_default_device()

    def _make_env(self, num_envs: int, mode: str):
        env = self._env_spec.make(num_envs=num_envs, mode=mode, seed=self._context.seed)
        return require_apg_env(env)

    def _make_head(self, env):
        return build_head(self._rlcfg.head, env)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ train

    def train(self) -> None:
        self._set_seed(self._context.seed)
        if self._context.logging.backend != "tensorboard":
            raise ValueError("motrix.apg supports only the 'tensorboard' logging backend.")
        run_dir = self._context.run_dir
        training_log = TrainingLogger(run_dir, self._context.metrics)
        try:
            # The learner needs task metadata and curriculum state, not a
            # second full simulator. Physics lives in the collector.
            env = self._make_env(1, mode="train")
            learner = ApgLearner(
                env,
                self._rlcfg.agent,
                self._make_head(env),
                device=self._device(),
                num_envs=self._context.num_envs,
                seed=self._context.seed,
            )
            num_iterations = self._rlcfg.trainer.num_learning_iterations
            source = self._rlcfg.trainer.initialize_from
            if source and self._context.resume_from:
                raise ValueError("initialize_from and resume_from cannot be combined")
            if source:
                checkpoint = torch.load(source, map_location=learner.device, weights_only=False)
                learner.initialize_from(checkpoint)
                training_log.event("initialized_from", step=0, path=source)
            if self._context.resume_from:
                checkpoint = torch.load(self._context.resume_from, map_location=learner.device, weights_only=False)
                learner.load_state_dict(checkpoint)
                training_log.event("resumed", step=learner.iteration, path=str(self._context.resume_from))

            print(
                f"[motrix.apg] training '{self._env_name}' on {learner.device} "
                f"num_envs={learner.num_envs} iters={num_iterations} (from {learner.iteration})"
            )
            training_log.event("run_started", step=learner.iteration)

            def record(path) -> None:
                checkpoints.record_checkpoint_artifact(
                    run_dir,
                    checkpoints.LATEST_TRAINING_STATE,
                    path,
                    checkpoints.TRAINING_STATE,
                    checkpoint_format=self._context.checkpoint_format,
                )

            self._run_loop(env, learner, training_log, num_iterations=num_iterations, record_checkpoint=record)

            ckpt_path = checkpoints.final_checkpoint_path(self._context.checkpoint_format, run_dir)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(learner.state_dict(include_optimizers=True), ckpt_path)
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.LATEST_TRAINING_STATE,
                ckpt_path,
                checkpoints.TRAINING_STATE,
                checkpoint_format=self._context.checkpoint_format,
            )
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.BEST_POLICY,
                ckpt_path,
                checkpoints.POLICY,
                checkpoint_format=self._context.checkpoint_format,
            )
            print(f"[motrix.apg] saved checkpoint to {ckpt_path}")
            training_log.event("checkpoint_saved", step=learner.iteration, path=str(ckpt_path))
        finally:
            training_log.close()

    def play(self, policy: str) -> None:
        attach_play(self, policy)

    def _run_loop(self, env, learner, training_log, *, num_iterations: int, record_checkpoint) -> None:
        cfg = self._rlcfg.agent
        tc = self._rlcfg.trainer
        self._validate_async_config(learner.device)
        provenance = {
            "collector_device": "cpu",
            "collector_policy_device": tc.collector_policy_device,
            "learner_device": str(learner.device),
            "wm_device": tc.wm_device or str(learner.device),
            "wm_model_lag": 1 if tc.wm_device else 0,
            "wm_transport": "persistent GPU snapshot" if tc.wm_device else "local",
            "resume": "fresh replay and simulator warmup",
        }
        self._context.run_dir.mkdir(parents=True, exist_ok=True)
        (self._context.run_dir / "apg_async.json").write_text(json.dumps(provenance, indent=2))
        logging_interval = max(self._context.logging.interval, 1)
        checkpoint_interval = max(self._context.checkpoint.interval, 1)
        required_steps = max(cfg.warmup_steps, cfg.wm_window)
        context = mp.get_context("spawn")
        transitions = context.Queue(maxsize=tc.queue_blocks)
        snapshots = context.Queue(maxsize=1)
        stop = context.Event()
        errors, error_sender = context.Pipe(duplex=False)
        collector = context.Process(
            target=collect_blocks,
            name="apg-cpu-collector",
            args=(
                self._env_spec,
                learner.num_envs,
                self._context.seed,
                cfg,
                self._rlcfg.head,
                tc.block_steps,
                tc.collector_threads,
                tc.collector_policy_device,
                policy_snapshot(learner),
                pack_owned(learner.head.sampling_state()),
                transitions,
                snapshots,
                stop,
                error_sender,
            ),
        )
        wm_client = None
        if tc.wm_device:
            wm_client = WMClient(
                context,
                SimpleNamespace(
                    cfg=self._rlcfg,
                    env_spec=self._env_spec,
                    seed=self._context.seed,
                    queue_blocks=tc.queue_blocks,
                    worker_timeout=tc.worker_timeout,
                    wm_device=tc.wm_device,
                ),
                learner,
                stop,
            )
        credit = UpdateCredit(cfg.utd, required_steps)
        sequence = replacements = 0
        started = time.monotonic()
        log_started = started
        logged_sim_steps = learner.sim_steps
        try:
            collector.start()
            error_sender.close()
            while learner.iteration < num_iterations:
                if wm_client is not None:
                    wm_client.check()
                receive_started = time.monotonic()
                block = receive_block(transitions, collector, errors, tc.worker_timeout)
                learner_wait_s = time.monotonic() - receive_started
                n = ingest_block(
                    learner,
                    block,
                    sequence,
                    update_wm_norms=wm_client is None,
                )
                if wm_client is not None:
                    wm_client.send({"kind": "ingest", "block": block})
                sequence += 1
                credit.receive(n)
                while credit.available and learner.iteration < num_iterations:
                    iteration = learner.iteration
                    lr = learner.apply_lr(iteration)
                    wm_loss = wm_error = torch.zeros((), device=learner.device)
                    wm_started = time.monotonic()
                    wm_train_s = 0.0
                    wm_job = None
                    if wm_client is None:
                        # The shared WM-fit stage; the deferred flag keeps every
                        # batch's loss tensor alive for the late metric readback.
                        wm_loss, wm_error = learner.fit_wm(defer_loss_sync=cfg.defer_wm_loss_sync)
                    else:
                        wm_job = iteration + 1
                        wm_client.send({"kind": "train", "version": wm_job, "watermark": sequence, "lr": lr})
                    actor_started = time.monotonic()
                    if iteration >= cfg.wm_only_iters:
                        policy = learner.policy_update()
                    else:
                        policy = (torch.zeros(()),) * 5
                    actor_s = time.monotonic() - actor_started
                    wm_wait_s = wm_install_s = 0.0
                    if wm_client is not None:
                        waited = time.monotonic()
                        result = wm_client.receive("done", wm_job, sequence)
                        wm_wait_s = time.monotonic() - waited
                        wm_loss = torch.tensor(result["loss"], device=learner.device)
                        wm_error = torch.tensor(result["error"], device=learner.device)
                        wm_train_s = result["train_s"]
                        installed = time.monotonic()
                        wm_client.install()
                        wm_install_s = time.monotonic() - installed
                    learner.iteration += 1
                    learner.wall_time = time.monotonic() - started
                    credit.consume()
                    if learner.iteration % tc.snapshot_every == 0:
                        replacements += put_latest(snapshots, policy_snapshot(learner))
                    if learner.iteration % logging_interval == 0:
                        now = time.monotonic()
                        stats = learner.stats
                        done = sum(stats["done"])
                        metrics = {
                            "train/learning_rate": lr,
                            "train/wm_loss": float(wm_loss.detach()),
                            "train/wm_last_error": float(wm_error.detach()),
                            "train/actor_s": actor_s,
                            "train/wm_train_s": wm_train_s or time.monotonic() - wm_started,
                            "train/wm_wait_s": wm_wait_s,
                            "train/wm_install_s": wm_install_s,
                            "train/policy_lag": float(learner.iteration - block["policy_version"]),
                            "train/normalizer_snapshot_steps": float(block.get("norm_steps", 0)),
                            "train/wm_model_lag": 1.0 if wm_client is not None else 0.0,
                            "train/update_credit": float(credit.available),
                            "train/sim_steps": float(learner.sim_steps),
                            "train/sim_reward_norm": _mean(stats["r"]),
                            "train/completed_episodes": done,
                            "train/iterations_per_s": logging_interval / max(now - log_started, 1e-9),
                            "train/wall_time": learner.wall_time,
                            "train/snapshots_replaced": float(replacements),
                            "train/collector_sample_s": float(block.get("collector_sample_s", 0.0)),
                            "train/learner_wait_s": learner_wait_s,
                            "train/collector_wait_s": float(block.get("collector_wait_s", 0.0)),
                            "train/collector_fps": float(
                                block.get("vector_steps", 0)
                                * learner.num_envs
                                / max(block.get("collector_elapsed_s", 0.0), 1e-9)
                            ),
                            "train/ingest_fps": float(credit.steps * learner.num_envs / max(now - started, 1e-9)),
                            # Canonical live-metrics key: env steps since the
                            # last log point over wall time.
                            "perf/env_steps_per_s": (learner.sim_steps - logged_sim_steps)
                            / max(now - log_started, 1e-9),
                        }
                        if learner.iteration > cfg.wm_only_iters:
                            metrics.update(
                                {
                                    "train/policy_loss": float(policy[0]),
                                    "train/value_loss": float(policy[1]),
                                    "train/imagined_reward": float(policy[2]),
                                    "train/policy_grad_norm": float(policy[3]),
                                    "train/imagined_survival": float(policy[4]),
                                }
                            )
                        if done:
                            # Canonical live-metrics rollout means (plus the
                            # legacy train/* alias kept for existing panels).
                            rollout = learner.episode_metrics()
                            metrics.update(rollout)
                            metrics["train/episode_length"] = rollout["rollout/mean_ep_len"]
                            metrics["train/termination_fraction"] = sum(stats["term"]) / done
                        sampling_metrics = learner.head.sampling_metrics()
                        metrics.update({f"train/{name}": value for name, value in sampling_metrics.items()})
                        training_log.scalars(metrics, learner.iteration, env_steps=learner.sim_steps)
                        learner.stats = {key: [] for key in stats}
                        logged_sim_steps = learner.sim_steps
                        log_started = now
                    if learner.iteration % checkpoint_interval == 0 or learner.iteration == num_iterations:
                        if wm_client is not None:
                            wm_client.send({"kind": "snapshot", "version": learner.iteration, "watermark": sequence})
                            snapshot = wm_client.receive("snapshot", learner.iteration, sequence)
                            learner.wm.load_state_dict(snapshot["wm"])
                            learner.opt_wm.load_state_dict(snapshot["opt_wm"])
                        path = self._context.run_dir / "checkpoints" / f"ckpt_{learner.iteration:06d}.pt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(learner.state_dict(include_optimizers=True), path)
                        record_checkpoint(path)
        finally:
            stop.set()
            if wm_client is not None:
                wm_client.close()
            if collector.pid is not None:
                collector.join(timeout=10)
                if collector.is_alive():
                    collector.terminate()
                    collector.join(timeout=5)
            error_sender.close()
            for queue in (transitions, snapshots):
                queue.cancel_join_thread()
                queue.close()
            errors.close()

    def _validate_async_config(self, learner_device: torch.device) -> None:
        cfg = self._rlcfg.agent
        tc = self._rlcfg.trainer
        if tc.block_steps <= 0 or tc.queue_blocks <= 0 or tc.collector_threads <= 0 or tc.snapshot_every <= 0:
            raise ValueError("APG async block, queue and snapshot intervals must be positive")
        if tc.worker_timeout <= 0 or cfg.utd <= 0 or cfg.buffer_steps < cfg.wm_window:
            raise ValueError("Invalid APG async or replay configuration")
        collector = torch.device(tc.collector_policy_device)
        if collector.type not in ("cpu", "cuda"):
            raise ValueError("collector_policy_device must be cpu or cuda[:index]")
        if collector.type == "cuda" and (
            not torch.cuda.is_available()
            or (collector.index is not None and collector.index >= torch.cuda.device_count())
        ):
            raise ValueError(f"collector_policy_device is unavailable: {collector}")
        if tc.wm_device:
            wm = torch.device(tc.wm_device)
            if (
                learner_device.type != "cuda"
                or wm.type != "cuda"
                or wm == learner_device
                or wm.index is not None
                and wm.index >= torch.cuda.device_count()
            ):
                raise ValueError("wm_device must be a different CUDA device from the actor learner")


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


class PlayRunner:
    """Playback of a trained APG policy over the env's own observations."""

    def __init__(self, env, learner: ApgLearner):
        self._env = env
        self._learner = learner

    @torch.no_grad()
    def run(self, renderer) -> None:
        env = self._env
        learner = self._learner
        learner.pi.eval()
        obs, goal = learner.observe(env.init_state().obs.policy)
        obs_hist = None
        while renderer.render():
            if obs_hist is None:
                obs_hist = obs.unsqueeze(1).expand(-1, learner.cfg.hist, -1).clone()
            else:
                obs_hist = shift_history(obs_hist, obs)
            action = learner.pi(obs_hist, goal)
            result = env.step(action.detach().cpu().numpy().astype(np.float32))
            obs, goal = learner.observe(result.obs.policy)


def attach_play(trainer: Trainer, policy: str) -> None:
    """Load a checkpoint into a play-mode trainer and run the rollout loop."""
    trainer._set_seed(trainer._context.seed)
    env = trainer._make_env(trainer._context.play_num_envs, mode="play")
    renderer = create_renderer(env, trainer._render)
    if renderer is None:
        raise ValueError("motrix.apg play requires a render configuration.")
    learner = ApgLearner(
        env,
        trainer._rlcfg.agent,
        trainer._make_head(env),
        device=trainer._device(),
        seed=trainer._context.seed,
    )
    checkpoint = torch.load(policy, map_location=learner.device, weights_only=False)
    learner.pi.load_state_dict(checkpoint["pi"])
    PlayRunner(env, learner).run(renderer)

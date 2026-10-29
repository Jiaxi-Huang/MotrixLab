# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Synchronous APG trainer: in-process collection and serial updates.

Mirrors the upstream single-process loop: the environment rolls out in this
process and every iteration collects ``1 / utd`` env steps, fits the world
model for ``wm_batches`` batches, then runs the imagination policy update —
strictly in order, inside :meth:`ApgLearner.update`.
"""

from __future__ import annotations

import random
import time

import numpy as np
import torch

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import create_renderer
from motrix_rl import checkpoints
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.training_logger import TrainingLogger
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.contracts import build_head, require_apg_env
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.rollout import shift_history


class Trainer(TrainerBase):
    """Single-process APG trainer with serial WM -> policy updates."""

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

    def _make_learner(self, env) -> ApgLearner:
        head = build_head(self._rlcfg.head, env)
        return ApgLearner(
            env,
            self._rlcfg.agent,
            head,
            device=self._device(),
            seed=self._context.seed,
        )

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
            env = self._make_env(self._context.num_envs, mode="train")
            learner = self._make_learner(env)
            learner.observe(env.init_state().obs.policy)
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
                f"[motrix.apg sync] training '{self._env_name}' on {learner.device} "
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
            print(f"[motrix.apg sync] saved checkpoint to {ckpt_path}")
            training_log.event("checkpoint_saved", step=learner.iteration, path=str(ckpt_path))
        finally:
            training_log.close()

    def play(self, policy: str) -> None:
        self._set_seed(self._context.seed)
        env = self._make_env(self._context.play_num_envs, mode="play")
        renderer = create_renderer(env, self._render)
        if renderer is None:
            raise ValueError("motrix.apg play requires a render configuration.")
        learner = self._make_learner(env)
        checkpoint = torch.load(policy, map_location=learner.device, weights_only=False)
        learner.pi.load_state_dict(checkpoint["pi"])
        learner.pi.eval()

        obs, goal = learner.observe(env.init_state().obs.policy)
        obs_hist = None
        while renderer.render():
            if obs_hist is None:
                obs_hist = obs.unsqueeze(1).expand(-1, learner.cfg.hist, -1).clone()
            else:
                obs_hist = shift_history(obs_hist, obs)
            action = learner.pi(obs_hist, goal)
            state = env.step(action.detach().cpu().numpy().astype(np.float32))
            obs, goal = learner.observe(state.obs.policy)

    # ------------------------------------------------------------------ sync train loop

    def _run_loop(self, env, learner, training_log, *, num_iterations: int, record_checkpoint) -> None:
        cfg = self._rlcfg.agent
        logging_interval = max(self._context.logging.interval, 1)
        checkpoint_interval = max(self._context.checkpoint.interval, 1)
        # Warmup collection covers the policy-update warmup, one full WM
        # window, and one observation history.
        learner.collect(max(cfg.warmup_steps, cfg.wm_window, cfg.hist))
        accumulator = 0.0
        started = time.monotonic()
        log_started = started
        logged_sim_steps = learner.sim_steps
        while learner.iteration < num_iterations:
            # One serial update per iteration; utd fixes the env steps
            # collected per update (1 / utd, accumulated so the collection
            # cadence averages out).
            accumulator += 1.0 / cfg.utd
            collect_steps = int(accumulator)
            if collect_steps:
                learner.collect(collect_steps)
                accumulator -= collect_steps
            # The three stages, explicit: schedule, world-model fit, then the
            # imagined policy/critic update once the WM warmup completes.
            wm_loss, wm_error = learner.fit_wm()
            update_metrics = {
                "learning_rate": learner.apply_lr(),
                "wm_loss": float(wm_loss.detach()),
                "wm_last_error": float(wm_error.detach()),
            }
            if learner.iteration >= cfg.wm_only_iters:
                update_metrics.update(learner.imagine_and_update_policy())
            learner.iteration += 1
            learner.wall_time = time.monotonic() - started
            iteration = learner.iteration

            if iteration % logging_interval == 0:
                now = time.monotonic()
                stats = learner.stats
                done = sum(stats["done"])
                metrics = {f"train/{name}": value for name, value in update_metrics.items()}
                metrics.update(
                    {
                        "train/sim_steps": float(learner.sim_steps),
                        "train/sim_reward_norm": _mean(stats["r"]),
                        "train/completed_episodes": done,
                        "train/iterations_per_s": logging_interval / max(now - log_started, 1e-9),
                        "train/wall_time": learner.wall_time,
                        # Canonical live-metrics key: env steps since the last
                        # log point over wall time.
                        "perf/env_steps_per_s": (learner.sim_steps - logged_sim_steps) / max(now - log_started, 1e-9),
                    }
                )
                if done:
                    # Canonical live-metrics rollout means (plus the legacy
                    # train/* alias kept for existing panels).
                    rollout = learner.episode_metrics()
                    metrics.update(rollout)
                    metrics["train/episode_length"] = rollout["rollout/mean_ep_len"]
                    metrics["train/termination_fraction"] = sum(stats["term"]) / done
                sampling_metrics = learner.head.sampling_metrics()
                metrics.update({f"train/{name}": value for name, value in sampling_metrics.items()})
                training_log.scalars(metrics, iteration, env_steps=learner.sim_steps)
                learner.stats = {key: [] for key in stats}
                logged_sim_steps = learner.sim_steps
                log_started = now
            if iteration % checkpoint_interval == 0 or iteration == num_iterations:
                path = self._context.run_dir / "checkpoints" / f"ckpt_{iteration:06d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(learner.state_dict(include_optimizers=True), path)
                record_checkpoint(path)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))

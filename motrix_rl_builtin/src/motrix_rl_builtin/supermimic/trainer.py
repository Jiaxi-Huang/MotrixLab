# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Synchronous SuperMimic trainer over the MotrixLab control plane."""

from __future__ import annotations

import faulthandler
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
from motrix_rl_builtin.supermimic.config import SuperMimicCfg
from motrix_rl_builtin.supermimic.contracts import require_supermimic_env
from motrix_rl_builtin.supermimic.learner import SuperMimicLearner
from motrix_rl_builtin.supermimic.mdp import build_supermimic_mdp

# Deliberately NOT torch.set_float32_matmul_precision("high"): on ROCm the
# TF32 matmul route makes the small 3x3 bmms of the analytic MDP pay tens of
# milliseconds of host-side time per call, while the fp32 default is fast
# there (bf16 autocast still covers the large MLP matmuls).


class Trainer(TrainerBase):
    """Alternate collection and world-model training in one process."""

    def __init__(self, *, context: TrainerContext[SuperMimicCfg]) -> None:
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
        return require_supermimic_env(env)

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
            raise ValueError("SuperMimic supports only the 'tensorboard' logging backend.")
        run_dir = self._context.run_dir
        training_log = TrainingLogger(run_dir, self._context.metrics)
        try:
            env = self._make_env(self._context.num_envs, mode="train")
            learner = SuperMimicLearner(
                env,
                self._rlcfg.agent,
                build_supermimic_mdp(env),
                device=self._device(),
                action_low=np.full(env.command_terms["motion"].clip.joint_pos.shape[1], -1.0, np.float32),
                action_high=np.full(env.command_terms["motion"].clip.joint_pos.shape[1], 1.0, np.float32),
                goal_frames=self._rlcfg.agent.goal_frames,
                seed=self._context.seed,
            )
            num_iterations = self._rlcfg.trainer.num_learning_iterations
            if self._context.resume_from:
                checkpoint = torch.load(self._context.resume_from, map_location=learner.device, weights_only=False)
                learner.load_state_dict(checkpoint)
                training_log.event("resumed", step=learner.iteration, path=str(self._context.resume_from))

            print(
                f"[motrix.supermimic] training '{self._env_name}' on {learner.device} "
                f"num_envs={env.num_envs} iters={num_iterations} (from {learner.iteration})"
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
            torch.save(learner.state_dict(include_optimizers=False), ckpt_path)
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
            print(f"[motrix.supermimic] saved checkpoint to {ckpt_path}")
            training_log.event("checkpoint_saved", step=learner.iteration, path=str(ckpt_path))
        finally:
            training_log.close()

    def play(self, policy: str) -> None:
        attach_play(self, policy)

    def _run_loop(self, env, learner, training_log, *, num_iterations: int, record_checkpoint) -> None:
        cfg = self._rlcfg.agent
        checkpoint_interval = max(self._context.checkpoint.interval, 1)
        logging_interval = max(self._context.logging.interval, 1)
        start_time = time.time()
        log_start = start_time
        train_start = start_time - learner.wall_time

        required_steps = max(cfg.hist, cfg.wm_window, cfg.warmup_steps)
        if learner.buffer.size < required_steps:
            learner.collect(required_steps - learner.buffer.size)

        # Long unattended runs: if the loop ever wedges (e.g. a hung device
        # kernel), dump the Python stack to stderr every 10 minutes instead
        # of spinning silently.
        faulthandler.dump_traceback_later(600, repeat=True)

        step_accumulator = 0.0
        for iteration in range(learner.iteration, num_iterations):
            if iteration < 300 or iteration % 25 == 0:
                print(
                    f"[motrix.supermimic] iter {iteration}/{num_iterations} "
                    f"buffer={learner.buffer.size} elapsed={time.time() - start_time:.0f}s "
                    f"mm_prec={torch.get_float32_matmul_precision()} "
                    f"tf32={torch.backends.cuda.matmul.allow_tf32} "
                    f"cudnn_tf32={torch.backends.cudnn.allow_tf32}",
                    flush=True,
                )
            learning_rate = learner.lr_at(iteration)
            for optimizer in (learner.opt_wm, learner.opt_pi, learner.opt_v):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

            step_accumulator += 1.0 / cfg.utd
            collect_steps = int(step_accumulator)
            step_accumulator -= collect_steps
            learner.collect(collect_steps)

            wm_loss = wm_error = torch.zeros((), device=learner.device)
            for _ in range(cfg.wm_batches):
                wm_loss, wm_error = learner.wm_update()

            metrics: dict[str, float] = {
                "train/wm_loss": float(wm_loss),
                "train/wm_last_error": float(wm_error),
                "train/learning_rate": learning_rate,
            }
            if iteration >= cfg.wm_only_iters:
                pi_loss, value_loss, imagined_reward, grad_norm, survival = learner.policy_update()
                metrics.update(
                    {
                        "train/policy_loss": float(pi_loss),
                        "train/value_loss": float(value_loss),
                        "train/imagined_reward": float(imagined_reward),
                        "train/policy_grad_norm": float(grad_norm),
                        "train/imagined_survival": float(survival),
                    }
                )
            learner.iteration = iteration + 1
            learner.wall_time = time.time() - train_start
            stats = learner.stats
            if learner.iteration % logging_interval == 0:
                done_total = max(sum(stats["done"]), 1.0)
                metrics.update(
                    {
                        "train/sim_reward_norm": _mean(stats["r"]),
                        "train/episode_length": _mean(stats["ep_len"]),
                        "train/termination_fraction": sum(stats["term"]) / done_total,
                        "train/iterations_per_s": logging_interval / (time.time() - log_start),
                        "train/sim_steps": float(learner.sim_steps),
                        "train/wall_time": learner.wall_time,
                    }
                )
                training_log.scalars(metrics, learner.iteration, env_steps=learner.sim_steps)
                log_start = time.time()
                learner.stats = {"r": [], "ep_len": [], "term": [], "done": []}

            if learner.iteration % checkpoint_interval == 0 or learner.iteration == num_iterations:
                path = run_checkpoint_path(self._context.run_dir, learner.iteration)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(learner.state_dict(include_optimizers=True), path)
                record_checkpoint(path)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def run_checkpoint_path(run_dir, iteration: int):
    return run_dir / "checkpoints" / f"ckpt_{iteration:06d}.pt"


class PlayRunner:
    """Deterministic playback of a trained SuperMimic policy checkpoint."""

    def __init__(self, env, learner: SuperMimicLearner):
        self._env = env
        self._learner = learner

    @torch.no_grad()
    def run(self, renderer) -> None:
        env = self._env
        learner = self._learner
        learner.pi.eval()
        mdp = learner.mdp
        motion = env.command_terms["motion"]
        env.init_state()
        obs_hist = None
        a_prev = torch.zeros((env.num_envs, mdp.num_joints), device=learner.device)
        while renderer.render():
            frames = torch.as_tensor(np.asarray(motion.steps[:, 0]), dtype=torch.long, device=learner.device)
            state = mdp.unflatten_state(
                torch.as_tensor(np.asarray(motion.state), dtype=torch.float32, device=learner.device)
            )
            obs = mdp.obs(state, a_prev, frames)
            if obs_hist is None:
                obs_hist = obs.unsqueeze(1).expand(-1, learner.cfg.hist, -1).clone()
            else:
                obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            goal = mdp.goal(frames, learner.goal_frames)
            action = learner.pi(obs_hist, goal).clamp(-1.0, 1.0)
            env.step(action.detach().cpu().numpy().astype(np.float32))
            a_prev = action


def attach_play(trainer: Trainer, policy: str) -> None:
    """Load a checkpoint into a play-mode trainer and run the rollout loop."""
    trainer._set_seed(trainer._context.seed)
    env = trainer._make_env(trainer._context.play_num_envs, mode="play")
    renderer = create_renderer(env, trainer._render)
    if renderer is None:
        raise ValueError("motrix.supermimic play requires a render configuration.")
    learner = SuperMimicLearner(
        env,
        trainer._rlcfg.agent,
        build_supermimic_mdp(env),
        device=trainer._device(),
        action_low=np.full(env.command_terms["motion"].clip.joint_pos.shape[1], -1.0, np.float32),
        action_high=np.full(env.command_terms["motion"].clip.joint_pos.shape[1], 1.0, np.float32),
        goal_frames=trainer._rlcfg.agent.goal_frames,
        seed=trainer._context.seed,
    )
    checkpoint = torch.load(policy, map_location=learner.device, weights_only=False)
    learner.pi.load_state_dict(checkpoint["pi"])
    PlayRunner(env, learner).run(renderer)

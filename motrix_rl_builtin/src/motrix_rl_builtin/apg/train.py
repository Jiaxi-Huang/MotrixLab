# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG training orchestration and deterministic policy playback."""

import copy
import random
import time
from dataclasses import fields

import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.renderer import create_renderer
from motrix_rl import checkpoints
from motrix_rl.console import EpisodeWindow, PanelSystemSampler, ScalarWindow, TrainingPanelSession, TrainingPanelStats
from motrix_rl.console.metrics import ROLLOUT_MEANS, system_scalars
from motrix_rl.contracts import TrainerBase
from motrix_rl.device import resolve_default_device
from motrix_rl.numa import available_cpu_ids, log_worker_resources, pin_worker_cpus, resolve_worker_cpus
from motrix_rl.replay import SequenceReplay
from motrix_rl.training_logger import TrainingLogger
from motrix_rl_builtin.apg.bridge import GenCoordLayout, TransitionAssembler
from motrix_rl_builtin.apg.collector import AsyncCollector, Collector, UpdateSchedule
from motrix_rl_builtin.apg.config import NetworkCfg
from motrix_rl_builtin.apg.learner import Learner
from motrix_rl_builtin.apg.networks import History, MotionActor, Policy
from motrix_rl_builtin.apg.observations import ApgObservationLayout


def restore_policy(checkpoint, device="cpu"):
    network = dict(checkpoint["network"])
    # Inference artifacts predate the agent configuration rename.
    if "mixed_precision" in network:
        network["amp"] = network.pop("mixed_precision")
    cfg = NetworkCfg(**network)
    actor = MotionActor(checkpoint["proprio_dim"], checkpoint["goal_dim"], checkpoint["action_dim"], cfg)
    weights = checkpoint["policy"]
    if checkpoint.get("policy_ema") is not None:
        weights = {
            key.removeprefix("module."): value
            for key, value in checkpoint["policy_ema"].items()
            if key.startswith("module.")
        }
    policy = Policy(actor, weights["low"], weights["high"])
    policy.load_state_dict(weights)
    return policy.to(device).eval()


class Trainer(TrainerBase):
    def __init__(self, *, context):
        self.context = context
        self.cfg = context.rl_cfg
        self.cfg.validate()
        self.spec = registry.resolve(context.env_name, sim=context.sim)
        if not issubclass(self.spec.env_cls, ManagerEnv):
            raise ValueError("APG requires a ManagerEnv with a derived modeling view")
        self.device = torch.device(self.cfg.device) if self.cfg.device else resolve_default_device()

    def _env(self, play=False, num_envs=None):
        cfg = copy.deepcopy(self.spec.env_cfg)
        if play:
            cfg = cfg.for_play()
        cfg.validate()
        return self.spec.env_cls(
            cfg,
            num_envs=self.context.play_num_envs if play else (num_envs or self.context.num_envs),
            backend=self.context.sim,
            seed=self.context.seed,
            capture_final_state=not play,
        )

    def train(self):
        if self.context.logging.backend != "tensorboard":
            raise ValueError("APG supports only the 'tensorboard' logging backend")
        torch.set_float32_matmul_precision("high")
        initial_cpus = available_cpu_ids()
        if self.cfg.asynchronous:
            pin_worker_cpus(resolve_worker_cpus(self.cfg.trainer.async_options.learner_cpu_cores, "learner_cpu_cores"))
        log_worker_resources("APG learner")
        seed = self.context.seed or 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.cfg.asynchronous and self.context.render:
            raise ValueError("APG asynchronous collection does not support render; use algo.asynchronous=false")
        env = self._env(num_envs=1 if self.cfg.asynchronous else None)
        collector = None
        renderer = create_renderer(env, self.context.render)
        log = TrainingLogger(self.context.run_dir, self.context.metrics)
        panel = TrainingPanelSession.open(title=f"{self.context.env_name}/motrix.apg")
        try:
            checkpoint = None
            if self.context.resume_from:
                checkpoint = torch.load(self.context.resume_from, map_location="cpu", weights_only=True)
                # Commands are constructed before init_state; restore global
                # curriculum now so the very first reset samples from it.
                for name, term in env.command_terms.items():
                    term.load_training_state_dict(checkpoint["commands"][name])
            state = env.init_state()
            view = env.modeling_view()
            layout = GenCoordLayout(env, view, self.device)
            learner = Learner(env, view, layout, self.cfg, self.device, seed=seed)
            assembler = TransitionAssembler(env, view, learner.proprio_dim)
            replay = SequenceReplay(
                schema=assembler.schema,
                num_envs=self.context.num_envs,
                capacity=self.cfg.agent.buffer_size,
                device=self.device,
                seed=seed,
                compile=self.cfg.agent.compile,
            )
            step = 0
            if checkpoint is not None:
                learner.load_state_dict(checkpoint)
                replay.load_rng_state(checkpoint["replay_rng"])
                torch.set_rng_state(checkpoint["torch_rng"])
                if self.device.type == "cuda":
                    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
                step = checkpoint["step"]
                log.event("resumed", step=step, path=str(self.context.resume_from))
            collector = (
                AsyncCollector(
                    self.spec,
                    self.cfg,
                    self.context,
                    learner.actor,
                    assembler.schema,
                    observation_layout=learner.observation_layout,
                    spawn_cpus=initial_cpus,
                    resume_step=step,
                    commands=checkpoint["commands"] if checkpoint else None,
                    rng=checkpoint.get("collector_rng") if checkpoint else None,
                )
                if self.cfg.asynchronous
                else Collector(
                    env,
                    state,
                    learner.actor,
                    learner.policy,
                    assembler,
                    self.cfg,
                    self.device,
                    observation_layout=learner.observation_layout,
                )
            )
            schedule = UpdateSchedule(
                self.cfg.agent, remainder=checkpoint.get("update_remainder", 0) if checkpoint else 0
            )
            warmup = schedule.warmup
            last_published_round = learner.world_model_iterations
            local_steps = 0
            episode_return = torch.zeros(self.context.num_envs)
            episode_length = torch.zeros(self.context.num_envs)
            episodes = EpisodeWindow(max_episodes=100)
            training_sums, training_counts = {}, {}
            reward_window = ScalarWindow()
            system_sampler = PanelSystemSampler()
            start_time = last_log_time = time.perf_counter()
            last_log_step = step
            collect_seconds = learn_seconds = collector_seconds = publish_seconds = weight_load_seconds = 0.0
            last_checkpoint_path = None
            last_checkpoint_step = None
            collected = None
            while step < self.cfg.trainer.num_learning_iterations:
                collect_start = time.perf_counter()
                next_step = step + 1
                interval = self.context.checkpoint.interval
                save = (
                    interval > 0 and next_step % interval == 0
                ) or next_step == self.cfg.trainer.num_learning_iterations
                with collector.borrow(
                    metrics=next_step % self.context.logging.interval == 0, checkpoint=save
                ) as collection:
                    collected = collection
                    collector_seconds += collected.seconds
                    weight_load_seconds += collected.weight_load_seconds
                    transition = collected.transition
                    device_transition = replay.extend(transition)
                    before = {key: device_transition[key] for key in view.fields}
                    learner.update_normalization(
                        before, device_transition["actions"], collected.observation, device_transition
                    )
                    episode_return += transition["rewards"]
                    done = (transition["terminated"] | transition["truncated"]).bool()
                    episode_length += 1
                    episodes.add(episode_return[done].tolist(), episode_length[done].tolist())
                    episode_length[done] = 0
                    reward_window.add(collected.reward_terms)
                    episode_return[done] = 0
                    if next_step % self.context.logging.interval == 0:
                        env_reward = float(transition["rewards"].mean())
                step += 1
                local_steps += 1
                collect_seconds += time.perf_counter() - collect_start
                if local_steps == warmup:
                    log.event("warmup_done", step=step)
                updates = schedule.advance(local_steps)
                if updates:
                    learn_start = time.perf_counter()
                    for _ in range(updates):
                        for key, value in learner.update(replay).items():
                            training_sums[key] = training_sums.get(key, 0) + value.detach()
                            training_counts[key] = training_counts.get(key, 0) + 1
                    learn_seconds += time.perf_counter() - learn_start
                if self.cfg.asynchronous and (
                    local_steps <= warmup
                    or (
                        updates
                        and learner.world_model_iterations - last_published_round
                        >= self.cfg.trainer.async_options.weight_publish_interval
                    )
                ):
                    # One atomic actor/normalizer bundle per learner round by default.
                    publish_start = time.perf_counter()
                    collector.publish()
                    publish_elapsed = time.perf_counter() - publish_start
                    publish_seconds += publish_elapsed
                    learn_seconds += publish_elapsed
                    last_published_round = learner.world_model_iterations
                if step % self.context.logging.interval == 0:
                    window_steps = step - last_log_step
                    episode_stats = episodes.snapshot()
                    mean_return = episode_stats.mean_return if episode_stats.episodes else float("nan")
                    mean_length = episode_stats.mean_length if episode_stats.episodes else float("nan")
                    # One device-to-host transfer per log window; no scalar reads per learner round.
                    names = list(training_sums)
                    values = (
                        torch.stack([training_sums[k] / training_counts[k] for k in names]).tolist() if names else []
                    )
                    now = time.perf_counter()
                    sps = window_steps * self.context.num_envs / max(now - last_log_time, 1e-6)
                    metrics = dict(zip(names, values))
                    if self.cfg.asynchronous:
                        metrics["weight_version_lag"] = collector.sender.version - collected.weight_version
                        metrics["ring_fill"] = collector.ring.size()
                    training_sums.clear()
                    training_counts.clear()
                    reward_terms = reward_window.drain()
                    env_metrics = collected.metrics
                    collect_ms = collect_seconds * 1000 / window_steps
                    learn_ms = learn_seconds * 1000 / window_steps
                    learn_percent = 100 * learn_seconds / max(collect_seconds + learn_seconds, 1e-9)
                    system = system_sampler.sample()
                    stats = TrainingPanelStats(
                        iteration=step,
                        total_iterations=self.cfg.trainer.num_learning_iterations,
                        steps_per_second=sps,
                        elapsed_seconds=now - start_time,
                        mean_return=mean_return,
                        mean_episode_length=mean_length,
                        episodes=episode_stats.episodes,
                        buffer_size=len(replay),
                        buffer_capacity=replay.capacity * self.context.num_envs,
                        collect_ms=collect_ms,
                        learn_ms=learn_ms,
                        learn_percent=learn_percent,
                        warming=local_steps < warmup,
                        training_metrics=metrics,
                        reward_terms=reward_terms,
                        env_metrics=env_metrics,
                        checkpoint_path=last_checkpoint_path,
                        cpu_load=system.cpu_load,
                        gpu_utilization_percent=system.gpu_utilization_percent,
                        memory_usage=system.memory_usage,
                        gpu_memory_usage=system.gpu_memory_usage,
                        gpu_devices=system.gpu_devices,
                    )
                    panel.submit(stats)
                    scalars = {
                        **{f"train/{k}": v for k, v in metrics.items()},
                        "rollout/env_reward": env_reward,
                        "rollout/mean_return": mean_return,
                        "rollout/mean_ep_len": mean_length,
                        "perf/env_steps_per_s": sps,
                        "perf/collect_ms_per_iter": collect_ms,
                        "perf/learn_ms_per_iter": learn_ms,
                        "perf/learn_pct": learn_percent,
                        "perf/collector_ms_per_iter": collector_seconds * 1000 / window_steps,
                        "perf/publish_ms_per_iter": publish_seconds * 1000 / window_steps,
                        "perf/weight_load_ms_per_iter": weight_load_seconds * 1000 / window_steps,
                        **{f"reward/{k}": v for k, v in reward_terms.items()},
                        **{f"metrics/{k}": v for k, v in env_metrics.items()},
                    }
                    log.scalars(
                        scalars,
                        step,
                        env_steps=step * self.context.num_envs,
                        live_extra=lambda: system_scalars(stats),
                        placeholders=() if episode_stats.episodes else ROLLOUT_MEANS,
                    )
                    last_log_time, last_log_step = now, step
                    collect_seconds = learn_seconds = collector_seconds = publish_seconds = weight_load_seconds = 0.0
                interval = self.context.checkpoint.interval
                if interval > 0 and step % interval == 0:
                    last_checkpoint_path = str(
                        self._save(
                            env,
                            learner,
                            replay,
                            step,
                            schedule.remainder,
                            collected,
                        )
                    )
                    last_checkpoint_step = step
                    log.event("checkpoint_saved", step=step, path=last_checkpoint_path)
                    panel.checkpoint(last_checkpoint_path)
                if renderer is not None and renderer.render() is False:
                    break
            if last_checkpoint_step != step:
                path = str(
                    self._save(
                        env,
                        learner,
                        replay,
                        step,
                        schedule.remainder,
                        collected,
                    )
                )
                log.event("checkpoint_saved", step=step, path=path)
                panel.checkpoint(path)
        finally:
            if isinstance(collector, AsyncCollector):
                collector.close()
            panel.close()
            if renderer is not None:
                renderer.close()
            log.close()

    def _save(self, env, learner, replay, step, update_remainder, collected=None):
        directory = self.context.checkpoint_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"model_{step:07d}.pt"
        network = {field.name: getattr(self.cfg.agent, field.name) for field in fields(NetworkCfg)}
        torch.save(
            {
                **learner.state_dict(),
                "network": network,
                "step": step,
                "update_remainder": update_remainder,
                "collector_rng": collected.rng if collected is not None else None,
                "commands": collected.commands
                if collected is not None and collected.commands is not None
                else {name: term.training_state_dict() for name, term in env.command_terms.items()},
                "replay_rng": replay.rng_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            },
            path,
        )
        checkpoints.record_checkpoint_artifact(
            self.context.run_dir,
            checkpoints.LATEST_TRAINING_STATE,
            path,
            checkpoints.TRAINING_STATE,
            checkpoint_format="pt",
        )
        policy_path = directory / "policy.pt"
        torch.save({**learner.policy_state(), "network": network}, policy_path)
        checkpoints.record_checkpoint_artifact(
            self.context.run_dir, checkpoints.BEST_POLICY, policy_path, checkpoints.POLICY, checkpoint_format="pt"
        )

        return path

    def play(self, policy):
        actor = restore_policy(torch.load(policy, map_location="cpu", weights_only=True), self.device)
        env = self._env(play=True)
        renderer = create_renderer(env, self.context.render)
        try:
            state = env.init_state()
            observation_layout = ApgObservationLayout.from_env(env)
            observation_layout.validate_policy_dimensions(
                actor.actor.proprio_dim, actor.actor.goal_dim, actor.low.numel()
            )
            obs = state.obs.policy.to(self.device)
            history = History(observation_layout.proprio(obs), actor.actor.history_length)
            with torch.inference_mode():
                while True:
                    action = actor(torch.cat([history.value.flatten(1), observation_layout.goal(obs)], -1))
                    state = env.step(action.cpu())
                    obs = state.obs.policy.to(self.device)
                    history.advance(
                        observation_layout.proprio(obs), (state.terminated | state.truncated).to(self.device)
                    )
                    if renderer is not None and renderer.render() is False:
                        break
        finally:
            if renderer is not None:
                renderer.close()

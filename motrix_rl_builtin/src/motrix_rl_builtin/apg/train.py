# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WM/BPTT sync trainer wired through the standard control plane.

Driven by ``TrainerContext``: build the registered environment with
final-state capture injected (S-B — a runtime consumer request, never a
task-cfg field), derive its modeling view (D3 — derivation failure of a
configured task is a configuration-time error), assemble the state layout,
then interleave collection with world-model/BPTT updates plus checkpoints
and real-environment evaluation. Resume restores the learning state
(modules, normalizers, optimizers, RNG, counters) and keeps a freshly
collected warm-up replay; it is not a step-exact replay of the pre-resume
stream. Sampler curriculum statistics restart with that warm-up (the motion
sampler exposes no serialization yet — the one D8 deviation of this
provider version).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from motrix_env_core import registry as env_registry
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.renderer import create_renderer
from motrix_env_core.sim import KinematicTreeQuery
from motrix_rl import checkpoints
from motrix_rl.console import (
    PanelSystemSampler,
    ScalarWindow,
    TimingWindow,
    TrainingPanelSession,
    TrainingPanelStats,
)
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.normalize import EmpiricalNormalization
from motrix_rl.replay import SequenceReplay
from motrix_rl.training_logger import TrainingLogger
from motrix_rl.transport.schema import TransitionField, wm_transition_schema
from motrix_rl_builtin.apg.assemble import build_assembly
from motrix_rl_builtin.apg.collector import ApgCollector, _term_leaves
from motrix_rl_builtin.apg.config import ApgCfg
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.networks import (
    FloatingBaseDynamics,
    GoalConditionedActor,
    MlpActor,
    MlpCritic,
    MlpDynamics,
)

torch.set_float32_matmul_precision("high")

_HEADLESS_PLAY_STEPS = 5_000


def _collect_chunks(steps_per_iteration: float, rounds: int, budget: float) -> tuple[list[int], float]:
    """Split one iteration's collection into per-round chunks (UTD accounting).

    Fractional per-iteration steps carry their remainder across iterations —
    upstream's ``1/utd`` accumulator: ``2.5`` yields alternating 2/3-step
    iterations.
    """
    total = int(budget + steps_per_iteration)
    budget = budget + steps_per_iteration - total
    base = total // rounds
    chunks = [base + (1 if r < total % rounds else 0) for r in range(rounds)]
    return chunks, budget


def _make_dynamics(cfg: ApgCfg, num_joints: int | None, flat_width: int, action_dim: int):
    if cfg.world_model.scheme == "residual":
        return MlpDynamics(flat_width, action_dim, cfg.world_model.hidden)
    if cfg.world_model.scheme == "floating_base":
        assert num_joints is not None
        return FloatingBaseDynamics(num_joints, action_dim, cfg.world_model.hidden)
    raise ValueError(f"Unknown world_model.scheme {cfg.world_model.scheme!r}; expected 'residual' or 'floating_base'.")


@dataclass
class _Setup:
    """Products of the per-run construction phase."""

    env: ManagerEnv
    collector: ApgCollector
    learner: ApgLearner
    task_dims: dict[str, Any]


class Trainer(TrainerBase):
    """Standard-form sync trainer for the APG provider."""

    def __init__(self, *, context: TrainerContext[ApgCfg]) -> None:
        self._env_name = context.env_name
        self._rlcfg = context.rl_cfg
        self._env_spec = env_registry.resolve(context.env_name, sim=context.sim)
        self._context = context
        self.history: list[dict[str, float]] = []
        self.eval_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ setup
    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return resolve_default_device()

    def _make_env(self, num_envs: int, *, mode: str = "train") -> ManagerEnv:
        """Build the env with the APG runtime injections applied to a cfg copy."""
        cfg = copy.deepcopy(self._env_spec.env_cfg)
        if mode == "play":
            cfg = cfg.for_play()
        if mode == "train":
            # The differentiable FK tree enters as a declared model constant
            # (device pinned to the learner's) and final-state capture is a
            # constructor injection — neither is task-cfg state.
            cfg.queries.model["apg_kinematic_tree"] = KinematicTreeQuery(
                body=cfg.scene.objs.robot.resolved_base_link_name, device=str(self._device())
            )
        cfg.validate()
        env_cls = self._env_spec.env_cls
        if env_cls is None or not issubclass(env_cls, ManagerEnv):
            raise TypeError(f"motrix.apg requires manager-based environments, {self._env_name} is not one.")
        env = ManagerEnv(
            cfg,
            num_envs=num_envs,
            backend=self._env_spec.sim,
            seed=self._context.seed,
            capture_final_state=mode == "train",
        )
        env.init_state()
        return env

    def _setup(self, env: ManagerEnv) -> _Setup:
        cfg = self._rlcfg
        device = self._device()
        view = env.modeling_view()
        tree = env.model.others.get("apg_kinematic_tree")
        assembly = build_assembly(view, env, tree, cfg.world_model.scheme)
        action_dim = env.action_space.shape[0]
        low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
        action_limit = float(high.max())
        bounds = None if bool(torch.equal(low, -high)) else (float(low.min()), float(high.max()))

        # The WM schema's state fields are the declared read set plus the
        # exogenous channels, the latter named BY CHANNEL so imagination
        # re-seeds the shadow host straight from replayed values. Channel
        # shapes come from the resident command term leaves.
        state_fields = [TransitionField(key, tuple(desc.shape), torch.float32) for key, desc in view.fields.items()]
        for channel in view.exogenous:
            term_name, path = channel.split(".", 1)
            leaf = _term_leaves(env.command_terms[term_name], env.num_envs)[path]
            state_fields.append(TransitionField(channel, tuple(leaf.shape[1:]), torch.float32))
        obs_dim = getattr(env.state.obs, cfg.policy_group).shape[-1]
        value_dim = getattr(env.state.obs, cfg.value_group).shape[-1]
        # Upstream v_max: the task's maximum discounted return per upstream's
        # clamp (ctrl_dt * positive weight sum / (1 - gamma)), suppressing
        # critic overestimation.
        positive_weight_sum = sum(
            reward_cfg.weight for reward_cfg in env.cfg.reward_cfgs().values() if reward_cfg.weight > 0.0
        )
        value_max = float(env.cfg.ctrl_dt * positive_weight_sum / (1.0 - cfg.gamma))
        # Flat goal columns: the declared value-group terms the actor consumes
        # without stacking (upstream stacks proprioception only).
        goal_columns = None
        goal_dim = 0
        if cfg.goal_terms:
            group = env.observation_groups[cfg.value_group]
            cursor = 0
            columns: list[int] = []
            found = set()
            for entry in group.terms:
                if entry.name in cfg.goal_terms:
                    columns.extend(range(cursor, cursor + entry.size))
                    found.add(entry.name)
                cursor += entry.size
            missing = sorted(set(cfg.goal_terms) - found)
            if missing:
                raise ValueError(
                    f"goal_terms {missing} are not observation terms of group {cfg.value_group!r}; "
                    f"available: {[entry.name for entry in group.terms]}."
                )
            goal_columns = torch.tensor(columns, dtype=torch.long)
            goal_dim = len(columns)
        schema = wm_transition_schema(state_fields, action_dim, obs_dim=obs_dim)
        replay = SequenceReplay(
            schema=schema,
            num_envs=env.num_envs,
            capacity=cfg.replay.capacity,
            device=device,
            seed=self._context.seed or 0,
        )
        num_joints = len(tree.joint_names) if tree is not None else None
        learner = ApgLearner(
            program=view.program,
            assembly=assembly,
            dynamics=_make_dynamics(cfg, num_joints, assembly.flat_width, action_dim),
            actor=(
                GoalConditionedActor(
                    obs_dim * cfg.actor.history + goal_dim,
                    goal_dim,
                    action_dim,
                    cfg.actor.hidden,
                    cfg.actor.goal_latent_dim,
                    output_activation=cfg.actor.output_activation,
                )
                if goal_dim
                else MlpActor(
                    obs_dim * cfg.actor.history,
                    action_dim,
                    cfg.actor.hidden,
                    output_activation=cfg.actor.output_activation,
                )
            ),
            critic=MlpCritic(value_dim, cfg.critic.hidden),
            replay=replay,
            cfg=cfg,
            device=device,
            obs_dim=obs_dim,
            value_dim=value_dim,
            action_limit=action_limit,
            value_max=value_max,
            goal_columns=goal_columns,
        )
        collector = ApgCollector(
            env=env,
            view=view,
            assembly=assembly,
            replay=replay,
            device=device,
            rng=np.random.default_rng(self._context.seed),
            actor=learner.actor,
            obs_norm=learner.obs_norm,
            goal_norm=learner.goal_norm,
            goal_columns=goal_columns,
            action_limit=action_limit,
            action_bounds=bounds,
            history_length=cfg.actor.history,
            policy_group=cfg.policy_group,
            exploration_noise_std=cfg.collector.exploration_noise_std,
            exploration_clip=cfg.collector.exploration_clip,
            on_transition=learner.update_dynamics_statistics,
        )
        task_dims = {
            "action_dim": action_dim,
            "action_limit": action_limit,
            "obs_dim": obs_dim * cfg.actor.history + goal_dim,
            "raw_obs_dim": obs_dim,
            "value_dim": value_dim,
            "goal_dim": goal_dim,
            "goal_latent_dim": cfg.actor.goal_latent_dim,
            "history": cfg.actor.history,
            "value_max": value_max,
            "fields": list(view.fields),
            "channels": list(view.exogenous),
        }
        return _Setup(env=env, collector=collector, learner=learner, task_dims=task_dims)

    @staticmethod
    def _set_seed(seed: int | None) -> None:
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
        cfg = self._rlcfg
        run_dir = self._context.run_dir
        training_log = TrainingLogger(run_dir, self._context.metrics)
        try:
            env = self._make_env(self._context.num_envs)
            setup = self._setup(env)
            collector, learner, task_dims = setup.collector, setup.learner, setup.task_dims

            iteration = 0
            env_steps = 0
            if self._context.resume_from:
                iteration, env_steps = self._restore(self._context.resume_from, learner)
                print(f"[motrix.apg] resumed from {self._context.resume_from} at iteration={iteration}")
                training_log.event("resumed", iteration=iteration, env_steps=env_steps)
                if iteration >= cfg.trainer.iterations:
                    print(
                        f"[motrix.apg] WARNING: checkpoint iteration ({iteration}) >= target "
                        f"iters ({cfg.trainer.iterations}); nothing to train."
                    )
            print(
                f"[motrix.apg] training '{self._env_name}' on {self._device()} "
                f"num_envs={env.num_envs} iters={cfg.trainer.iterations} (from {iteration})"
            )
            training_log.event("run_started", env=self._env_name, device=str(self._device()))

            if cfg.trainer.eval_interval:
                baseline = collector.evaluate(cfg.trainer.eval_steps)
                self.eval_history.append({"iteration": float(iteration), **baseline})
                training_log.scalars(baseline, step=iteration, env_steps=env_steps)

            if iteration < cfg.trainer.iterations:
                # Replay is fresh even on resume. Fill it with the restored
                # policy before learning, without granting extra UTD credit.
                prefill = max(cfg.replay.warmup_steps, cfg.world_model.window)
                collector.collect(prefill)
                env_steps += prefill * env.num_envs
                training_log.event("warmup_done", transitions=len(learner.replay), env_steps=env_steps)

            def _record_periodic_checkpoint(path, step_iteration: int, step_env_steps: int) -> None:
                self._save(path, learner, task_dims, step_iteration, step_env_steps)
                checkpoints.record_checkpoint_artifact(
                    run_dir,
                    checkpoints.LATEST_TRAINING_STATE,
                    path,
                    checkpoints.TRAINING_STATE,
                    checkpoint_format=self._context.checkpoint_format,
                )
                training_log.event("checkpoint_saved", path=str(path), iteration=step_iteration)

            env_steps = self._run_loop(
                collector,
                learner,
                training_log,
                num_iterations=cfg.trainer.iterations,
                start_iteration=iteration,
                start_env_steps=env_steps,
                logging_interval=self._context.logging.interval,
                save_interval=self._context.checkpoint.interval,
                record_checkpoint=_record_periodic_checkpoint,
            )

            final_path = checkpoints.final_checkpoint_path(self._context.checkpoint_format, run_dir)
            self._save(final_path, learner, task_dims, cfg.trainer.iterations, env_steps)
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.LATEST_TRAINING_STATE,
                final_path,
                checkpoints.TRAINING_STATE,
                checkpoint_format=self._context.checkpoint_format,
            )
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.BEST_POLICY,
                final_path,
                checkpoints.POLICY,
                checkpoint_format=self._context.checkpoint_format,
            )
            print(f"[motrix.apg] saved checkpoint to {final_path}")
            training_log.event("checkpoint_saved", path=str(final_path), iteration=cfg.trainer.iterations)
            last_eval = self.eval_history[-1] if self.eval_history else {}
            if last_eval.get("iteration") != cfg.trainer.iterations and cfg.trainer.final_eval:
                evals = collector.evaluate(cfg.trainer.eval_steps)
                self.eval_history.append({"iteration": float(cfg.trainer.iterations), **evals})
                training_log.scalars(evals, step=cfg.trainer.iterations, env_steps=env_steps)
        finally:
            training_log.close()

    # ---------------------------------------------------- interleaved loop
    def _run_loop(
        self,
        collector: ApgCollector,
        learner: ApgLearner,
        training_log: TrainingLogger,
        *,
        num_iterations: int,
        start_iteration: int,
        start_env_steps: int,
        logging_interval: int,
        save_interval: int,
        record_checkpoint,
    ) -> int:
        """Interleaved collect-update loop (a few fresh steps per update)."""
        cfg = self._rlcfg
        env_steps = start_env_steps
        decay_iterations = cfg.trainer.lr_decay_iterations or num_iterations
        start_time = time.time()
        timing = TimingWindow()
        scalars_window = ScalarWindow()
        system_sampler = PanelSystemSampler()
        panel = TrainingPanelSession.open(title=f"{self._env_name}/motrix.apg")
        last_checkpoint_path: str | None = None
        collect_budget = 0.0
        try:
            for step in range(start_iteration + 1, num_iterations + 1):
                metrics: dict[str, float] = {"iteration": float(step)}
                lr_scale = self._lr_scale_at(step, decay_iterations)
                learner.set_learning_rate_scale(lr_scale)
                metrics["apg.lr"] = cfg.actor.learning_rate * lr_scale

                policy_updates = cfg.actor.updates_per_iteration if step > cfg.trainer.warmup_iterations else 0
                rounds = max(policy_updates, 1)
                chunks, collect_budget = _collect_chunks(
                    cfg.trainer.collect_steps_per_iteration, rounds, collect_budget
                )
                for index in range(rounds):
                    began = time.perf_counter()
                    round_metrics = collector.collect(chunks[index])
                    metrics.update(round_metrics)
                    scalars_window.add(round_metrics, track_range=True)
                    timing.add("collect", time.perf_counter() - began)
                    if index == 0 or policy_updates:
                        began = time.perf_counter()
                        wm_metrics: dict[str, torch.Tensor] | None = None
                        if index == 0:
                            for _ in range(cfg.world_model.batches_per_iteration):
                                wm_metrics = learner.update_world_model()
                        if policy_updates:
                            update_metrics = learner.policy_update()
                            metrics.update(update_metrics)
                            scalars_window.add(update_metrics, track_range=True)
                        if wm_metrics is not None:
                            folded = {key: float(value) for key, value in wm_metrics.items()}
                            metrics.update(folded)
                            scalars_window.add(folded, track_range=True)
                        timing.add("learn", time.perf_counter() - began)
                env_steps += sum(chunks) * collector.env.num_envs
                self.history.append(metrics)

                if save_interval and step % save_interval == 0:
                    path = self._context.checkpoint_dir / f"model_{step:07d}.pt"
                    record_checkpoint(path, step, env_steps)
                    last_checkpoint_path = str(path)
                if step % logging_interval == 0:
                    self._emit_window(
                        collector,
                        learner,
                        training_log,
                        panel,
                        system_sampler,
                        scalars_window,
                        timing,
                        step=step,
                        num_iterations=num_iterations,
                        env_steps=env_steps,
                        start_time=start_time,
                        lr=metrics["apg.lr"],
                        checkpoint_path=last_checkpoint_path,
                        warming=step <= cfg.trainer.warmup_iterations,
                    )
                if cfg.trainer.eval_interval and step % cfg.trainer.eval_interval == 0:
                    began = time.perf_counter()
                    evals = collector.evaluate(cfg.trainer.eval_steps)
                    timing.add("eval", time.perf_counter() - began)
                    self.eval_history.append({"iteration": float(step), **evals})
                    training_log.scalars(evals, step=step, env_steps=env_steps)
            return env_steps
        finally:
            panel.close()

    def _emit_window(
        self,
        collector: ApgCollector,
        learner: ApgLearner,
        training_log: TrainingLogger,
        panel: TrainingPanelSession,
        system_sampler: PanelSystemSampler,
        scalars_window: ScalarWindow,
        timing: TimingWindow,
        *,
        step: int,
        num_iterations: int,
        env_steps: int,
        start_time: float,
        lr: float,
        checkpoint_path: str | None,
        warming: bool,
    ) -> None:
        drained = scalars_window.drain(include_range=True)
        episode = collector.episode_summary()
        elapsed = time.time() - start_time
        steps_per_second = env_steps / max(elapsed, 1e-9)
        system = system_sampler.sample()
        stats = TrainingPanelStats(
            iteration=step,
            total_iterations=num_iterations,
            steps_per_second=steps_per_second,
            elapsed_seconds=elapsed,
            mean_return=episode.mean_return if episode.mean_return is not None else float("nan"),
            mean_episode_length=episode.mean_length if episode.mean_length is not None else float("nan"),
            episodes=episode.episodes,
            buffer_size=len(learner.replay),
            buffer_capacity=collector.env.num_envs * self._rlcfg.replay.capacity,
            collect_ms=timing.mean_ms("collect"),
            learn_ms=timing.mean_ms("learn"),
            learn_percent=100.0
            * timing.total_seconds("learn")
            / max(timing.total_seconds("collect") + timing.total_seconds("learn"), 1e-9),
            warming=warming,
            training_metrics=drained,
            reward_terms={},
            env_metrics={},
            cpu_load=system.cpu_load,
            gpu_utilization_percent=system.gpu_utilization_percent,
            memory_usage=system.memory_usage,
            gpu_memory_usage=system.gpu_memory_usage,
            gpu_devices=system.gpu_devices,
            checkpoint_path=checkpoint_path,
        )
        panel.submit(stats)
        training_log.scalars(
            {
                **drained,
                "apg.lr": lr,
                "perf/env_steps_per_s": steps_per_second,
            },
            step=step,
            env_steps=env_steps,
        )
        timing.reset()

    def _lr_scale_at(self, step: int, decay_iterations: int) -> float:
        """Shared cosine decay to 3% of each optimizer's configured base rate."""
        if step >= decay_iterations:
            return 0.03
        return 0.03 + 0.97 * 0.5 * (1.0 + np.cos(np.pi * step / decay_iterations))

    # ------------------------------------------------------------ checkpointing
    def _save(self, path, learner: ApgLearner, task_dims: dict, iteration: int, env_steps: int) -> None:
        payload = {
            "iteration": iteration,
            "env_steps": env_steps,
            "task": dict(task_dims),
            **learner.learning_state(),
            "rng": {
                "numpy": learner.replay.rng_state(),
                "torch": torch.get_rng_state(),
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def _restore(self, path: str, learner: ApgLearner) -> tuple[int, int]:
        payload = torch.load(path, map_location=self._device(), weights_only=False)
        learner.load_learning_state(payload)
        if "rng" in payload:
            learner.replay.load_rng_state(payload["rng"]["numpy"])
            torch.set_rng_state(payload["rng"]["torch"].cpu())
        return int(payload["iteration"]), int(payload["env_steps"])

    # ---------------------------------------------------------------- playback
    def play(self, policy: str) -> None:
        self._set_seed(self._context.seed)
        device = self._device()
        cfg = self._rlcfg
        env = self._make_env(self._context.play_num_envs, mode="play")
        payload = torch.load(policy, map_location=device, weights_only=False)
        dims = payload["task"]
        goal_columns_for_play = None
        if self._rlcfg.goal_terms and dims.get("goal_dim", 0):
            group = env.observation_groups[self._rlcfg.value_group]
            cursor = 0
            columns: list[int] = []
            for entry in group.terms:
                if entry.name in self._rlcfg.goal_terms:
                    columns.extend(range(cursor, cursor + entry.size))
                cursor += entry.size
            goal_columns_for_play = torch.tensor(columns, dtype=torch.long, device=device)
        goal_dim = int(dims.get("goal_dim", 0))
        actor = (
            GoalConditionedActor(
                dims["obs_dim"],
                goal_dim,
                dims["action_dim"],
                cfg.actor.hidden,
                int(dims.get("goal_latent_dim", cfg.actor.goal_latent_dim)),
                output_activation=cfg.actor.output_activation,
            )
            if goal_dim
            else MlpActor(
                dims["obs_dim"],
                dims["action_dim"],
                cfg.actor.hidden,
                output_activation=cfg.actor.output_activation,
            )
        ).to(device)
        actor.load_state_dict(payload["actor"])
        actor.eval()
        obs_norm = EmpiricalNormalization(shape=dims["raw_obs_dim"], device=device)
        obs_norm.load_state_dict(payload["obs_norm"])
        obs_norm.eval()
        goal_dim = int(dims.get("goal_dim", 0))
        goal_norm = None
        if goal_dim:
            goal_norm = EmpiricalNormalization(shape=goal_dim, device=device)
            goal_norm.load_state_dict(payload["goal_norm"])
            goal_norm.eval()
            goal_columns = goal_columns_for_play
        else:
            goal_columns = None
        history: torch.Tensor | None = None
        renderer = create_renderer(env, self._context.render)
        try:
            with torch.no_grad():
                for _ in range(_HEADLESS_PLAY_STEPS):
                    raw_obs = getattr(env.state.obs, cfg.policy_group).to(device)
                    if history is None or history.shape[0] != raw_obs.shape[0]:
                        history = raw_obs.new_zeros((raw_obs.shape[0], dims["history"], raw_obs.shape[-1]))
                    pushed = torch.cat((history[:, 1:], raw_obs.unsqueeze(1)), dim=1)
                    fresh = torch.as_tensor(env.state.episode_steps == 0, device=device, dtype=torch.bool)
                    history = torch.where(fresh[:, None, None], raw_obs.unsqueeze(1).expand_as(pushed), pushed)
                    batch, length, dim = history.shape
                    normalized = obs_norm(history.reshape(batch * length, dim), update=False).reshape(history.shape)
                    features = normalized.flatten(1)
                    if goal_dim:
                        goal = getattr(env.state.obs, cfg.value_group).to(device)[:, goal_columns]
                        features = torch.cat((features, goal_norm(goal, update=False)), dim=-1)
                    command = dims["action_limit"] * actor(features)
                    env.step(command.to(raw_obs.device))
                    if renderer is not None and renderer.render() is False:
                        return
        finally:
            if renderer is not None:
                renderer.close()

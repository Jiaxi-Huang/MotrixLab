# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared APG collection and bounded spawn execution using the RL transports."""

import copy
import logging
import pickle
import random
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from motrix_rl.device import resolve_default_device
from motrix_rl.numa import log_worker_resources, pin_worker_cpus, resolve_worker_cpus, spawn_worker_cpus
from motrix_rl.transport.common import Control, bind_flat_params
from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl.transport.weight_channel import HostWeightReceiver, HostWeightSender, WeightChannelShared
from motrix_rl_builtin.apg.bridge import TransitionAssembler
from motrix_rl_builtin.apg.networks import History, MotionActor, Policy
from motrix_rl_builtin.apg.observations import ApgObservationLayout


def normalizers(actor):
    return {"obs": actor.obs_normalizer, "goal": actor.goal_normalizer}


@dataclass
class Collection:
    transition: dict[str, torch.Tensor]
    observation: torch.Tensor
    reward_terms: dict[str, float]
    metrics: dict[str, float]
    commands: dict | None
    seconds: float
    weight_version: int = 0
    rng: dict | None = None
    weight_load_seconds: float = 0.0


class Collector:
    """Single owner of the environment, action history and terminal-state assembly."""

    def __init__(self, env, state, actor, policy, assembler, cfg, device, *, observation_layout):
        self.observation_layout = observation_layout
        self.env, self.actor, self.policy = env, actor, policy
        self.assembler, self.cfg, self.device = assembler, cfg, device
        self.obs = state.obs.policy.to(device).clone()
        self.history = History(self.observation_layout.proprio(self.obs), cfg.agent.history_length)

    @torch.no_grad()
    def collect(self, *, metrics=False, checkpoint=False):
        start = time.perf_counter()
        before = self.assembler.snapshot()
        observation = self.obs
        actions = self.actor(self.history.value, self.observation_layout.goal(observation))
        actions += self.cfg.agent.exploration_noise * torch.randn_like(actions)
        actions = actions.clamp(self.policy.low, self.policy.high).cpu()
        state = self.env.step(actions)
        transition = self.assembler.assemble(before, actions, self.observation_layout.proprio(observation).cpu(), state)
        self.obs = state.obs.policy.to(self.device).clone()
        self.history.advance(self.observation_layout.proprio(self.obs), state.done.to(self.device))
        return Collection(
            transition,
            observation,
            {name: float(value.mean()) for name, value in state.reward_terms.items()},
            state.process_metrics() if metrics else {},
            {name: term.training_state_dict() for name, term in self.env.command_terms.items()} if checkpoint else None,
            time.perf_counter() - start,
        )

    @contextmanager
    def borrow(self, **kwargs):
        """Expose a batch for immediate replay ingestion and statistics updates."""
        yield self.collect(**kwargs)


def _run_collector(
    spec,
    cfg,
    num_envs,
    seed,
    sim,
    resume_step,
    commands,
    rng,
    ring,
    shared,
    params,
    control,
    messages,
    log_interval,
    save_interval,
    observation_layout,
):
    """Spawn entry point; tensor payloads use the bounded ring, metadata uses a pipe."""
    try:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pin_worker_cpus(resolve_worker_cpus(cfg.trainer.async_options.collector_cpu_cores, "collector_cpu_cores"))
        log_worker_resources("APG collector")
        torch.set_float32_matmul_precision("high")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        opts = cfg.trainer.async_options
        device = (
            (torch.device(cfg.device) if cfg.device else resolve_default_device())
            if opts.collector_inference_device == "auto"
            else torch.device(opts.collector_inference_device)
        )
        env_cfg = copy.deepcopy(spec.env_cfg)
        env_cfg.validate()
        env = spec.env_cls(env_cfg, num_envs=num_envs, seed=seed, backend=sim, capture_final_state=True)
        if commands is not None:
            for name, term in env.command_terms.items():
                term.load_training_state_dict(commands[name])
        state = env.init_state()
        view = env.modeling_view()
        if ApgObservationLayout.from_env(env) != observation_layout:
            raise ValueError("APG collector observation layout differs from learner environment")
        actor = (
            MotionActor(
                observation_layout.proprio_dim,
                observation_layout.goal_dim,
                observation_layout.action_dim,
                cfg.agent,
            )
            .to(device)
            .eval()
        )
        policy = Policy(actor, env.action_space.low, env.action_space.high).to(device)
        receiver = HostWeightReceiver(shared, params)
        flat_params = {"actor": bind_flat_params(actor)}
        receiver.maybe_load({"actor": actor}, normalizers(actor), flat_params)
        if opts.collector_compile and device.type == "cuda":
            actor.compile()
        collector = Collector(
            env,
            state,
            actor,
            policy,
            TransitionAssembler(env, view, observation_layout.proprio_dim),
            cfg,
            device,
            observation_layout=observation_layout,
        )
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if device.type == "cuda" and rng["cuda"] is not None:
                torch.cuda.set_rng_state(rng["cuda"], device)
        for step in range(resume_step + 1, cfg.trainer.num_learning_iterations + 1):
            while ring.is_full() and not control.stop:
                time.sleep(opts.idle_sleep_s)
            if control.stop:
                return
            weight_start = time.perf_counter()
            receiver.maybe_load({"actor": actor}, normalizers(actor), flat_params)
            weight_seconds = time.perf_counter() - weight_start
            save = (save_interval > 0 and step % save_interval == 0) or step == cfg.trainer.num_learning_iterations
            result = collector.collect(metrics=step % log_interval == 0, checkpoint=save)
            result.weight_version = receiver.loaded_version
            result.weight_load_seconds = weight_seconds
            if save:
                result.rng = {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                }
            ring.push({**result.transition, "collector_observation": result.observation.cpu()})
            # Inline serialization keeps metadata independent of child tensor FD lifetime.
            result.transition, result.observation = {}, None
            payload = pickle.dumps(result)
            if control.stop:
                return
            messages.send_bytes(payload)
        # Keep shared tensor storage alive until the consumer drains the final slot.
        while not control.stop:
            time.sleep(opts.idle_sleep_s)
    except BaseException:
        messages.send_bytes(pickle.dumps(RuntimeError(traceback.format_exc())))
        raise
    finally:
        messages.close()


class AsyncCollector:
    """Bounded collector process; the caller retains learner, recorder and replay ownership."""

    def __init__(
        self,
        spec,
        cfg,
        context,
        actor,
        schema,
        *,
        observation_layout,
        resume_step=0,
        commands=None,
        rng=None,
        spawn_cpus=None,
    ):
        self.opts = cfg.trainer.async_options
        self.actor = actor
        bundle = InferenceBundleSpec(
            (ModuleSection("actor", sum(p.numel() for p in actor.parameters())),),
            (NormalizerSection("obs", actor.proprio_dim), NormalizerSection("goal", actor.goal_dim)),
        )
        shared = WeightChannelShared(bundle)
        self.sender = HostWeightSender(shared)
        self.publish()
        schema = TransitionSchema(
            (
                *schema.fields,
                TransitionField("collector_observation", (actor.proprio_dim + actor.goal_dim,), torch.float32),
            )
        )
        self.ring = HostTransitionRing(schema, self.opts.ring_capacity, context.num_envs)
        self.control = Control()
        mp = torch.multiprocessing.get_context("spawn")
        self.messages, sender = mp.Pipe(duplex=False)
        self.process = mp.Process(
            target=_run_collector,
            args=(
                spec,
                cfg,
                context.num_envs,
                context.seed or 0,
                context.sim,
                resume_step,
                commands,
                rng,
                self.ring,
                shared,
                self.sender.params,
                self.control,
                sender,
                context.logging.interval,
                context.checkpoint.interval,
                observation_layout,
            ),
            name="apg-collector",
        )
        with spawn_worker_cpus(spawn_cpus):
            self.process.start()
        sender.close()

    def publish(self):
        self.sender.publish({"actor": self.actor}, normalizers(self.actor))

    def collect(self, **kwargs):
        """Return owned tensors for consumers that retain a collection."""
        with self.borrow(**kwargs) as result:
            result.transition = {key: value.clone() for key, value in result.transition.items()}
            result.observation = result.observation.clone()
            return result

    @contextmanager
    def borrow(self, **_):
        """Lease ring tensor views until all host consumers have finished.

        Replay must copy into its own staging before this context exits.
        Metadata remains owned; tensor views must not escape the lease.
        """
        while not self.messages.poll(0.1):
            if not self.process.is_alive():
                raise RuntimeError(f"APG collector exited with code {self.process.exitcode}")
        try:
            result = pickle.loads(self.messages.recv_bytes())
        except EOFError as exc:
            raise RuntimeError("APG collector closed its metadata pipe unexpectedly") from exc
        if isinstance(result, BaseException):
            raise result
        _, span = self.ring.read_span()
        batch = {key: value[0] for key, value in span.items()}
        result.observation = batch.pop("collector_observation")
        result.transition = batch
        try:
            yield result
        finally:
            self.ring.commit_reads(1)

    def close(self):
        self.control.set_stop()
        self.process.join(timeout=8)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=3)
        self.messages.close()


class UpdateSchedule:
    def __init__(self, agent, *, remainder=0):
        self.warmup = max(agent.learning_starts, agent.history_length, agent.world_model.horizon)
        ratio = Fraction(str(agent.num_updates))
        self.numerator, self.denominator = ratio.numerator, ratio.denominator
        self.remainder = remainder

    def advance(self, local_steps):
        # Replay is empty on resume: refill it without consuming saved update credit.
        if local_steps <= self.warmup:
            return 0
        updates, self.remainder = divmod(self.remainder + self.numerator, self.denominator)
        return updates

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Owned CPU messages between the APG simulator collector and device learner."""

from __future__ import annotations

import os
import pickle
import time
import traceback
from dataclasses import dataclass
from multiprocessing.context import Process
from queue import Empty, Full
from typing import TYPE_CHECKING

import numpy as np
import torch

from motrix_env_core.registry import EnvBuildSpec
from motrix_env_core.torch.geometry import unflatten_state
from motrix_rl_builtin.apg.config import ApgAgentCfg
from motrix_rl_builtin.apg.contracts import build_head, head_transition_arrays
from motrix_rl_builtin.apg.networks import Policy
from motrix_rl_builtin.apg.rollout import select_action, shift_history

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from motrix_rl_builtin.apg.learner import ApgLearner


@dataclass
class UpdateCredit:
    """Strict post-warmup update credit used by the reference trainer."""

    utd: float
    warmup: int
    steps: int = 0
    updates: int = 0

    def receive(self, steps: int) -> None:
        if steps <= 0:
            raise ValueError("Expected positive collector steps")
        self.steps += steps

    @property
    def available(self) -> int:
        earned = int(max(self.steps - self.warmup, 0) * self.utd + 1e-9)
        return max(0, earned - self.updates)

    def consume(self) -> None:
        if not self.available:
            raise RuntimeError("Cannot update ahead of real collector steps")
        self.updates += 1


def pack_owned(value: object) -> bytes:
    """Serialize before Queue.put so its feeder cannot observe later mutation."""
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpack_owned(value: bytes) -> object:
    return pickle.loads(value)


def cpu_tree(value):
    """Copy nested optimizer state without retaining device storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def policy_snapshot(learner: ApgLearner) -> bytes:
    state = {key: value.detach().cpu().clone() for key, value in learner.pi.state_dict().items()}
    return pack_owned({"version": learner.iteration, "norm_steps": learner.sim_steps, "state": state})


def put_latest(queue: Queue, message: bytes) -> int:
    """Drop an unpublished policy snapshot, never a transition block.

    Returns how many unpublished snapshots were dropped to make room (the
    ``train/snapshots_replaced`` metric); 0 when the message went straight in.
    """
    replaced = 0
    while True:
        try:
            queue.put_nowait(message)
            return replaced
        except Full:
            try:
                queue.get(timeout=0.05)
                replaced += 1
            except Empty:
                pass


def check_collector(process: Process, errors) -> None:
    if errors.poll():
        raise RuntimeError("APG collector failed:\n" + errors.recv())
    if process.exitcode is not None:
        raise RuntimeError(f"APG collector exited unexpectedly ({process.exitcode})")


def collect_blocks(
    spec: EnvBuildSpec,
    num_envs: int,
    seed: int | None,
    cfg: ApgAgentCfg,
    head_path: str,
    block_steps: int,
    collector_threads: int,
    policy_device: str,
    initial_policy: bytes,
    initial_sampling: bytes,
    transitions,
    snapshots,
    stop,
    errors,
) -> None:
    """Spawn target: own the CPU simulator and publish complete, ordered blocks."""
    try:
        parent_pid = os.getppid()
        torch.set_num_threads(collector_threads)
        if seed is not None:
            torch.manual_seed(seed + 1)
            np.random.seed(seed + 1)
        env = spec.make(num_envs=num_envs, mode="train", seed=seed)
        head = build_head(head_path, env)
        action_low = torch.from_numpy(np.asarray(head.action_low, dtype=np.float32))
        action_high = torch.from_numpy(np.asarray(head.action_high, dtype=np.float32))
        sampling = unpack_owned(initial_sampling)
        if sampling is not None:
            head.load_sampling_state(sampling)
        policy = Policy(
            head.obs_dim,
            head.goal_dim,
            cfg.hist,
            cfg.z_dim,
            head.num_joints,
            cfg.policy_hidden,
        ).eval()
        snapshot = unpack_owned(initial_policy)
        device = torch.device(policy_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            policy = policy.to(device)
        policy.load_state_dict(snapshot["state"])

        def policy_forward(history, goal):
            if device.type == "cuda":
                return policy(history.to(device), goal.to(device)).cpu()
            return policy(history, goal)

        history = torch.zeros(num_envs, cfg.hist, head.obs_dim)
        previous_action = torch.zeros(num_envs, head.num_joints)
        episode_step = torch.zeros(num_envs, dtype=torch.int64)
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        sequence = 0
        steps = 0
        started = time.monotonic()
        collector_wait_s = 0.0
        # The env's own observation drives collection (noise included), the
        # same source the sync learner consumes through ApgLearner.observe.
        obs, goal = head.split_observation(
            torch.as_tensor(np.asarray(env.init_state().obs.policy), dtype=torch.float32)
        )
        with torch.inference_mode():
            while not stop.is_set() and os.getppid() == parent_pid:
                fresh = None
                try:
                    while True:
                        fresh = unpack_owned(snapshots.get_nowait())
                except Empty:
                    pass
                if fresh is not None:
                    snapshot = fresh
                    policy.load_state_dict(snapshot["state"])
                rows: list[dict[str, np.ndarray]] = []
                obs_rows: list[np.ndarray] = []
                goal_rows: list[np.ndarray] = []
                sample_started = time.monotonic()
                stats = {key: 0.0 for key in ("r", "ep_len_sum", "ep_ret_sum", "term", "done")}
                for _ in range(block_steps):
                    # The buffers mutate on every step, so snapshot them now.
                    state_raw, frames_raw, context_raw = head_transition_arrays(head)
                    state = torch.from_numpy(state_raw.copy())
                    frames = torch.from_numpy(frames_raw.copy())
                    context = torch.from_numpy(context_raw.copy())
                    history = shift_history(history, obs, episode_step == 0)
                    action = select_action(policy_forward, history, goal, action_low, action_high, cfg.action_noise)
                    result = env.step(action.numpy().astype(np.float32))
                    obs, goal = head.split_observation(
                        torch.as_tensor(np.asarray(result.obs.policy), dtype=torch.float32)
                    )
                    done = np.asarray(result.terminated, dtype=bool) | np.asarray(result.truncated, dtype=bool)
                    hard = head.termination_counts(result.metrics)
                    reward = np.asarray(result.reward, dtype=np.float32).copy()
                    rows.append(
                        {
                            "s": state.numpy(),
                            "s1": np.asarray(head.endpoint_state(), dtype=np.float32).copy(),
                            "a": action.numpy(),
                            "a_prev": previous_action.numpy(),
                            "xi": context.numpy(),
                            "r": reward,
                            "f": frames.numpy(),
                            "ep_step": episode_step.to(torch.int32).numpy(),
                            "done": done.copy(),
                        }
                    )
                    obs_rows.append(obs.numpy())
                    goal_rows.append(goal.numpy())
                    stats["r"] += float(reward.mean()) / (head.max_step_reward * head.ctrl_dt)
                    stats["ep_len_sum"] += float(((episode_step.numpy() + 1) * done).sum())
                    episode_returns += reward
                    if done.any():
                        stats["ep_ret_sum"] += float(episode_returns[done].sum())
                        episode_returns[done] = 0.0
                    stats["term"] += float(hard)
                    stats["done"] += float(done.sum())
                    previous_action = action.clone()
                    episode_step = torch.where(torch.from_numpy(done), 0, episode_step + 1)
                block = pack_owned(
                    {
                        "sequence": sequence,
                        "vector_steps": steps + block_steps,
                        "policy_version": snapshot["version"],
                        "norm_steps": snapshot["norm_steps"],
                        "rows": rows,
                        "obs": obs_rows,
                        "goal": goal_rows,
                        "stats": stats,
                        "sampling": head.sampling_state(),
                        "collector_sample_s": time.monotonic() - sample_started,
                        "collector_elapsed_s": time.monotonic() - started,
                        "collector_wait_s": collector_wait_s,
                    }
                )
                steps += block_steps
                queued_at = time.monotonic()
                while not stop.is_set() and os.getppid() == parent_pid:
                    try:
                        transitions.put(block, timeout=0.1)
                        break
                    except Full:
                        pass
                collector_wait_s = time.monotonic() - queued_at
                sequence += 1
    except BaseException:
        errors.send(traceback.format_exc())
    finally:
        errors.close()
        transitions.cancel_join_thread()


@torch.no_grad()
def ingest_block(
    learner: ApgLearner,
    block: dict,
    expected_sequence: int,
    *,
    update_wm_norms: bool = True,
) -> int:
    if block["sequence"] != expected_sequence:
        raise RuntimeError(f"Out-of-order collector block: {block['sequence']} != {expected_sequence}")
    for row, obs, goal in zip(block["rows"], block["obs"], block["goal"], strict=True):
        transition = {key: torch.from_numpy(value).to(learner.device) for key, value in row.items()}
        learner.buffer.add(transition)
        learner.pi.obs_norm.update(torch.from_numpy(obs).to(learner.device))
        learner.pi.goal_norm.update(torch.from_numpy(goal).to(learner.device))
        first = unflatten_state(transition["s"])
        last = unflatten_state(transition["s1"])
        if update_wm_norms:
            learner.wm.in_norm.update(
                learner.wm.inputs(first, transition["a"], learner.head.normalize_context(transition["xi"]))
            )
            learner.wm.out_norm.update(learner.wm.targets(first, last))
        learner.sim_steps += learner.num_envs
    for key, value in block["stats"].items():
        learner.stats[key].append(value / len(block["rows"]) if key == "r" else value)
    if block["sampling"] is not None:
        learner.head.load_sampling_state(block["sampling"])
    return len(block["rows"])


def receive_block(queue: Queue, process: Process, errors, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        check_collector(process, errors)
        try:
            return unpack_owned(queue.get(timeout=0.1))
        except Empty:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No APG collector block for {timeout}s") from None

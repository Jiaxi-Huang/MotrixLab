# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deterministic per-clip evaluation for SuperMimic checkpoints.

Ported from UniSuperMimic ``tasks/.../supermimic/evaluate.py`` (evaluation
core only; render / sim2sim tooling is not ported): one environment lane per
corpus clip, every episode pinned to its clip's first frame, no observation
or exploration noise, running until each lane finishes its clip or fails.
Reports the mean and frame-weighted normalized returns (the per-clip return
over ``(clip length - 1) * max step reward``), the failed-clip list, and the
per-clip breakdown.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .contracts import State  # noqa: F401 - re-exported for typing clarity
from .geometry import unflatten_state
from .mdp import SuperMimicTorchMDP
from .networks import Policy


def clip_bounds(clip_end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip start frames and lengths from the per-frame ``frame_clip_end``."""
    ends = torch.unique(clip_end)
    starts = torch.cat((torch.zeros(1, dtype=torch.int64, device=ends.device), ends[:-1] + 1))
    lengths = ends - starts + 1
    return starts, lengths


def load_policy(mdp: SuperMimicTorchMDP, checkpoint: dict[str, Any], device: torch.device) -> Policy:
    """Rebuild the goal-conditioned policy from a learner checkpoint."""
    args = checkpoint["args"]
    goal_frames = int(args["goal_frames"])
    policy = Policy(
        mdp.obs_dim,
        mdp.goal_frame_dim * goal_frames,
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        num_joints=mdp.num_joints,
        hidden=tuple(args["policy_hidden"]),
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


@torch.inference_mode()
def evaluate(
    env,
    policy: Policy,
    mdp: SuperMimicTorchMDP,
    *,
    goal_frames: int,
    hist: int,
    device: torch.device,
    clip_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate every motion once from its first frame without noise.

    Requires one environment lane per corpus clip. Each lane's return
    accumulates until it finishes its clip (termination on the clip's final
    frame — a success) or fails a height/tilt check; the report normalizes
    per-clip returns by the clip's attainable maximum.
    """
    motion = env.command_terms["motion"]
    num_clips = int(torch.unique(mdp.clip_end).numel())
    if env.num_envs != num_clips:
        raise ValueError(
            f"SuperMimic evaluation requires one environment per clip: "
            f"got {env.num_envs} environments for {num_clips} clips"
        )

    starts, _ = clip_bounds(mdp.clip_end)
    env.init_state()
    motion.requested_frames[:, 0] = starts.cpu().numpy()
    env.reset(np.arange(env.num_envs))
    frames = torch.as_tensor(np.asarray(motion.steps[:, 0]), dtype=torch.long, device=device)
    # With reset noise disabled the teleport lands exactly on the reference
    # frame, so the first observation comes from the corpus reference state
    # (the state buffer refreshes on the next evaluate pass).
    state = mdp.flatten_state(mdp.ref_state(frames))
    previous_action = torch.zeros((env.num_envs, mdp.num_joints), device=device)

    returns = torch.zeros(num_clips, device=device)
    steps = torch.zeros(num_clips, dtype=torch.int64, device=device)
    finished = torch.zeros(num_clips, dtype=torch.bool, device=device)
    failed = torch.zeros_like(finished)
    history: torch.Tensor | None = None

    while not bool(finished.all()):
        obs = mdp.obs(unflatten_state(state), previous_action, frames)
        if history is None:
            history = obs.unsqueeze(1).repeat(1, hist, 1)
        else:
            history = torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        goal = mdp.goal(frames, goal_frames)
        action = policy(history, goal)
        result = env.step(action.detach().cpu().numpy().astype(np.float32))

        reward = torch.as_tensor(np.asarray(result.reward), device=device)
        terminated = torch.as_tensor(np.asarray(result.terminated), dtype=torch.bool, device=device)
        truncated = torch.as_tensor(np.asarray(result.truncated), dtype=torch.bool, device=device)
        clip_end_done = torch.as_tensor(np.asarray(result.metrics["motion_clip_end"]), dtype=torch.bool, device=device)
        alive = ~finished
        returns += torch.where(alive, reward, torch.zeros_like(reward))
        steps += alive.to(torch.int64)
        failed |= alive & terminated & ~clip_end_done
        finished |= alive & (terminated | truncated)

        state = torch.as_tensor(np.asarray(motion.state), device=device)
        frames = torch.as_tensor(np.asarray(motion.steps[:, 0]), dtype=torch.long, device=device)
        previous_action = action

    starts_cpu, lengths = clip_bounds(mdp.clip_end)
    max_returns = (lengths - 1).float() * (mdp.max_step_reward * mdp.ctrl_dt)
    normalized = returns / max_returns
    failed_indices = failed.nonzero(as_tuple=False).flatten().tolist()
    per_clip = {
        index: {
            "start_frame": int(starts_cpu[index]),
            "length": int(lengths[index]),
            "norm_return": normalized[index].item(),
            "steps": int(steps[index]),
            "failed": bool(failed[index]),
        }
        for index in range(num_clips)
    }
    if clip_names is not None:
        if len(clip_names) != num_clips:
            raise ValueError(f"Expected {num_clips} clip names, got {len(clip_names)}")
        for name, entry in zip(clip_names, per_clip.values(), strict=True):
            entry["clip"] = name
    return {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "norm_return": normalized.mean().item(),
        "norm_return_framew": (returns.sum() / max_returns.sum()).item(),
        "n_failed": len(failed_indices),
        "failed": [clip_names[index] for index in failed_indices] if clip_names is not None else None,
        "failed_clip_indices": failed_indices,
        "per_clip": per_clip,
    }


__all__ = ["clip_bounds", "evaluate", "load_policy"]

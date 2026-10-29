# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared collection-side rollout mechanics for the APG paths.

The sync learner, the async CPU collector, and both play loops select actions
through the same pieces: an observation-history shift that resets on fresh
episodes, and an exploration action clamped to the head's bounds. Imagination
deliberately does NOT use these — it preserves the autograd graph, consumes
analytic observations, and runs the raw policy output without noise or
clamping (see ``imagined_env.py``).
"""

from __future__ import annotations

from collections.abc import Callable

import torch


@torch.no_grad()
def shift_history(obs_hist: torch.Tensor, obs: torch.Tensor, fresh: torch.Tensor | None = None) -> torch.Tensor:
    """Shift ``obs_hist`` one step left, appending ``obs``; reset fresh episodes.

    ``fresh`` marks new episodes (``ep_step == 0``); their history is filled
    with the current observation instead of the previous episode's tail.
    """
    shifted = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
    if fresh is None:
        return shifted
    return torch.where(fresh[:, None, None], obs.unsqueeze(1).expand_as(obs_hist), shifted)


@torch.no_grad()
def select_action(
    pi: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    obs_hist: torch.Tensor,
    goal: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    action_noise: float,
) -> torch.Tensor:
    """Exploration action: policy output plus Gaussian noise, clamped to bounds.

    The caller owns any device placement around ``pi`` (the async collector
    rounds a CUDA policy through the CPU); the noise and clamp stay with the
    returned action's device.
    """
    action = pi(obs_hist, goal)
    return torch.clamp(action + action_noise * torch.randn_like(action), action_low, action_high)


__all__ = ["select_action", "shift_history"]

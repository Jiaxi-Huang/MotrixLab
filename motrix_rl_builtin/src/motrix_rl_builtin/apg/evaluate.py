# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Policy rebuilding from APG learner checkpoints.

Task-side evaluation harnesses (e.g. deterministic per-clip rollouts over a
reference corpus) live in the task package next to their parsing head; this
module only owns the generic piece — rebuilding the goal-conditioned policy
from a checkpoint using the task head's dimensions.
"""

from __future__ import annotations

from typing import Any

import torch

from .networks import Policy


def load_policy(head, checkpoint: dict[str, Any], device: torch.device) -> Policy:
    """Rebuild the goal-conditioned policy from a learner checkpoint."""
    args = checkpoint["args"]
    policy = Policy(
        head.obs_dim,
        head.goal_dim,
        hist=int(args["hist"]),
        z_dim=int(args["z_dim"]),
        num_joints=head.num_joints,
        hidden=tuple(args["policy_hidden"]),
    ).to(device)
    policy.load_state_dict(checkpoint["pi"])
    policy.eval()
    return policy


__all__ = ["load_policy"]

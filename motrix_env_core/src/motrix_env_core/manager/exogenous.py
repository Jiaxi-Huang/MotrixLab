# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Exogenous-channel rules for imagination-side state advance.

An exogenous channel is one per-environment command-state leaf whose value is
not predicted by the world model: it advances by rule. Rules are declared on
the command config (``CommandCfg.exogenous``, keyed by the leaf's dotted
path) and consumed by the derived modeling program — see
[Manager 双路径建模设计](../../../wiki/design/manager/modeling.md).

One flat config class with a ``kind`` discriminator (rather than a union of
rule classes) keeps the field schema-driven-config compatible: env cfg trees
convert to structured configs, which cannot carry union value types.
"""

from enum import Enum

from motrix_env_core.config import configclass


class ExogenousKind(Enum):
    """How imagination advances one exogenous channel."""

    INCREMENT = "increment"
    COPY_ACTION = "copy_action"
    CONSTANT = "constant"


@configclass(kw_only=True)
class ExogenousRule:
    """Advance rule for one exogenous channel.

    - ``increment``: the channel steps by ``step`` every imagination step
      (motion phase); ``step`` applies only to this kind. With ``clamp``
      naming a sibling channel, the increment holds at that channel's value
      (upstream supermimic's clip-end clamp-and-continue: the motion frame
      index freezes at the clip's last frame instead of bleeding into the
      next clip).
    - ``copy_action``: the channel receives ``action_term``'s latest raw
      policy action (the term's width must equal the channel's trailing
      width); ``action_term`` applies only to this kind.
    - ``constant``: the channel holds its episode-start sampled value;
      terminal values reach the learner through the FinalState capture.

    ``clamp`` stores the sibling channel's leaf path (the same keying as the
    ``exogenous`` dict itself) and applies only to ``increment`` rules.
    """

    kind: ExogenousKind = ExogenousKind.CONSTANT
    step: float = 1.0
    action_term: str = ""
    clamp: str = ""


__all__ = ["ExogenousKind", "ExogenousRule"]

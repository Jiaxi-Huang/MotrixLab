# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 supermimic tracking task — the whole-body tracking family's APG variant.

Same family math as the dance/largebox/backflip variants (shared WBT command,
rewards, and motion infrastructure; the frame index's exogenous annotation
comes with the family). The APG-specific deltas are pure configuration:

- contact forces are not part of the modeling state, so the non-modelable
  contact query and its reward term are dropped here;
- reaching the clip's last frame truncates (time-limit semantics) instead of
  the real advance's clip-end branches — in imagination the truncation gate
  bounds the ``steps`` increment channel.

zh_CN: G1 supermimic 全身跟踪任务——WBT 家族的 APG 变体：去掉不可想象的
接触项，clip 结束映射为截断，其余家族数学零改动。
"""

from dataclasses import fields

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv

from ..mdp.terminations import MotionClipEndTruncationCfg
from .common import MOTION_DIR, G1WbtEnvCfg


@registry.envcfg("g1-wbt-supermimic")
def make_g1_wbt_supermimic_cfg() -> EnvCfg:
    """Track the G1 supermimic motion corpus with the APG-trainable variant."""
    cfg = G1WbtEnvCfg()
    # The corpus directory holds the supermimic clips (schema v1 npz, one or
    # more); the dance corpus is the runnable stand-in until the target
    # corpus lands in assets.
    cfg.commands.motion.motion_files = (str(MOTION_DIR / "dance"),)
    # Uniform start frames first; the curriculum switches to
    # ``sampling_mode: return`` from the training config.
    cfg.commands.motion.sampling_mode = "uniform"
    # Contact sensing is not modelable: drop the query and its reward so the
    # declared read set stays inside the modeling state.
    del cfg.queries.data["undesired_contact_forces"]
    rewards = {field.name: getattr(cfg.rewards, field.name) for field in fields(cfg.rewards)}
    del rewards["undesired_contacts"]
    cfg.rewards = rewards
    # Whole-clip episodes: the frame index reaching the clip's last frame is
    # a time-limit truncation, not a failure.
    cfg.truncations["motion_clip_end"] = MotionClipEndTruncationCfg()
    return cfg


registry.env("g1-wbt-supermimic")(ManagerEnv)

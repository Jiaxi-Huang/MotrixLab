# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 supermimic tracking task — the whole-body tracking family's APG variant.

Same family math as the dance/largebox/backflip variants (shared WBT command,
rewards, and motion infrastructure; the frame index's exogenous annotation
comes with the family). The APG-specific deltas are pure configuration, and
task-side values follow the upstream supermimic configuration exactly (see
[supermimic-vs-motrixlab-comparison](../../../../../../wiki/research/supermimic-vs-motrixlab-comparison.md)):

- the tracking anchor is the pelvis (upstream re-anchors from the mjlab
  default torso) and terminations are the upstream pair — reference z error
  0.25 and reference orientation 0.8 — plus the family's joint-level guards;
- reward weights follow upstream: reference position 0.5, relative body
  position 1.0, action rate −0.1, joint limits −2.0 (σ and the surviving
  weights already match);
- the policy observation is the upstream goal window — ten future joint
  target frames at stride five (~0.9 s lookahead) plus the anchor-relative
  reference orientation window and projected gravity — instead of the
  family's current-frame terms; the value group adds the anchor-relative
  reference position window;
- contact forces are not part of the modeling state, so the non-modelable
  contact query and its reward term are dropped here;
- reaching the clip's last frame truncates (time-limit semantics, kept over
  upstream's clamp-and-continue) — in imagination the truncation gate bounds
  the ``steps`` increment channel.

zh_CN: G1 supermimic 全身跟踪任务——WBT 家族的 APG 变体，任务侧数值对齐
上游 supermimic（锚点/奖励权重/终止阈值/目标窗口观测），去掉不可想象的
接触项，clip 结束映射为截断。
"""

from dataclasses import fields

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyLinearVelocityObsCfg,
    UniformNoiseCfg,
)

from ..mdp.observations import (
    DofPosRelObsCfg,
    DofVelObsCfg,
    MotionFutureJointObsCfg,
    MotionFutureReferenceOrientationObsCfg,
    MotionFutureReferencePositionObsCfg,
    ProjectedGravityObsCfg,
    RobotBodyOrientationObsCfg,
    RobotBodyPositionInReferenceFrameObsCfg,
    SoftTerminationObsCfg,
)
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
    # Upstream anchors tracking on the pelvis, not the mjlab default torso.
    cfg.commands.motion.reference_body_name = "pelvis"
    # The upstream default curriculum: 0.5 uniform + 0.4 low-return focus +
    # 0.1 keep (mixture ratios are the command cfg defaults).
    cfg.commands.motion.sampling_mode = "return"
    # Contact sensing is not modelable: drop the query and its reward so the
    # declared read set stays inside the modeling state.
    del cfg.queries.data["undesired_contact_forces"]
    rewards = {field.name: getattr(cfg.rewards, field.name) for field in fields(cfg.rewards)}
    del rewards["undesired_contacts"]
    # Upstream reward weights (σ and the surviving weights already match).
    rewards["motion_global_ref_position_error_exp"].weight = 0.5
    rewards["motion_relative_body_position_error_exp"].weight = 1.0
    rewards["action_rate_l2"].weight = -0.1
    rewards["limits_dof_pos"].weight = -2.0
    cfg.rewards = rewards
    # Upstream terminations: reference z error on the (pelvis) anchor at 0.25
    # and reference orientation at 0.8. The end-effector z termination is
    # upstream-deleted; the family's joint-level guards stay as simulation
    # blow-up protection.
    terminations = {
        "bad_ref_z": type(cfg.terminations.bad_ref_z)(threshold=0.25),
        "bad_ref_ori": cfg.terminations.bad_ref_ori,
        "bad_dof_pos": cfg.terminations.bad_dof_pos,
        "bad_dof_vel": cfg.terminations.bad_dof_vel,
    }
    cfg.terminations = terminations
    # Upstream observation split: the history stack covers proprioception
    # only; the goal window (future joint targets + anchor-relative reference
    # orientation) rides in the value group and the trainer feeds it to the
    # actor as a single flat current frame (apg.goal_terms names the slices).
    # Proprio noise half-widths match upstream sensor noise.
    cfg.observations = {
        "policy": {
            "projected_gravity": ProjectedGravityObsCfg(),
            "base_ang_vel": BodyAngularVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.2)),
            "dof_pos": DofPosRelObsCfg(noise=UniformNoiseCfg(amplitude=0.01)),
            "dof_vel": DofVelObsCfg(noise=UniformNoiseCfg(amplitude=0.5)),
            "actions": ActionsObsCfg(),
        },
        "value": {
            "motion_future_joint": MotionFutureJointObsCfg(),
            "motion_future_ref_pos_b": MotionFutureReferencePositionObsCfg(),
            "motion_future_ref_ori_b": MotionFutureReferenceOrientationObsCfg(),
            "soft_termination": SoftTerminationObsCfg(),
            "robot_body_pos_b": RobotBodyPositionInReferenceFrameObsCfg(),
            "robot_body_ori_b": RobotBodyOrientationObsCfg(),
            "base_lin_vel": BodyLinearVelocityObsCfg(),
            "base_ang_vel": BodyAngularVelocityObsCfg(),
            "dof_pos": DofPosRelObsCfg(),
            "dof_vel": DofVelObsCfg(),
            "actions": ActionsObsCfg(),
        },
    }
    # Whole-clip episodes: the frame index reaching the clip's last frame is
    # a time-limit truncation, not a failure.
    cfg.truncations["motion_clip_end"] = MotionClipEndTruncationCfg()
    return cfg


registry.env("g1-wbt-supermimic")(ManagerEnv)

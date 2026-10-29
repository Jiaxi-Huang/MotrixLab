# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager configuration for the SuperMimic G1 whole-body tracking task."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from omegaconf import MISSING

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerCommandsCfg,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerResetCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    ObsTerm,
    SimQueriesCfg,
)
from motrix_env_core.mdp.action import JointPositionActionCfg, JointPositionActionTerm
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyProjectedGravityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import (
    ActuatorKpQuery,
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyJointPositionLimitsQuery,
    GeomFrictionQuery,
    JointPositionQuery,
    JointVelocityQuery,
    LinkCenterOfMassQuery,
)
from motrix_envs.locomotion.supermimic.mdp import (
    MotionClipEndTerminationCfg,
    SuperMimicDofPosResetCfg,
    SuperMimicGoalObsCfg,
    SuperMimicMotionCommandCfg,
    SuperMimicTrackingErrorObsCfg,
    supermimic_dof_pos_rel_obs,
)
from motrix_envs.locomotion.supermimic.randomization import SuperMimicDynamicsResetCfg
from motrix_envs.locomotion.wbt.mdp.observations import (
    DofPosRelObsCfg,
    DofVelObsCfg,
    MotionJointObsCfg,
    MotionReferenceOrientationObsCfg,
    MotionReferencePositionObsCfg,
    RelativePositionParams,
)
from motrix_envs.locomotion.wbt.mdp.reset import (
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadRefOrientationTerminationCfg,
    BadRefZTerminationCfg,
)


class SuperMimicJointPositionActionTerm(JointPositionActionTerm):
    """Add upstream's per-reset joint offset to the position target."""

    motion = None

    def bind_motion(self, motion) -> None:
        self.motion = motion

    def _process(self, actions: np.ndarray) -> np.ndarray | None:
        target = super()._process(actions)
        if self.motion is not None:
            target = target + np.asarray(self.motion.context)[:, 1:30]
        return target


@configclass(kw_only=True)
class SuperMimicJointPositionActionCfg(JointPositionActionCfg):
    def __call__(self, env, actuators):
        base = super().__call__(env, actuators)
        return SuperMimicJointPositionActionTerm(base.action_space, base.state)


@configclass
class SuperMimicDofPosRelObsCfg(DofPosRelObsCfg):
    """Subtract the reset joint offset from the policy joint-position input."""

    def __call__(self, ctx) -> ObsTerm:
        queried_joints = ctx.model.bodies["robot"].joint_names
        robot = ctx.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}.")
        positions = dict(
            zip(
                (robot.resolve_name(name) for name in robot.key_pose.joint_names),
                robot.key_pose.poses[self.reference_key_pose],
                strict=True,
            )
        )
        reference = np.asarray([positions[name] for name in queried_joints], dtype=np.float32)
        return ObsTerm(
            reference.shape[0],
            supermimic_dof_pos_rel_obs,
            RelativePositionParams(reference, np.float32(self.noise.amplitude)),
        )


@configclass
class SuperMimicActionsCfg(ManagerActionsCfg):
    # default_q + joint_offset + 0.25 * effort/kp * action.
    joint_position: SuperMimicJointPositionActionCfg = SuperMimicJointPositionActionCfg(
        action_scales_by_effort_limit_over_p_gain=True
    )


@configclass
class SuperMimicCommandsCfg(ManagerCommandsCfg):
    motion: SuperMimicMotionCommandCfg = SuperMimicMotionCommandCfg()


@configclass
class SuperMimicPolicyObsCfg(ManagerObservationGroupCfg):
    """Goal-conditioned policy observation.

    The proprioceptive block matches upstream's 99-D layout for the 29-DoF
    G1 — joint positions relative to the default pose, joint velocities,
    local angular velocity, gravity, previous action, and the relative
    rotation to the current reference frame — each with upstream's noise
    amplitudes, followed by the exact future-reference goal term.
    """

    dof_pos: SuperMimicDofPosRelObsCfg = SuperMimicDofPosRelObsCfg(noise=UniformNoiseCfg(amplitude=0.01))
    dof_vel: DofVelObsCfg = DofVelObsCfg(noise=UniformNoiseCfg(amplitude=0.5))
    base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.2))
    gravity: BodyProjectedGravityObsCfg = BodyProjectedGravityObsCfg(noise=UniformNoiseCfg(amplitude=0.05))
    actions: ActionsObsCfg = ActionsObsCfg()
    motion_ref_ori_b: MotionReferenceOrientationObsCfg = MotionReferenceOrientationObsCfg(
        noise=UniformNoiseCfg(amplitude=0.05)
    )
    goal: SuperMimicGoalObsCfg = SuperMimicGoalObsCfg()


@configclass
class SuperMimicValueObsCfg(ManagerObservationGroupCfg):
    """Privileged value observation.

    The policy terms without noise, plus the current-frame reference joints,
    the reference root position error, and the anchored tracked-body
    position errors (the simulation context is supplied separately to the
    critic and world model).
    """

    dof_pos: SuperMimicDofPosRelObsCfg = SuperMimicDofPosRelObsCfg()
    dof_vel: DofVelObsCfg = DofVelObsCfg()
    base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg()
    gravity: BodyProjectedGravityObsCfg = BodyProjectedGravityObsCfg()
    actions: ActionsObsCfg = ActionsObsCfg()
    motion_ref_ori_b: MotionReferenceOrientationObsCfg = MotionReferenceOrientationObsCfg()
    goal: SuperMimicGoalObsCfg = SuperMimicGoalObsCfg()
    motion_joint: MotionJointObsCfg = MotionJointObsCfg()
    motion_ref_pos_b: MotionReferencePositionObsCfg = MotionReferencePositionObsCfg()
    tracking_error: SuperMimicTrackingErrorObsCfg = SuperMimicTrackingErrorObsCfg()


@configclass
class SuperMimicObservationsCfg(ManagerObservationsCfg):
    policy: SuperMimicPolicyObsCfg = SuperMimicPolicyObsCfg()
    value: SuperMimicValueObsCfg = SuperMimicValueObsCfg()


# SuperMimic's tracking recipe (upstream MDP.RewardCfg): four unit-weight
# body kernels, two half-weighted root kernels, action-rate -0.1 and the
# joint-limit penalty -2.0 over the 0.9 soft range. The manager applies the
# ctrl_dt factor, so the per-step reward scale matches upstream's `* dt`.
@configclass
class SuperMimicRewardsCfg(ManagerRewardsCfg):
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=0.5,
        sigma=0.3,
    )
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=0.5,
        sigma=0.4,
    )
    motion_relative_body_position_error_exp: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(
        weight=1.0,
        sigma=0.3,
    )
    motion_relative_body_orientation_error_exp: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )
    motion_global_body_lin_vel: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(
        weight=1.0,
        sigma=1.0,
    )
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=3.14,
    )
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.1)
    limits_dof_pos: DofLimitRewardCfg = DofLimitRewardCfg(weight=-2.0, soft_limit=0.9, cap=float("inf"))


# Upstream terminates on the reference height (0.25 m) and tilt (0.8) checks
# only; the dof guards are MotrixLab-side NaN/divergence safety nets that
# never fire on healthy rollouts.
@configclass
class SuperMimicTerminationsCfg(ManagerTerminationsCfg):
    bad_ref_z: BadRefZTerminationCfg = BadRefZTerminationCfg(threshold=0.25)
    bad_ref_ori: BadRefOrientationTerminationCfg = BadRefOrientationTerminationCfg(threshold=0.8)
    motion_clip_end: MotionClipEndTerminationCfg = MotionClipEndTerminationCfg()


@configclass
class SuperMimicResetCfg(ManagerResetCfg):
    # Teleport-to-motion reset with SuperMimic's noise magnitudes: the WBT
    # defaults are exactly upstream's (pos 0.05/0.05/0.01, euler 0.1/0.1/0.2,
    # lin vel 0.5/0.5/0.2, ang vel 0.52/0.52/0.78, dof 0.01 rad), plus
    # upstream-style robot friction and torso CoM randomization. All samples
    # ride the 33-D simulation context used by the world model and critic.
    body_pos: BodyPosResetCfg = BodyPosResetCfg()
    body_rot: BodyRotResetCfg = BodyRotResetCfg()
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg()
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg()
    body_dof_pos: SuperMimicDofPosResetCfg = SuperMimicDofPosResetCfg()
    dynamics: SuperMimicDynamicsResetCfg = SuperMimicDynamicsResetCfg()


@configclass(kw_only=True)
class SuperMimicEnvCfg(ManagerBasedEnvCfg):
    """Goal-conditioned whole-body tracking of a LAFAN1 motion corpus.

    The episode timeline follows SuperMimic: episodes start from bins drawn
    from the return-based mixture, end on the owning clip's final frame or
    the height/tilt checks, and are capped at 10 s (500 control steps).
    """

    # Keep the simulator cadence aligned with UniSuperMimic's 50 Hz control
    # setup; solver details are backend-owned rather than task hyperparameters.
    sim: SimCfg = SimCfg(dt=0.005)
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 10.0
    queries: SimQueriesCfg = SimQueriesCfg(
        model={
            "actuator_kp": ActuatorKpQuery(),
            "robot_joint_position_limits": BodyJointPositionLimitsQuery(body=MISSING),
        },
    )
    commands: SuperMimicCommandsCfg = SuperMimicCommandsCfg()
    actions: SuperMimicActionsCfg = SuperMimicActionsCfg()
    observations: SuperMimicObservationsCfg = SuperMimicObservationsCfg()
    rewards: SuperMimicRewardsCfg = SuperMimicRewardsCfg()
    terminations: SuperMimicTerminationsCfg = SuperMimicTerminationsCfg()
    sim_reset: ManagerResetCfg = SuperMimicResetCfg()

    def __post_init__(self) -> None:
        robot = self.scene.objs.robot
        joint_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        tracked_body_names = tuple(robot.resolve_name(name) for name in self.commands.motion.tracked_body_names)
        self.commands.motion.joint_names = joint_names
        self.commands.motion.tracked_body_names = tracked_body_names
        self.actions.joint_position.actuator_names = joint_names
        self.queries.data.update(
            {
                "robot_dof_pos": JointPositionQuery(joints=joint_names),
                "robot_dof_vel": JointVelocityQuery(joints=joint_names),
                "tracked_body_pos": BatchLinkPositionQuery(links=tracked_body_names),
                "tracked_body_quat": BatchLinkQuaternionQuery(links=tracked_body_names),
                "tracked_body_linear_velocity": BatchLinkLinearVelocityQuery(links=tracked_body_names),
                "tracked_body_angular_velocity": BatchLinkAngularVelocityQuery(links=tracked_body_names),
            }
        )
        self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(
            body=robot.resolved_base_link_name
        )
        dynamics = self.sim_reset.dynamics
        self.queries.model.update(
            {
                "supermimic_friction_default": GeomFrictionQuery(name=dynamics.ground_geom),
                "supermimic_torso_com": LinkCenterOfMassQuery(name=robot.resolve_name(dynamics.com_link)),
            }
        )

    def for_play(self) -> SuperMimicEnvCfg:
        cfg = deepcopy(self)
        cfg.max_episode_seconds = None
        # Sequential corpus playback: every lane tracks the timeline frame by
        # frame, crossing clip boundaries without episode resets.
        cfg.commands.motion.supermimic_sampling = False
        cfg.commands.motion.sequential_clips = True
        cfg.commands.motion.hold_at_clip_end = False
        cfg.sim_reset.body_pos.noise_scale = 0.0
        cfg.sim_reset.body_rot.noise_scale = 0.0
        cfg.sim_reset.body_lin_vel.noise_scale = 0.0
        cfg.sim_reset.body_rot_vel.noise_scale = 0.0
        cfg.sim_reset.body_dof_pos.noise_scale = 0.0
        cfg.sim_reset.body_dof_pos.eval_mode = True
        # Deterministic dynamics: the eval scale rewrites the nominal
        # friction and a zero CoM offset on every reset.
        cfg.sim_reset.dynamics.eval_mode = True
        cfg.sim_reset.dynamics.com_offset_noise = (0.0, 0.0, 0.0)
        return cfg

    def for_eval(self) -> SuperMimicEnvCfg:
        """Deterministic per-clip evaluation variant (one lane per clip).

        Every lane keeps its own clip: no start sampling, no reset noise, no
        physical-domain randomization, no episode time limit — episodes end
        on the owning clip's final frame or a tracking failure. The evaluator
        pins each lane to its clip's first frame through the command's
        ``requested_frames`` buffer before the reset.
        """
        cfg = deepcopy(self)
        cfg.max_episode_seconds = None
        cfg.commands.motion.supermimic_sampling = False
        cfg.commands.motion.sequential_clips = False
        cfg.commands.motion.hold_at_clip_end = True
        cfg.sim_reset.body_pos.noise_scale = 0.0
        cfg.sim_reset.body_rot.noise_scale = 0.0
        cfg.sim_reset.body_lin_vel.noise_scale = 0.0
        cfg.sim_reset.body_rot_vel.noise_scale = 0.0
        cfg.sim_reset.body_dof_pos.noise_scale = 0.0
        cfg.sim_reset.body_dof_pos.eval_mode = True
        cfg.sim_reset.dynamics.eval_mode = True
        cfg.sim_reset.dynamics.com_offset_noise = (0.0, 0.0, 0.0)
        return cfg

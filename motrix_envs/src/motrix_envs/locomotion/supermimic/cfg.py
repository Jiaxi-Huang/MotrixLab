# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager configuration for the SuperMimic G1 whole-body tracking task."""

from __future__ import annotations

from copy import deepcopy

from omegaconf import MISSING

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerCommandsCfg,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerResetCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    SimQueriesCfg,
)
from motrix_env_core.mdp.action import JointPositionActionCfg
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyProjectedGravityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.mdp.terminations import BadDofVelocityTerminationCfg
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
    SuperMimicGoalObsCfg,
    SuperMimicMotionCommandCfg,
    SuperMimicTrackingErrorObsCfg,
)
from motrix_envs.locomotion.supermimic.randomization import ROBOT_COLLISION_GEOMS, SuperMimicDynamicsResetCfg
from motrix_envs.locomotion.wbt.mdp.observations import (
    DofPosRelObsCfg,
    DofVelObsCfg,
    MotionJointObsCfg,
    MotionReferenceOrientationObsCfg,
    MotionReferencePositionObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.reset import (
    BodyDofPosResetCfg,
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
    BadDofPositionTerminationCfg,
    BadRefOrientationTerminationCfg,
    BadRefZTerminationCfg,
)


@configclass
class SuperMimicActionsCfg(ManagerActionsCfg):
    # default_q + 0.25 * effort/kp * action — SuperMimic's action structure.
    joint_position: JointPositionActionCfg = JointPositionActionCfg(action_scales_by_effort_limit_over_p_gain=True)


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

    dof_pos: DofPosRelObsCfg = DofPosRelObsCfg(noise=UniformNoiseCfg(amplitude=0.01))
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
    position errors (upstream's critic feature set minus the simulation
    context, which this task does not randomize).
    """

    dof_pos: DofPosRelObsCfg = DofPosRelObsCfg()
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
    bad_dof_pos: BadDofPositionTerminationCfg = BadDofPositionTerminationCfg(threshold=0.5)
    bad_dof_vel: BadDofVelocityTerminationCfg = BadDofVelocityTerminationCfg(threshold=100.0)


@configclass
class SuperMimicResetCfg(ManagerResetCfg):
    # Teleport-to-motion reset with SuperMimic's noise magnitudes: the WBT
    # defaults are exactly upstream's (pos 0.05/0.05/0.01, euler 0.1/0.1/0.2,
    # lin vel 0.5/0.5/0.2, ang vel 0.52/0.52/0.78, dof 0.1 rad), plus the
    # MotrixLab dynamics randomization (ground-friction scale, torso CoM
    # offset) whose samples also ride the simulation context for the world
    # model.
    body_pos: BodyPosResetCfg = BodyPosResetCfg()
    body_rot: BodyRotResetCfg = BodyRotResetCfg()
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg()
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg()
    body_dof_pos: BodyDofPosResetCfg = BodyDofPosResetCfg()
    dynamics: SuperMimicDynamicsResetCfg = SuperMimicDynamicsResetCfg()


@configclass(kw_only=True)
class SuperMimicEnvCfg(ManagerBasedEnvCfg):
    """Goal-conditioned whole-body tracking of a LAFAN1 motion corpus.

    The episode timeline follows SuperMimic: episodes start from bins drawn
    from the return-based mixture, end on the owning clip's final frame or
    the height/tilt checks, and are capped at 10 s (500 control steps).
    """

    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=3)
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
                "supermimic_robot_friction_default": GeomFrictionQuery(name=ROBOT_COLLISION_GEOMS[0]),
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
        cfg.sim_reset.dynamics.eval_mode = True
        cfg.sim_reset.dynamics.com_offset_noise = (0.0, 0.0, 0.0)
        return cfg

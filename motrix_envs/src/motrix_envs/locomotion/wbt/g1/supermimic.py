# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 SuperMimic GenCoord motion recipe, independent of legacy WBT defaults."""

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.manager.exogenous import ExogenousKind, ExogenousRule
from motrix_env_core.manager.observations import ManagerObservationGroupCfg, ManagerObservationsCfg
from motrix_env_core.manager.rewards import ManagerRewardsCfg
from motrix_env_core.manager.terminations import ManagerTerminationsCfg
from motrix_env_core.manager.truncations import ManagerTruncationsCfg
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim.model import KinematicTreeQuery
from motrix_env_core.sim.read import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_envs.locomotion.wbt.cfg import CommandsCfg
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.supermimic import (
    ClipEndCfg,
    HeightTerminationCfg,
    MotionCommandCfg,
    MotionGoalCfg,
    ProprioCfg,
    TiltTerminationCfg,
)


@configclass
class RewardsCfg(ManagerRewardsCfg):
    root_position: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(weight=0.5, sigma=0.3)
    root_orientation: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(weight=0.5, sigma=0.4)
    body_position: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(weight=1.0, sigma=0.3)
    body_orientation: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(weight=1.0, sigma=0.4)
    linear_velocity: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(weight=1.0, sigma=1.0)
    angular_velocity: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(weight=1.0, sigma=3.14)
    action_rate: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.1)
    joint_limit: DofLimitRewardCfg = DofLimitRewardCfg(weight=-2.0, soft_limit=0.9, cap=float("inf"))


@configclass
class TerminationsCfg(ManagerTerminationsCfg):
    height: HeightTerminationCfg = HeightTerminationCfg(threshold=0.25)
    tilt: TiltTerminationCfg = TiltTerminationCfg(threshold=0.8)


@configclass
class TruncationsCfg(ManagerTruncationsCfg):
    clip_end: ClipEndCfg = ClipEndCfg()


@configclass
class PolicyCfg(ManagerObservationGroupCfg):
    proprio: ProprioCfg = ProprioCfg()
    goal: MotionGoalCfg = MotionGoalCfg()


@configclass
class ValueCfg(ManagerObservationGroupCfg):
    goal: MotionGoalCfg = MotionGoalCfg(privileged=True)


@configclass
class ObservationsCfg(ManagerObservationsCfg):
    policy: PolicyCfg = PolicyCfg()
    value: ValueCfg = ValueCfg()


@registry.envcfg("g1-wbt-supermimic")
@configclass(kw_only=True)
class G1SuperMimicWbtEnvCfg(G1WbtEnvCfg):
    commands: CommandsCfg = CommandsCfg(
        motion=MotionCommandCfg(
            motion_files=("~/.cache/motrixlab/lafan1",),
            sampling_mode="return",
            exogenous={
                "steps": ExogenousRule(kind=ExogenousKind.INCREMENT, upper_bound="frame_end"),
                "frame_end": ExogenousRule(kind=ExogenousKind.CONSTANT),
            },
        )
    )
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    truncations: TruncationsCfg = TruncationsCfg()

    def __post_init__(self, motion_files=None) -> None:
        super().__post_init__(motion_files)
        root = self.scene.objs.robot.resolved_base_link_name
        self.commands.motion.reference_body_name = root
        self.queries.data.pop("undesired_contact_forces")
        self.queries.data.update(
            {
                "root_pos": LinkPositionQuery(link=root),
                "root_quat": LinkQuaternionQuery(link=root),
                "root_lin_vel": LinkLinearVelocityQuery(link=root),
                "root_ang_vel": LinkAngularVelocityQuery(link=root),
            }
        )
        self.queries.model["kinematic_tree"] = KinematicTreeQuery(body=root)

    def for_play(self):
        cfg = super().for_play()
        cfg.commands.motion.sequential_clips = False
        cfg.observations.policy.proprio.noisy = False
        return cfg


registry.env("g1-wbt-supermimic")(ManagerEnv)

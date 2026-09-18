# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/bounce_ball_ctrl.xml"


@registry.envcfg("bounce_ball")
@configclass
class BounceBallEnvCfg(DirectEnvCfg):
    """Control a paddle to keep bouncing a table-tennis ball.

    zh_CN: 控制球拍连续颠起乒乓球。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 20.0

    # Ball and paddle physics parameters
    ball_restitution: float = 0.9  # Slightly less than perfect for realistic bouncing
    ball_linear_damping: float = 0.55
    ball_lateral_friction: float = 1.1
    paddle_restitution: float = 0.8
    paddle_linear_damping: float = 0.55
    paddle_lateral_friction: float = 1.1

    # Initial conditions
    ball_init_pos: list = None
    ball_init_vel: list = None
    arm_init_qpos: list = None

    # Target positions
    target_ball_x: float = 0.58856  # Target x position (m)
    target_ball_y: float = 0.0  # Target y position (m)

    # Target height for bouncing (configurable parameter)
    target_height_range: tuple = (0.53, 0.83)  # Range for random target height in meters
    randomize_target_height: bool = True  # Whether to randomize target height on reset
    height_tolerance: float = 0.1  # Tolerance for reward calculation

    # Paddle behavior constraints
    paddle_home_position_z: float = 0.28  # Home position height for paddle (m)
    max_paddle_height_deviation: float = 0.1  # Max deviation from home position (m)
    encourage_return_home: bool = True  # Encourage paddle to return to home position
    encourage_impact_velocity: bool = True  # Encourage high upward velocity at impact

    # Physics constants
    gravity: float = 9.81  # Gravity acceleration (m/s^2)

    # Reward function parameters
    # Position reward
    vertical_weight_scale: float = 0.15  # Scale for vertical distance weight
    weighted_horizontal_base_scale: float = 0.1  # Base scale for horizontal position
    weighted_horizontal_weight_factor: float = 3.0  # Weight factor for vertical proximity

    # Out of position penalty
    out_of_position_threshold: float = 0.05  # Threshold distance (m)
    out_of_position_sharpness: float = 0.03  # Sigmoid sharpness

    # Velocity matching
    desired_velocity_at_target: float = 0.5  # Target velocity at target height (m/s)
    velocity_error_sigma: float = 0.8  # Sigma for velocity error Gaussian

    # Height reward
    height_error_sigma: float = 0.15  # Sigma for height error Gaussian
    height_progress_scale: float = 2.0  # Scale for height progress bonus
    height_progress_threshold: float = 0.2  # Minimum height for progress bonus (m)

    # Controlled upward velocity
    positioning_quality_sigma: float = 0.02  # Sigma for positioning quality
    ideal_velocity_min: float = 0.5  # Minimum ideal launch velocity (m/s)
    ideal_velocity_max: float = 3.0  # Maximum ideal launch velocity (m/s)
    upward_velocity_sigma: float = 0.5  # Sigma for upward velocity quality
    upward_mask_scale: float = 0.1  # Scale for upward mask sigmoid
    controlled_upward_clip_max: float = 1.5  # Max clip value for controlled upward

    # Velocity penalties
    excessive_upward_threshold: float = 3.5  # Threshold for excessive upward velocity (m/s)
    excessive_upward_sharpness: float = 0.3  # Sigmoid sharpness
    downward_velocity_threshold: float = -0.2  # Threshold for downward penalty (m/s)
    downward_velocity_scale: float = 0.3  # Scale for downward penalty magnitude
    downward_velocity_clip_max: float = 0.5  # Max clip for downward penalty

    # Bounce rewards
    bounce_positioning_sigma: float = 0.05  # Sigma for bounce positioning quality
    bounce_log_scale: float = 0.5  # Scale for logarithmic bounce reward
    high_bounce_threshold: float = 2.0  # Threshold for high bounce bonus
    high_bounce_sharpness: float = 0.5  # Sigmoid sharpness for high bounce
    high_bounce_scale: float = 0.15  # Scale for high bounce bonus

    # Paddle alignment
    paddle_alignment_sigma: float = 0.03  # Sigma for paddle-ball alignment
    vertical_proximity_scale: float = 0.1  # Scale for vertical proximity weight
    paddle_alignment_weight_factor: float = 2.0  # Weight factor for vertical proximity
    bounce_boost_factor: float = 2.0  # Boost factor when bounce detected
    paddle_center_scale: float = 0.3  # Overall scale for paddle center reward

    # Home position reward
    home_position_sigma: float = 0.05  # Sigma for home position deviation
    distance_factor_threshold: float = 0.15  # Threshold for distance factor (m)
    distance_factor_sharpness: float = 0.03  # Sigmoid sharpness
    distance_factor_scale: float = 0.5  # Scale for distance factor
    height_violation_scale: float = 20.0  # Scale for height violation penalty

    # Action and velocity penalties
    action_penalty_rate: float = 1e-4  # Penalty rate for action changes
    joint_vel_penalty_rate: float = 1e-4  # Penalty rate for joint velocities

    # Reward weights
    weighted_position_weight: float = 2.0
    velocity_matching_weight: float = 2.0
    height_reward_weight: float = 4.5
    height_progress_weight: float = 1.0
    controlled_upward_weight: float = 1.5
    consecutive_bounces_weight: float = 0.8
    high_bounce_weight: float = 0.3
    paddle_center_weight: float = 0.6
    home_position_weight: float = 1.5
    out_of_position_weight: float = 1.0
    excessive_upward_weight: float = 1.0
    downward_velocity_weight: float = 1.0
    height_violation_weight: float = 1.0

    # Action scaling parameters
    action_scale: list = None
    action_bias: list = None

    # Debug options
    # Whether to store the detailed reward breakdown in state.reward_terms (very slow)
    store_reward_details: bool = False

    def __post_init__(self):
        if self.ball_init_pos is None:
            self.ball_init_pos = [0.58856, 0, 0.68]
        if self.ball_init_vel is None:
            self.ball_init_vel = [0.0, 0.0, -0.2]
        if self.arm_init_qpos is None:
            self.arm_init_qpos = [0, 40, 110, 0, -60, 0]

        if self.action_scale is None:
            self.action_scale = [0.0008] * 6
        if self.action_bias is None:
            self.action_bias = [0.0] * 6

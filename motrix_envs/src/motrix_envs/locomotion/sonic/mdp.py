# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import cast

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    ObservationTermCfg,
    ObsTerm,
    RewardTerm,
    RewardTermCfg,
    SharedArray,
    TerminationTerm,
    TerminationTermCfg,
    dispatch,
    kernel_data,
    metric,
    njit,
)
from motrix_env_core.manager.math.quaternion import inverse as quat_inverse
from motrix_env_core.manager.math.quaternion import mul as quat_mul
from motrix_env_core.manager.math.quaternion import (
    rotate_inverse,
    rotate_vector,
    rotation_distance,
    to_matrix_first_two_columns,
)
from motrix_env_core.mdp.rewards import (
    ActionRateRewardCfg as ActionRateRewardCfg,
)
from motrix_env_core.numba.manager.commands import ResetContext
from motrix_env_core.numba.manager.rand import next_uniform
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg as DofLimitRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalBodyAngularVelocityRewardCfg as GlobalBodyAngularVelocityRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalBodyLinearVelocityRewardCfg as GlobalBodyLinearVelocityRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalRefOrientationRewardCfg as GlobalRefOrientationRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalRefPositionRewardCfg as GlobalRefPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    RelativeBodyOrientationRewardCfg as RelativeBodyOrientationRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    RelativeBodyPositionRewardCfg as RelativeBodyPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    UndesiredContactsRewardCfg as UndesiredContactsRewardCfg,
)
from motrix_envs.motion.adaptive import AdaptiveMotionSampler, clip_segment_bounds, count_segment_bins
from motrix_envs.motion.rotation import CorpusRotator
from motrix_envs.motion.sonic import SonicMotionClip

# Policy joint order: the g1_sonic.xml actuator order (grouped by limb).
# Every joint-indexed plane follows this one declaration — the corpus is
# reordered by name at load, the sim queries and the action term resolve by
# name — so layouts stay internally consistent by construction. Running a
# checkpoint laid out in the upstream interleaved order would need an
# explicit permutation table.
G1_SONIC_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_SONIC_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
SONIC_VECTOR_DIM = 3
SONIC_ROTATION_REPRESENTATION_DIM = 2 * SONIC_VECTOR_DIM
SONIC_ACTOR_JOINT_FEATURES = 3
# Width of the per-env encoder mask (multi-hot, upstream column order):
# g1, teleop, smpl.
SONIC_ENCODER_COUNT = 3
# Joint selections into G1_SONIC_JOINTS, resolved by name so they follow the
# declared order instead of hardcoding positions.
SONIC_WRIST_JOINTS = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
SONIC_WRIST_POLICY_INDICES = tuple(G1_SONIC_JOINTS.index(name) for name in SONIC_WRIST_JOINTS)
SONIC_FOOT_JOINTS = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
# Hybrid ("teleop") stream inputs, resolved against G1_SONIC_JOINTS /
# G1_SONIC_BODY_NAMES. The vr_3point bodies are both wrists plus a torso point
# offset 0.35 m up (standing in for the head); the leg dof set keeps the
# upstream semantic order — the six left-leg joints (hip pitch/roll/yaw, knee,
# ankle pitch/roll) first, then the six right-leg joints.
HYBRID_VR_BODY_INDICES = (10, 13, 7)  # left wrist, right wrist, torso
HYBRID_VR_BODY_OFFSETS = ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0), (0.0, 0.0, 0.35))
HYBRID_LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
HYBRID_LEG_POLICY_INDICES = tuple(G1_SONIC_JOINTS.index(name) for name in HYBRID_LEG_JOINTS)
# Upstream teleop encoder input width: a multi-future lower-body block (12
# joint positions + velocities per G1-stride frame) followed by a
# current-frame tail of 3 vr point positions (9), 3 vr point quaternions
# xyzw (12), and the anchor orientation 6D.
HYBRID_LOWER_BODY_PER_FRAME = 2 * len(HYBRID_LEG_POLICY_INDICES)
HYBRID_TAIL_DIM = (
    len(HYBRID_VR_BODY_INDICES) * SONIC_VECTOR_DIM + len(HYBRID_VR_BODY_INDICES) * 4 + SONIC_ROTATION_REPRESENTATION_DIM
)
# Upstream ``teleop_sample_prob_when_smpl`` (release value): smpl-native rows
# also activate the teleop encoder with this probability.
HYBRID_TELEOP_PROB_WHEN_SMPL = 0.5
SONIC_ROOT_BODY_INDEX = 0  # ``pelvis`` is the first tracked body in SONIC order.
# Upstream SONIC stride contract (50 Hz references, one clip frame per ctrl step):
# the G1 encoder samples sparsely (0.1 s apart, ~1 s horizon) while the SMPL
# encoder samples densely (0.02 s apart, ~0.2 s horizon); the hybrid stream's
# lower-body block shares the G1 grid.
G1_FUTURE_STRIDE = 5
SMPL_FUTURE_STRIDE = 1
# Temporal depths are independent: future frames size the G1/SMPL/teleop
# reference encoders, while history frames size actor proprioception and critic
# history. Both must stay aligned with the SONIC policy model recipe.
SONIC_DEFAULT_NUM_FUTURE_FRAMES = 5
SONIC_DEFAULT_NUM_HISTORY_FRAMES = 5
# encoder_sampling modes select the *native* encoder per reset: 0 = mixed
# (uniform draw over the three streams), then one id per encoder_index
# column — 1 = g1, 2 = teleop, 3 = smpl — so a pinned mode maps directly
# onto its mask column. The legacy upstream multi-hot expansion then applies
# on top: smpl-native rows also activate g1 and, with probability 0.5, teleop.
ENCODER_SAMPLING_MODES = ("mixed", "g1", "teleop", "smpl")


@njit(inline="always")
def _sonic_actor_frame_dim(joint_count: int) -> int:
    return SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * SONIC_VECTOR_DIM


@njit(inline="always")
def _g1_reference_frame_dim(joint_count: int) -> int:
    return 2 * joint_count + SONIC_ROTATION_REPRESENTATION_DIM


@njit(inline="always")
def _smpl_reference_frame_dim(joint_count: int) -> int:
    return joint_count * SONIC_VECTOR_DIM + SONIC_ROTATION_REPRESENTATION_DIM + len(SONIC_WRIST_POLICY_INDICES)


def _hybrid_reference_width(num_future_frames: int) -> int:
    """Upstream teleop encoder input width: lower-body block plus static tail."""
    return num_future_frames * HYBRID_LOWER_BODY_PER_FRAME + HYBRID_TAIL_DIM


@njit(inline="always")
def _sample_encoder_slot(rng_state: np.ndarray, mode: np.int64) -> np.int64:
    """Resolve the native encoder slot for one reset.

    Column order mirrors upstream ``encoder_sample_probs``: 0 = g1,
    1 = teleop, 2 = smpl. A pinned mode id maps one-to-one onto its column;
    mixed mode draws uniformly over the three streams.
    """
    if mode > 0:
        return mode - np.int64(1)
    unit = (next_uniform(rng_state) + np.float32(1.0)) * np.float32(0.5)
    if unit < np.float32(1.0 / 3.0):
        return np.int64(0)
    if unit < np.float32(2.0 / 3.0):
        return np.int64(1)
    return np.int64(2)


@njit(inline="always")
def _sample_motion_step_from_cdf(
    rand_state,
    sampling_cdf,
    num_frames: np.int64,
    start_at_timestep_zero_prob: np.float32,
):
    """Draw one motion frame from the adaptive-bin CDF (uniformly when disabled)."""
    unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
    if sampling_cdf.size == 0:
        step_value = unit * np.float32(num_frames - 1)
    else:
        # Leftmost bin whose CDF mass reaches ``unit``; the CDF is non-decreasing
        # and ends at 1.0, so the search always terminates inside the array.
        low = np.int64(0)
        high = np.int64(sampling_cdf.size) - 1
        while low < high:
            mid = (low + high) // np.int64(2)
            if unit > sampling_cdf[mid]:
                low = mid + 1
            else:
                high = mid
        bin_id = low
        bin_unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
        phase = (np.float32(bin_id) + bin_unit) / np.float32(sampling_cdf.size)
        step_value = phase * np.float32(num_frames)
    max_value = np.float32(num_frames - 2)
    step_value = min(max(step_value, np.float32(0.0)), max_value)
    step = np.int64(step_value)
    if (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5) < start_at_timestep_zero_prob:
        step = np.int64(0)
    return step


@njit(inline="always")
def _draw_round_robin_start_step(
    rand_state,
    rr_cursor,
    rr_clip_start,
    rr_clip_frames,
    num_clips: np.int64,
) -> np.int64:
    """Draw the next start frame from a persistent clip round-robin cursor.

    Every reset claims the next clip in cyclic order and a uniform frame
    inside it: each clip receives the same number of episodes regardless of
    its length, eliminating the coverage fluctuation of iid uniform frame
    sampling over a large corpus. The cursor lives across resets, so the
    balance holds even when resets arrive at uneven times per lane.
    """
    clip_id = rr_cursor[0] % num_clips
    rr_cursor[0] += 1
    start = rr_clip_start[clip_id]
    frames = rr_clip_frames[clip_id]
    unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
    return start + np.int64(unit * np.float32(frames))


@njit(inline="always")
def _draw_adaptive_start_step(
    rand_state,
    sampling_cdf,
    window_base: np.int64,
    window_frames: np.int64,
    start_at_timestep_zero_prob: np.float32,
    pre_failure_sample_window: np.int64,
) -> np.int64:
    """Draw one start frame from the CDF, shifted back toward the failure's cause.

    Port of gear_sonic ``pre_failure_sample_window``: episode starts drawn from
    a failure bin are moved back by a uniform offset of up to the window size
    so the policy practices the approach into a hard segment instead of
    teleporting straight into it (cold-start failures carry little learning
    signal, and the actions that lead into the failure are the teachable
    part).
    """
    step = window_base + _sample_motion_step_from_cdf(
        rand_state, sampling_cdf, window_frames, start_at_timestep_zero_prob
    )
    if pre_failure_sample_window > 0:
        unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
        step -= np.int64(unit * np.float32(pre_failure_sample_window))
        if step < window_base:
            step = window_base
    return step


@njit(inline="always")
def _sample_motion_step_from_active_bins(
    rand_state,
    sampling_cdf,
    bin_start,
    bin_end,
    active_bin_count: np.int64,
    frame_source_start,
    start_at_timestep_zero_prob: np.float32,
    pre_failure_sample_window: np.int64,
):
    """Draw one start frame from the per-source adaptive bin table.

    Port of gear_sonic ``sample_motion_ids_and_time_steps``: pick a bin from
    the CDF, draw a uniform frame inside it, then shift the start back by a
    uniform offset of up to ``pre_failure_sample_window`` frames so the
    episode practices the approach into the hard segment instead of
    teleporting into a cold-start failure. Clamping is to the source clip's
    head frame — upstream's ``clamp_min(0)`` on the motion-local timeline.
    """
    unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
    low = np.int64(0)
    high = active_bin_count - np.int64(1)
    while low < high:
        mid = (low + high) // np.int64(2)
        if unit > sampling_cdf[mid]:
            low = mid + 1
        else:
            high = mid
    start = bin_start[low]
    end = bin_end[low]
    bin_unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
    step = start + np.int64(bin_unit * np.float32(end - start))
    clip_start = frame_source_start[step]
    if pre_failure_sample_window > 0:
        offset_unit = (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5)
        step -= np.int64(offset_unit * np.float32(pre_failure_sample_window))
        if step < clip_start:
            step = clip_start
    if (next_uniform(rand_state) + np.float32(1.0)) * np.float32(0.5) < start_at_timestep_zero_prob:
        step = clip_start
    return step


@njit(inline="always")
def _update_foot_joint_acceleration(
    joint_velocity: np.ndarray,
    foot_joint_policy_indices: np.ndarray,
    previous: np.ndarray,
    acceleration: np.ndarray,
    step_dt: np.float32,
) -> None:
    for index in range(foot_joint_policy_indices.shape[0]):
        current = joint_velocity[foot_joint_policy_indices[index]]
        acceleration[index] = (current - previous[index]) / step_dt
        previous[index] = current


@kernel_data
class SonicMotionCommand(CommandTerm):
    clip: SonicMotionClip
    # Active rotation window as ``[base, frames]``: reset and clip-end
    # re-anchoring sample within ``[base, base + frames)`` of the frame axis.
    # Without rotation this is ``[0, total_frames]`` and every code path is
    # identical to a fully resident corpus.
    window_range: SharedArray
    reference_index: np.int64
    target_body_position_relative: np.ndarray
    target_body_orientation_relative: np.ndarray
    # Adaptive sampling tables, republished host-side by AdaptiveMotionSampler:
    # the CDF over active per-source bins plus each bin's frame range.
    sampling_cdf: SharedArray
    adaptive_active_bin_count: SharedArray
    adaptive_bin_start: SharedArray
    adaptive_bin_end: SharedArray
    # Host-side failure handoff: reset marks terminated lanes with their
    # source clip and source-local frame; the sampler drains them per tick.
    adaptive_failure_pending: np.ndarray
    adaptive_failure_source_id: np.ndarray
    adaptive_failure_local_step: np.ndarray
    start_at_timestep_zero_prob: np.float32
    pre_failure_sample_window: np.int64
    clip_round_robin: np.bool_
    rr_cursor: SharedArray
    rr_clip_start: SharedArray
    rr_clip_frames: SharedArray
    rr_num_clips: np.int64
    encoder_index: np.ndarray
    encoder_sampling_mode: np.int64
    running_ref_height: np.ndarray
    low_reference: np.ndarray
    low_reference_height_threshold: np.float32
    reset_counter: np.ndarray
    previous_foot_joint_velocity: np.ndarray
    foot_joint_acceleration: np.ndarray
    foot_joint_policy_indices: SharedArray
    step_dt: np.float32
    num_future_frames: np.int64
    steps: np.ndarray = metric(name="motion_step", dtype=np.float32)

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        _update_foot_joint_acceleration(
            ctx.sim["robot_dof_vel"],
            self.foot_joint_policy_indices,
            self.previous_foot_joint_velocity,
            self.foot_joint_acceleration,
            self.step_dt,
        )

        step = self.steps[0]
        motion_reference_body_pos_w = self.clip.reference_body_pos_w[step]
        # Upstream low-reference contract: track the reference anchor height
        # with an EMA and relax the anchor/ee z terminations while the motion
        # crouches below the configured threshold.
        height = self.running_ref_height[0] * np.float32(0.9) + np.float32(0.1) * motion_reference_body_pos_w[2]
        self.running_ref_height[0] = height
        self.low_reference[0] = height < self.low_reference_height_threshold
        motion_reference_body_quat_w = self.clip.reference_body_quat_w[step]
        motion_tracked_bodies_pos_w = self.clip.tracked_bodies_pos_w[step]
        motion_tracked_bodies_quat_w = self.clip.tracked_bodies_quat_w[step]
        robot_reference_body_pos_w = ctx.sim["tracked_body_pos"][self.reference_index]
        robot_reference_body_quat_w = ctx.sim["tracked_body_quat"][self.reference_index]
        out_body_pos = self.target_body_position_relative
        out_body_quat = self.target_body_orientation_relative

        delta_quat = out_body_quat[0]
        quat_inverse(motion_reference_body_quat_w, delta_quat)
        quat_mul(robot_reference_body_quat_w, delta_quat, delta_quat)
        dx, dy, dz, dw = delta_quat
        half_yaw = np.float32(0.5 * math.atan2(2.0 * (dw * dz + dx * dy), 1.0 - 2.0 * (dy * dy + dz * dz)))
        yaw_z = math.sin(half_yaw)
        yaw_w = math.cos(half_yaw)
        height_delta = motion_reference_body_pos_w[2] - robot_reference_body_pos_w[2]
        for body_id in range(motion_tracked_bodies_pos_w.shape[0]):
            out_quat = out_body_quat[body_id]
            out_quat[0] = 0.0
            out_quat[1] = 0.0
            out_quat[2] = yaw_z
            out_quat[3] = yaw_w
            out_pos = out_body_pos[body_id]
            rotate_vector(
                out_quat,
                (
                    motion_tracked_bodies_pos_w[body_id, 0] - motion_reference_body_pos_w[0],
                    motion_tracked_bodies_pos_w[body_id, 1] - motion_reference_body_pos_w[1],
                    motion_tracked_bodies_pos_w[body_id, 2] - motion_reference_body_pos_w[2],
                ),
                out_pos,
            )
            out_pos[0] += robot_reference_body_pos_w[0]
            out_pos[1] += robot_reference_body_pos_w[1]
            out_pos[2] += robot_reference_body_pos_w[2] + height_delta
            quat_mul(out_quat, motion_tracked_bodies_quat_w[body_id], out_quat)

    def reset(self, ctx: ResetContext) -> None:
        """Mark terminated lanes for the adaptive sampler.

        Every termination is a failure (gear_sonic ``update_adaptive_sampling``
        credits the bin where the episode ended): the lane records its source
        clip and source-local frame, and the host sampler folds the failure
        into that clip's per-bin statistics on its next tick. Lanes still
        draining a retired rotation window keep reporting to their own source.
        """
        env_ids = ctx.env_ids
        if self.sampling_cdf.size:
            failed_env_ids = env_ids[ctx.terminated[env_ids]]
            if failed_env_ids.size:
                failed_steps = self.steps[failed_env_ids, 0]
                self.adaptive_failure_source_id[failed_env_ids] = self.clip.frame_source_id[failed_steps]
                self.adaptive_failure_local_step[failed_env_ids] = (
                    failed_steps - self.clip.frame_source_start[failed_steps]
                )
                self.adaptive_failure_pending[failed_env_ids] = True
        self.reset_counter[ctx.env_ids] += 1

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        if self.sampling_cdf.size:
            self.steps[0] = _sample_motion_step_from_active_bins(
                ctx.rand.state,
                self.sampling_cdf,
                self.adaptive_bin_start,
                self.adaptive_bin_end,
                self.adaptive_active_bin_count[0],
                self.clip.frame_source_start,
                self.start_at_timestep_zero_prob,
                self.pre_failure_sample_window,
            )
        elif self.clip_round_robin:
            self.steps[0] = _draw_round_robin_start_step(
                ctx.rand.state,
                self.rr_cursor,
                self.rr_clip_start,
                self.rr_clip_frames,
                self.rr_num_clips,
            )
        else:
            self.steps[0] = _draw_adaptive_start_step(
                ctx.rand.state,
                self.sampling_cdf,
                self.window_range[0],
                self.window_range[1],
                self.start_at_timestep_zero_prob,
                self.pre_failure_sample_window,
            )
        # Seed the finite-difference velocity from the frame the sim reset
        # teleports this env to (BodyDofPosReset writes clip.joint_vel[step]),
        # matching upstream's articulation layer seeding the previous velocity
        # at write time. Zero here would make the first post-reset update
        # report (v - 0)/dt as a large spurious acceleration.
        step = self.steps[0]
        for i in range(self.foot_joint_policy_indices.shape[0]):
            self.previous_foot_joint_velocity[i] = self.clip.joint_vel[step, self.foot_joint_policy_indices[i]]
        self.foot_joint_acceleration[:] = 0.0
        slot = _sample_encoder_slot(ctx.rand.state, self.encoder_sampling_mode)
        # ``self.encoder_index`` is the lane's writable 1-D row view.
        for encoder in range(self.encoder_index.shape[0]):
            self.encoder_index[encoder] = 1.0 if encoder == slot else 0.0
        # Upstream legacy multi-hot: smpl-native rows also activate g1 (the
        # alignment losses then see real g1 latents on smpl data) and with
        # probability ``HYBRID_TELEOP_PROB_WHEN_SMPL`` also teleop, creating
        # the tri-hot rows the g1-teleop / teleop-smpl losses train on.
        if slot == 2:
            self.encoder_index[0] = 1.0
            unit = (next_uniform(ctx.rand.state) + np.float32(1.0)) * np.float32(0.5)
            if unit < np.float32(HYBRID_TELEOP_PROB_WHEN_SMPL):
                self.encoder_index[1] = 1.0
        self.running_ref_height[0] = self.clip.reference_body_pos_w[self.steps[0], 2]
        self.low_reference[0] = self.running_ref_height[0] < self.low_reference_height_threshold

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        step = self.steps[0]
        self.steps[0] = step + 1
        if self.steps[0] > self.clip.frame_clip_end[step]:
            # The clip's final frame has been tracked for one step: switch to
            # a freshly sampled segment mid-episode. The inline draw only
            # keeps ``steps`` valid here; the reset pipeline reruns
            # ``reset_env`` for the requested lane, whose draw is
            # authoritative, and the sim reset terms teleport the robot to the
            # new frame. Episode bookkeeping and action-term state stay
            # untouched, so the episode continues until time-out or failure.
            if self.sampling_cdf.size:
                self.steps[0] = _sample_motion_step_from_active_bins(
                    ctx.rand.state,
                    self.sampling_cdf,
                    self.adaptive_bin_start,
                    self.adaptive_bin_end,
                    self.adaptive_active_bin_count[0],
                    self.clip.frame_source_start,
                    self.start_at_timestep_zero_prob,
                    self.pre_failure_sample_window,
                )
            elif self.clip_round_robin:
                self.steps[0] = _draw_round_robin_start_step(
                    ctx.rand.state,
                    self.rr_cursor,
                    self.rr_clip_start,
                    self.rr_clip_frames,
                    self.rr_num_clips,
                )
            else:
                self.steps[0] = _draw_adaptive_start_step(
                    ctx.rand.state,
                    self.sampling_cdf,
                    self.window_range[0],
                    self.window_range[1],
                    self.start_at_timestep_zero_prob,
                    self.pre_failure_sample_window,
                )
            ctx.sim_reset_requested[0] = True


@configclass(kw_only=True)
class SonicMotionCommandCfg(CommandCfg):
    # NPZ corpus source (files and/or directories; a directory expands to its
    # schema v1 clips in sorted order). One clip per file, SMPL extension
    # channels required.
    motion_files: tuple[str, ...] = ()
    # Load a deterministic random subsample of this many clips instead of the
    # whole corpus; None loads everything. Full corpora (100k+ files) spend
    # many minutes in serial decompression, so training runs pin a subsample.
    num_motions: int | None = 1024
    # Rotate the resident window of ``num_motions`` clips through the full
    # corpus every this many control steps, for off-policy training where the
    # whole corpus cannot stay resident. None keeps the startup subsample
    # fixed for the whole run. The window swap never force-resets envs; lanes
    # migrate through their natural resets and clip-end re-anchoring.
    rotation_period: int | None = None
    # Background decode threads used to prefetch the next rotation window.
    rotation_workers: int = 8
    joint_names: tuple[str, ...] = G1_SONIC_JOINTS
    tracked_body_names: tuple[str, ...] = G1_SONIC_BODY_NAMES
    reference_body_name: str = "pelvis"
    num_future_frames: int = SONIC_DEFAULT_NUM_FUTURE_FRAMES
    encoder_sampling: str = "mixed"
    # gear_sonic ``adaptive_sampling.enable`` (upstream release: true): every
    # adaptive default below mirrors the upstream release values, so the
    # default sampling behavior is equivalent. ``for_play()`` disables it for
    # evaluation and ``clip_round_robin`` replaces it.
    adaptive_sampling_enabled: bool = True
    start_at_timestep_zero_prob: float = 0.0
    low_reference_height_threshold: float = 0.5
    # gear_sonic ``uniform_sampling_rate`` (upstream release: 0.1): uniform
    # floor blended into the per-bin sampling distribution; guards every clip
    # region against starvation.
    uniform_ratio: float = 0.1
    # gear_sonic ``adp_samp_failure_rate_max_over_mean`` (upstream release:
    # 50): clip the per-bin failure RATE at this multiple of the mean rate.
    failure_rate_max_over_mean: float = 50.0
    # gear_sonic ``pre_failure_sample_window`` (upstream release: 200):
    # episode starts shift back by a uniform offset of up to this many frames,
    # so the policy practices the approach into the hard segment instead of
    # teleporting straight into a cold-start failure.
    pre_failure_sample_window: int = 200
    # Balanced clip coverage: every reset claims the next clip in cyclic order
    # and a uniform frame inside it, so each clip receives the same number of
    # episode starts regardless of length. Replaces adaptive sampling (they
    # are mutually exclusive).
    clip_round_robin: bool = False
    # gear_sonic ``bin_size``: source-clip-local bin width in frames — one
    # second at SONIC's 50 Hz source rate. Shared by the resident and rotated
    # corpus paths; a bin never straddles two clips.
    adaptive_bin_size: int = 50
    # gear_sonic ``init_num_failures`` (upstream release: 1): failure and
    # episode counters both start here, so the initial rate estimate is one
    # everywhere and the initial distribution is clip-equalized uniform.
    adaptive_init_count: float = 1.0
    # gear_sonic ``sequence_length_agnostic`` (upstream release: true):
    # equalize total probability mass across clips so long clips cannot
    # dominate short ones purely through length.
    adaptive_sequence_length_agnostic: bool = True
    # Optional gear_sonic ``max_prob_per_bin``: cap any single bin at this
    # multiple of the uniform bin probability, water-filled so the bound
    # survives normalization. Upstream ships the knob but leaves it unset in
    # the release config, so None disables it here too.
    adaptive_max_prob_per_bin: float | None = None
    # Republish the host-side active-window CDF after this many control steps
    # with accumulated statistics. A window publication republishes eagerly.
    adaptive_update_period: int = 128
    # gear_sonic motion-level resample (``update_adaptive_sampling_motion_sequences``
    # + ``ImResampleCallback``): fraction of each rotation window's clip slots
    # drawn with replacement from the sampler's per-clip difficulty weights,
    # deduplicated within the window. The remaining slots keep coming from the
    # without-replacement coverage permutation, so every clip still rides the
    # pass (off-policy buffer diversity) while hard clips reappear more often.
    # Requires adaptive sampling; 0 disables and keeps pure coverage rotation.
    rotation_difficulty_ratio: float = 0.5
    # gear_sonic ``tracking_local_vr_5point_error`` under the sonic_release
    # override: the release trains on three anchor-local points — torso_link
    # shifted 0.5 m up (standing in for the head) plus both wrists with zero
    # offset. The generic function also accepts the motion.yaml default
    # five points (pelvis, both wrists on their forearm offsets, both
    # ankles), which the release replaces.
    reward_point_body_names: tuple[str, ...] = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    reward_point_body_offsets: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    def __post_init__(self) -> None:
        if isinstance(self.num_future_frames, bool) or self.num_future_frames <= 0:
            raise ValueError("SONIC num_future_frames must be a positive integer")
        if isinstance(self.num_motions, bool) or (self.num_motions is not None and self.num_motions <= 0):
            raise ValueError("SONIC num_motions must be a positive integer or None")
        if self.rotation_period is not None and (isinstance(self.rotation_period, bool) or self.rotation_period <= 0):
            raise ValueError("SONIC rotation_period must be a positive integer or None")
        if self.rotation_period is not None and self.num_motions is None:
            raise ValueError("SONIC rotation needs num_motions to size the per-window clip count")
        if isinstance(self.rotation_workers, bool) or self.rotation_workers < 2:
            raise ValueError("SONIC rotation_workers must be an integer >= 2")
        if isinstance(self.adaptive_bin_size, bool) or self.adaptive_bin_size <= 0:
            raise ValueError("SONIC adaptive_bin_size must be a positive integer")
        if isinstance(self.adaptive_update_period, bool) or self.adaptive_update_period <= 0:
            raise ValueError("SONIC adaptive_update_period must be a positive integer")
        if not 0.0 <= self.uniform_ratio <= 1.0:
            raise ValueError("SONIC uniform_ratio must be in [0, 1]")
        if self.failure_rate_max_over_mean <= 0.0:
            raise ValueError("SONIC failure_rate_max_over_mean must be positive")
        if isinstance(self.pre_failure_sample_window, bool) or self.pre_failure_sample_window < 0:
            raise ValueError("SONIC pre_failure_sample_window must be a non-negative integer")
        if self.adaptive_init_count <= 0.0:
            raise ValueError("SONIC adaptive_init_count must be positive")
        if self.adaptive_max_prob_per_bin is not None and self.adaptive_max_prob_per_bin < 1.0:
            raise ValueError("SONIC adaptive_max_prob_per_bin must be >= 1 (cap below uniform is degenerate)")
        if not 0.0 <= self.rotation_difficulty_ratio <= 1.0:
            raise ValueError("SONIC rotation_difficulty_ratio must be in [0, 1]")
        if not isinstance(self.clip_round_robin, bool):
            raise ValueError("SONIC clip_round_robin must be a boolean")
        if self.encoder_sampling not in ENCODER_SAMPLING_MODES:
            raise ValueError(f"SONIC encoder_sampling must be one of {ENCODER_SAMPLING_MODES}")
        if self.motion_files and not all(isinstance(path, str) and path for path in self.motion_files):
            raise ValueError("SONIC motion_files entries must be non-empty path strings")
        if len(self.reward_point_body_names) != len(self.reward_point_body_offsets):
            raise ValueError("SONIC reward-point body names and offsets must have equal length")
        if len(set(self.reward_point_body_names)) != len(self.reward_point_body_names):
            raise ValueError("SONIC reward-point body names must be unique")
        if any(name not in self.tracked_body_names for name in self.reward_point_body_names):
            raise ValueError("SONIC reward-point body names must be tracked bodies")
        offsets = np.asarray(self.reward_point_body_offsets, dtype=np.float32)
        if offsets.shape != (len(self.reward_point_body_names), 3) or not np.isfinite(offsets).all():
            raise ValueError("SONIC reward-point offsets must be finite XYZ vectors")

    def __call__(self, env: ManagerEnv) -> CommandTerm:
        robot = cast(RobotCfg, env.cfg.scene.objs.robot)
        if not self.motion_files:
            raise ValueError("SONIC motion_files must name at least one npz file or directory.")
        rotator: CorpusRotator | None = None
        if self.rotation_period is None:
            clip = SonicMotionClip.from_corpus(
                self.motion_files,
                joint_names=self.joint_names,
                tracked_body_names=self.tracked_body_names,
                reference_body_name=self.reference_body_name,
                root_body_name=robot.base_link_name,
                num_motions=self.num_motions,
            )
        else:
            clip, rotator = CorpusRotator.build(
                paths=self.motion_files,
                window_clips=self.num_motions,
                period=self.rotation_period,
                workers=self.rotation_workers,
                seed=env.seed,
                joint_names=list(self.joint_names),
                tracked_body_names=self.tracked_body_names,
                reference_body_name=self.reference_body_name,
                root_body_name=robot.base_link_name,
                track_sources=self.adaptive_sampling_enabled,
            )
        window_range = np.asarray(
            [0, clip.joint_pos.shape[0] if rotator is None else rotator.active_frames], dtype=np.int64
        )
        reference_index = self.tracked_body_names.index(self.reference_body_name)
        if self.clip_round_robin and self.adaptive_sampling_enabled:
            raise ValueError("clip_round_robin replaces adaptive sampling; enable only one")
        # Per-source bin capacity: the exact count for a resident corpus, the
        # frame-budget bound for any rotated window.
        if not self.adaptive_sampling_enabled:
            nbin = 0
        elif rotator is not None:
            nbin = rotator.max_adaptive_bins(self.adaptive_bin_size)
        else:
            nbin = count_segment_bins(
                clip_segment_bounds(clip.frame_clip_end, 0, int(window_range[1])), self.adaptive_bin_size
            )
        # Round-robin clip table: cyclic per-clip frame ranges over the resident
        # window, so every clip receives the same number of episode starts.
        rr_clip_start = np.empty(0, np.int64)
        rr_clip_frames = np.empty(0, np.int64)
        if self.clip_round_robin:
            source_starts = np.asarray(clip.frame_source_start, dtype=np.int64)
            starts = np.unique(source_starts)
            rr_clip_start = starts
            rr_clip_frames = np.append(starts[1:], source_starts.size) - starts
        # Term builds run before the read program exists, so derive the
        # tracked-body layout statically instead of reading env.sim_data.
        shape = (len(self.tracked_body_names), 3)
        command = SonicMotionCommand(
            clip=clip,
            window_range=window_range,
            reference_index=np.int64(reference_index),
            steps=np.zeros((env.num_envs, 1), np.int64),
            reset_counter=np.zeros((env.num_envs, 1), np.int64),
            previous_foot_joint_velocity=np.zeros((env.num_envs, 4), np.float32),
            foot_joint_acceleration=np.zeros((env.num_envs, 4), np.float32),
            foot_joint_policy_indices=np.asarray(
                [G1_SONIC_JOINTS.index(name) for name in SONIC_FOOT_JOINTS], dtype=np.int64
            ),
            step_dt=np.float32(env.cfg.ctrl_dt),
            encoder_index=np.tile(np.asarray((1.0, 0.0, 0.0), np.float32), (env.num_envs, 1)),
            encoder_sampling_mode=np.int64(ENCODER_SAMPLING_MODES.index(self.encoder_sampling)),
            running_ref_height=np.zeros((env.num_envs, 1), np.float32),
            low_reference=np.zeros((env.num_envs, 1), bool),
            low_reference_height_threshold=np.float32(self.low_reference_height_threshold),
            command=np.empty((env.num_envs, 2 * len(self.joint_names)), np.float32),
            target_body_position_relative=np.empty((env.num_envs, *shape), np.float32),
            target_body_orientation_relative=np.empty((env.num_envs, shape[0], 4), np.float32),
            adaptive_active_bin_count=np.zeros(1, np.int64),
            sampling_cdf=np.ones(nbin, np.float32) if nbin else np.empty((0,), np.float32),
            adaptive_bin_start=np.empty(nbin, np.int64) if nbin else np.empty((0,), np.int64),
            adaptive_bin_end=np.empty(nbin, np.int64) if nbin else np.empty((0,), np.int64),
            adaptive_failure_pending=np.zeros(env.num_envs, bool),
            adaptive_failure_source_id=np.full(env.num_envs, -1, np.int64),
            adaptive_failure_local_step=np.zeros(env.num_envs, np.int64),
            start_at_timestep_zero_prob=np.float32(self.start_at_timestep_zero_prob),
            pre_failure_sample_window=np.int64(self.pre_failure_sample_window),
            clip_round_robin=np.bool_(self.clip_round_robin),
            rr_cursor=np.zeros(1, np.int64),
            rr_clip_start=rr_clip_start,
            rr_clip_frames=rr_clip_frames,
            rr_num_clips=np.int64(rr_clip_start.size),
            num_future_frames=np.int64(self.num_future_frames),
        )
        if rotator is not None:
            # Kernel-data leaves are copied into env-owned arrays after this
            # call, so the rotator itself must travel outside the command
            # tree; the SONIC env subclass picks it up once construction
            # finishes and ticks it every control step.
            if not hasattr(env, "_pending_motion_rotation"):
                raise ValueError("SONIC corpus rotation requires the SONIC manager env (rotation handoff missing).")
            env._pending_motion_rotation = rotator
        if self.adaptive_sampling_enabled:
            # Same handoff for the host-side sampler: it owns the per-source
            # failure-rate statistics and republishes the command's sampling
            # tables (CDF + active bin ranges) every tick.
            if not hasattr(env, "_pending_motion_adaptive"):
                raise ValueError("SONIC adaptive sampling requires the SONIC manager env (sampler handoff missing).")
            sampler = AdaptiveMotionSampler(
                bin_size=self.adaptive_bin_size,
                init_count=self.adaptive_init_count,
                uniform_ratio=self.uniform_ratio,
                failure_rate_max_over_mean=self.failure_rate_max_over_mean,
                sequence_length_agnostic=self.adaptive_sequence_length_agnostic,
                max_prob_per_bin=self.adaptive_max_prob_per_bin,
                update_period=self.adaptive_update_period,
            )
            env._pending_motion_adaptive = sampler
            if rotator is not None and self.rotation_difficulty_ratio > 0.0:
                rotator.enable_difficulty_sampling(sampler.source_sampling_weights, self.rotation_difficulty_ratio)
        return command


# Upstream observation-noise amplitudes (gear_sonic ``local_dir_hist.yaml``
# proprio terms + ``unitoken_all_noz.yaml`` reference terms): additive uniform
# noise applied per feature at observation readout. Actions, the reference
# joint position/velocity blocks, the SMPL wrist tail, the hybrid lower-body
# and vr blocks, the encoder mask, and every critic term carry no noise
# upstream. Noise stays on in play/eval — upstream never disables corruption.
SONIC_ANG_VEL_NOISE = 0.2
SONIC_JOINT_POS_NOISE = 0.01
SONIC_JOINT_VEL_NOISE = 0.5
SONIC_GRAVITY_NOISE = 0.05
SONIC_REFERENCE_NOISE = 0.05


@njit(inline="always")
def _add_observation_noise(out: np.ndarray, amplitudes: np.ndarray, rng_state: np.ndarray) -> None:
    """Per-dimension additive uniform noise; zero-amplitude dims stay exact."""
    for i in range(out.shape[0]):
        if amplitudes[i] != 0.0:
            out[i] += next_uniform(rng_state) * amplitudes[i]


@kernel_data
class SonicActorObservation:
    history: np.ndarray
    seen_reset: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, state: SonicActorObservation, noise: np.ndarray) -> None:
        motion = ctx.commands["motion"]
        action = ctx.actions["joint_position"]
        # Observation receivers live under ``ctx.observations`` and are
        # therefore shared by all lanes.  Select this lane's history row
        # explicitly; otherwise ``self.history[0]`` is the whole history matrix
        # array and quaternion helpers receive a 2-D output during Numba
        # compilation.
        env_id = ctx.env_id
        history = state.history[env_id]
        # Proprioception is expressed in the robot root (pelvis) frame.  The
        # SONIC's actor observation uses the robot root (pelvis) state.
        root_q = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        root_av = ctx.sim["tracked_body_angular_velocity"][SONIC_ROOT_BODY_INDEX]
        reset_history = state.seen_reset[env_id] != motion.reset_counter[0]
        h = history.shape[0]
        if reset_history:
            frame = history[0]
        else:
            # The released local_dir_hist contract is oldest-to-newest.
            for i in range(h - 1):
                history[i] = history[i + 1]
            frame = history[h - 1]
        joint_count = action.action_scales.shape[0]
        root_dim = SONIC_VECTOR_DIM
        pos_end = root_dim + joint_count
        action_start = pos_end + joint_count
        gravity_start = action_start + joint_count
        rotate_inverse(root_q, root_av, frame[:root_dim])
        # Explicit loop instead of an array subtraction, which would allocate
        # a temporary inside the fused kernel lane.
        dof_pos = ctx.sim["robot_dof_pos"]
        for j in range(joint_count):
            frame[root_dim + j] = dof_pos[j] - action.default_angles[j]
        frame[pos_end:action_start] = ctx.sim["robot_dof_vel"]
        frame[action_start:gravity_start] = action.action_queue[action.action_ptr[0]]
        rotate_inverse(root_q, (0.0, 0.0, -1.0), frame[gravity_start : gravity_start + SONIC_VECTOR_DIM])
        if reset_history:
            # ``np.tile`` is not supported by Numba's nopython mode.  Keep
            # this reset path allocation-free with explicit row copies.
            for i in range(1, h):
                history[i] = frame
            state.seen_reset[env_id] = motion.reset_counter[0]
        o0, o1, o2, o3 = (
            root_dim * h,
            (root_dim + joint_count) * h,
            (root_dim + 2 * joint_count) * h,
            (root_dim + 3 * joint_count) * h,
        )
        # Slices such as ``history[:, :3]`` are strided views and Numba
        # cannot reshape them without first materialising a contiguous array.
        # Emit the feature-major layout directly.
        for i in range(h):
            out[root_dim * i : root_dim * i + root_dim] = history[i, :root_dim]
            out[o0 + joint_count * i : o0 + joint_count * i + joint_count] = history[i, root_dim:pos_end]
            out[o1 + joint_count * i : o1 + joint_count * i + joint_count] = history[i, pos_end:action_start]
            out[o2 + joint_count * i : o2 + joint_count * i + joint_count] = history[i, action_start:gravity_start]
            out[o3 + SONIC_VECTOR_DIM * i : o3 + SONIC_VECTOR_DIM * (i + 1)] = history[i, gravity_start:]
        _add_observation_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class SonicActorObservationCfg(ObservationTermCfg):
    num_history_frames: int = SONIC_DEFAULT_NUM_HISTORY_FRAMES

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        joint_count = len(env.model.bodies["robot"].joint_names)
        frame_dim = _sonic_actor_frame_dim(joint_count)
        state = SonicActorObservation(
            np.zeros((env.num_envs, self.num_history_frames, frame_dim), np.float32),
            np.full(env.num_envs, -1, np.int64),
        )
        return ObsTerm(
            self.num_history_frames * frame_dim,
            SonicActorObservation.compute,
            state,
            _actor_observation_noise(self.num_history_frames, joint_count),
        )


def _actor_observation_noise(history: int, joint_count: int) -> np.ndarray:
    """Per-dim uniform noise for the feature-major proprio layout.

    Blocks: [ang_vel 3H | joint_pos JH | joint_vel JH | actions JH | gravity
    3H] — the actions block stays unnoised (upstream ``local_dir_hist``).
    """
    amplitudes = np.zeros(history * (SONIC_ACTOR_JOINT_FEATURES * joint_count + 2 * SONIC_VECTOR_DIM), np.float32)
    ang_vel_end = SONIC_VECTOR_DIM * history
    amplitudes[:ang_vel_end] = SONIC_ANG_VEL_NOISE
    joint_end = ang_vel_end + joint_count * history
    amplitudes[ang_vel_end:joint_end] = SONIC_JOINT_POS_NOISE
    vel_end = joint_end + joint_count * history
    amplitudes[joint_end:vel_end] = SONIC_JOINT_VEL_NOISE
    amplitudes[vel_end + joint_count * history :] = SONIC_GRAVITY_NOISE
    return amplitudes


@kernel_data
class SonicG1ReferenceObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, noise: np.ndarray) -> None:
        m = ctx.commands["motion"]
        clip = m.clip
        q = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        n = int(m.num_future_frames)
        stride = G1_FUTURE_STRIDE
        joint_count = clip.joint_pos.shape[1]
        position_width = n * joint_count
        velocity_start = position_width
        rotation_start = 2 * position_width
        inverse_root = np.empty(4, np.float32)
        quat_inverse(q, inverse_root)
        # ``quat_mul`` overwrites every component, so one scratch quaternion
        # serves all frames without per-frame Numba allocations.
        rel = np.empty(4, np.float32)
        for i in range(n):
            step = min(int(m.steps[0]) + i * stride, clip.frame_clip_end[int(m.steps[0])])
            position_offset = i * joint_count
            velocity_offset = velocity_start + position_offset
            rotation_offset = rotation_start + i * SONIC_ROTATION_REPRESENTATION_DIM
            out[position_offset : position_offset + joint_count] = clip.joint_pos[step]
            out[velocity_offset : velocity_offset + joint_count] = clip.joint_vel[step]
            quat_mul(inverse_root, clip.tracked_bodies_quat_w[step, SONIC_ROOT_BODY_INDEX], rel)
            to_matrix_first_two_columns(
                rel,
                out[rotation_offset : rotation_offset + SONIC_ROTATION_REPRESENTATION_DIM],
            )
        _add_observation_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class SonicG1ReferenceObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        joint_count = motion.clip.joint_pos.shape[1]
        size = int(motion.num_future_frames) * _g1_reference_frame_dim(joint_count)
        return ObsTerm(
            size,
            SonicG1ReferenceObservation.compute,
            _g1_reference_noise(int(motion.num_future_frames), joint_count),
        )


def _g1_reference_noise(num_future_frames: int, joint_count: int) -> np.ndarray:
    """Per-dim noise for [joint_pos F*J | joint_vel F*J | rotation 6F]: rotations only."""
    amplitudes = np.zeros(num_future_frames * _g1_reference_frame_dim(joint_count), np.float32)
    amplitudes[2 * num_future_frames * joint_count :] = SONIC_REFERENCE_NOISE
    return amplitudes


@kernel_data
class SonicSmplReferenceObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, noise: np.ndarray) -> None:
        m = ctx.commands["motion"]
        clip = m.clip
        n = int(m.num_future_frames)
        joint_count = clip.smpl_joints.shape[1]
        joints_dim = joint_count * SONIC_VECTOR_DIM
        human_frame_dim = joints_dim + SONIC_ROTATION_REPRESENTATION_DIM
        wrist_start = n * human_frame_dim
        inverse_robot_root = np.empty(4, np.float32)
        quat_inverse(ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX], inverse_robot_root)
        # Both scratch quaternions are fully overwritten per frame, so they are
        # allocated once instead of once per future frame.
        inverse_human_root = np.empty(4, np.float32)
        relative_human_root = np.empty(4, np.float32)
        for i in range(n):
            step = min(int(m.steps[0]) + i * SMPL_FUTURE_STRIDE, clip.frame_clip_end[int(m.steps[0])])
            start = i * human_frame_dim
            rotation_start = start + joints_dim
            quat_inverse(clip.smpl_root_quat[step], inverse_human_root)
            for joint_index in range(joint_count):
                joint_start = start + joint_index * SONIC_VECTOR_DIM
                rotate_vector(
                    inverse_human_root,
                    clip.smpl_joints[step, joint_index],
                    out[joint_start : joint_start + SONIC_VECTOR_DIM],
                )
            quat_mul(inverse_robot_root, clip.smpl_root_quat[step], relative_human_root)
            to_matrix_first_two_columns(
                relative_human_root,
                out[rotation_start : rotation_start + SONIC_ROTATION_REPRESENTATION_DIM],
            )
            wrist_offset = wrist_start + i * len(SONIC_WRIST_POLICY_INDICES)
            for wrist_index, policy_index in enumerate(SONIC_WRIST_POLICY_INDICES):
                out[wrist_offset + wrist_index] = clip.joint_pos[step, policy_index]
        _add_observation_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class SonicSmplReferenceObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        joint_count = motion.clip.smpl_joints.shape[1]
        size = int(motion.num_future_frames) * _smpl_reference_frame_dim(joint_count)
        return ObsTerm(
            size,
            SonicSmplReferenceObservation.compute,
            _smpl_reference_noise(int(motion.num_future_frames), joint_count),
        )


def _smpl_reference_noise(num_future_frames: int, joint_count: int) -> np.ndarray:
    """Per-dim noise for [body (3J+6)F | wrist 6F]: body blocks only (upstream
    noises ``smpl_joints_multi_future_local_nonflat`` and
    ``smpl_root_ori_b_multi_future``; the wrist tail stays clean)."""
    body = num_future_frames * (joint_count * SONIC_VECTOR_DIM + SONIC_ROTATION_REPRESENTATION_DIM)
    wrist = num_future_frames * len(SONIC_WRIST_POLICY_INDICES)
    amplitudes = np.zeros(body + wrist, np.float32)
    amplitudes[:body] = SONIC_REFERENCE_NOISE
    return amplitudes


@kernel_data
class SonicHybridReferenceObservation:
    """Third reference stream: the hybrid (upstream "teleop") channel.

    Flat upstream release layout: a multi-future lower-body block — the 12
    leg joint positions over the G1-stride future frames, then the same
    frames' velocities — followed by a current-frame tail: the vr 3-point
    targets (both wrists plus a torso offset standing in for the head) as
    positions and xyzw quaternions canonicalized in the current
    reference-anchor local frame, and the motion anchor orientation
    relative to the robot root. Everything derives from the standard corpus
    arrays, so the stream needs no extra data channels; hybrid-disabled
    tasks build a zero-width term and keep the legacy observation contract
    untouched.
    """

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, noise: np.ndarray) -> None:
        m = ctx.commands["motion"]
        clip = m.clip
        n = int(m.num_future_frames)
        point_count = len(HYBRID_VR_BODY_INDICES)
        leg_count = len(HYBRID_LEG_POLICY_INDICES)
        now = int(m.steps[0])
        last = clip.frame_clip_end[now]
        # Lower-body reference on the G1 stride grid: all position frames
        # first, then all velocity frames, matching the upstream concat order.
        velocity_offset = n * leg_count
        for i in range(n):
            step = min(now + i * G1_FUTURE_STRIDE, last)
            position_start = i * leg_count
            for leg in range(leg_count):
                policy_index = HYBRID_LEG_POLICY_INDICES[leg]
                out[position_start + leg] = clip.joint_pos[step, policy_index]
                out[velocity_offset + position_start + leg] = clip.joint_vel[step, policy_index]
        # Current-frame vr targets canonicalized in the reference-anchor
        # (pelvis) local frame: body position plus the body-local offset.
        tail = velocity_offset + n * leg_count
        anchor_quat = clip.tracked_bodies_quat_w[now, SONIC_ROOT_BODY_INDEX]
        anchor_pos = clip.tracked_bodies_pos_w[now, SONIC_ROOT_BODY_INDEX]
        inverse_anchor = np.empty(4, np.float32)
        quat_inverse(anchor_quat, inverse_anchor)
        relative_quat = np.empty(4, np.float32)
        offset_world = np.empty(3, np.float32)
        target_world = np.empty(3, np.float32)
        for point in range(point_count):
            body = HYBRID_VR_BODY_INDICES[point]
            rotate_vector(
                clip.tracked_bodies_quat_w[now, body],
                HYBRID_VR_BODY_OFFSETS[point],
                offset_world,
            )
            target_world[0] = clip.tracked_bodies_pos_w[now, body, 0] + offset_world[0] - anchor_pos[0]
            target_world[1] = clip.tracked_bodies_pos_w[now, body, 1] + offset_world[1] - anchor_pos[1]
            target_world[2] = clip.tracked_bodies_pos_w[now, body, 2] + offset_world[2] - anchor_pos[2]
            pos_start = tail + point * SONIC_VECTOR_DIM
            rotate_vector(
                inverse_anchor,
                (target_world[0], target_world[1], target_world[2]),
                out[pos_start : pos_start + SONIC_VECTOR_DIM],
            )
        quat_start = tail + point_count * SONIC_VECTOR_DIM
        for point in range(point_count):
            body = HYBRID_VR_BODY_INDICES[point]
            quat_mul(inverse_anchor, clip.tracked_bodies_quat_w[now, body], relative_quat)
            start = quat_start + point * 4
            out[start] = relative_quat[0]
            out[start + 1] = relative_quat[1]
            out[start + 2] = relative_quat[2]
            out[start + 3] = relative_quat[3]
        # Motion anchor orientation relative to the robot's live anchor.
        anchor_start = quat_start + point_count * 4
        inverse_robot_root = np.empty(4, np.float32)
        quat_inverse(ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX], inverse_robot_root)
        quat_mul(inverse_robot_root, clip.tracked_bodies_quat_w[now, SONIC_ROOT_BODY_INDEX], relative_quat)
        to_matrix_first_two_columns(relative_quat, out[anchor_start : anchor_start + SONIC_ROTATION_REPRESENTATION_DIM])
        _add_observation_noise(out, noise, ctx.rand.state)


@configclass(kw_only=True)
class SonicHybridReferenceObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        size = _hybrid_reference_width(int(env.command_terms["motion"].num_future_frames))
        return ObsTerm(
            size,
            SonicHybridReferenceObservation.compute,
            _hybrid_reference_noise(size),
        )


def _hybrid_reference_noise(size: int) -> np.ndarray:
    """Per-dim noise for [lower-body | vr pos | vr quat | anchor 6]: anchor only."""
    amplitudes = np.zeros(size, np.float32)
    amplitudes[-SONIC_ROTATION_REPRESENTATION_DIM:] = SONIC_REFERENCE_NOISE
    return amplitudes


@kernel_data
class SonicEncoderIndexObservation:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray) -> None:
        out[:] = ctx.commands["motion"].encoder_index


@configclass(kw_only=True)
class SonicEncoderIndexObservationCfg(ObservationTermCfg):
    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        return ObsTerm(motion.encoder_index.shape[1], SonicEncoderIndexObservation.compute)


@kernel_data
class SonicCriticObservation:
    """Port of the gear_sonic ``privileged_mf_hist`` critic group (sonic_release).

    Layout (feature-grouped, upstream term order): [future reference joints —
    all positions then all velocities on the G1-stride grid |
    motion_anchor_pos_b (3, the raw reference anchor expressed in the robot
    anchor frame) | motion_anchor_ori_b (6, reference anchor relative to the
    robot anchor) | robot_body_pos_b (3 per tracked body, robot anchor frame) |
    robot_body_ori_b (6 per tracked body) | proprioception history
    (``lin_vel | ang_vel | joint_pos | joint_vel | action`` blocks, frames
    oldest-to-newest within each block)]. The upstream group carries no
    observation noise.
    """

    history: np.ndarray
    seen_reset: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, out: np.ndarray, state: SonicCriticObservation) -> None:
        motion = ctx.commands["motion"]
        clip = motion.clip
        action = ctx.actions["joint_position"]
        joint_count = clip.joint_pos.shape[1]
        step = int(motion.steps[0])
        num_future_frames = int(motion.num_future_frames)
        width = num_future_frames * joint_count
        last = clip.frame_clip_end[step]
        for i in range(num_future_frames):
            future = min(step + G1_FUTURE_STRIDE * i, last)
            start = i * joint_count
            out[start : start + joint_count] = clip.joint_pos[future]
            out[width + start : width + start + joint_count] = clip.joint_vel[future]
        root_pos = ctx.sim["tracked_body_pos"][SONIC_ROOT_BODY_INDEX]
        root_quat = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        inverse = np.empty(4, np.float32)
        relative = np.empty(4, np.float32)
        quat_inverse(root_quat, inverse)
        # motion_anchor_pos_b / motion_anchor_ori_b: the raw current-frame
        # reference anchor relative to the live robot anchor. The z component
        # keeps the privileged height error; the xy part is the drift between
        # the robot and the clip's recorded world trajectory.
        anchor_start = 2 * width
        ref_anchor_pos = clip.reference_body_pos_w[step]
        rotate_vector(
            inverse,
            (ref_anchor_pos[0] - root_pos[0], ref_anchor_pos[1] - root_pos[1], ref_anchor_pos[2] - root_pos[2]),
            out[anchor_start : anchor_start + SONIC_VECTOR_DIM],
        )
        quat_mul(inverse, clip.reference_body_quat_w[step], relative)
        to_matrix_first_two_columns(relative, out[anchor_start + 3 : anchor_start + 9])
        # robot_body_pos_b / robot_body_ori_b: every tracked body pose in the
        # robot anchor (pelvis) frame.
        tracked_body_pos = ctx.sim["tracked_body_pos"]
        tracked_body_quat = ctx.sim["tracked_body_quat"]
        body_count = clip.tracked_bodies_pos_w.shape[1]
        pos_base = anchor_start + 9
        ori_base = pos_base + SONIC_VECTOR_DIM * body_count
        for k in range(body_count):
            body_pos = tracked_body_pos[k]
            rotate_vector(
                inverse,
                (body_pos[0] - root_pos[0], body_pos[1] - root_pos[1], body_pos[2] - root_pos[2]),
                out[pos_base + k * SONIC_VECTOR_DIM : pos_base + (k + 1) * SONIC_VECTOR_DIM],
            )
            quat_mul(inverse, tracked_body_quat[k], relative)
            rotation_start = ori_base + k * SONIC_ROTATION_REPRESENTATION_DIM
            to_matrix_first_two_columns(
                relative, out[rotation_start : rotation_start + SONIC_ROTATION_REPRESENTATION_DIM]
            )
        offset = ori_base + SONIC_ROTATION_REPRESENTATION_DIM * body_count
        history = state.history[ctx.env_id]
        num_history_frames = history.shape[0]
        reset = state.seen_reset[ctx.env_id] != motion.reset_counter[0]
        for i in range(num_history_frames - 1):
            history[i] = history[i + 1]
        frame = history[num_history_frames - 1]
        pelvis_quat = ctx.sim["tracked_body_quat"][SONIC_ROOT_BODY_INDEX]
        rotate_inverse(pelvis_quat, ctx.sim["tracked_body_linear_velocity"][SONIC_ROOT_BODY_INDEX], frame[:3])
        rotate_inverse(pelvis_quat, ctx.sim["tracked_body_angular_velocity"][SONIC_ROOT_BODY_INDEX], frame[3:6])
        dof_pos = ctx.sim["robot_dof_pos"]
        for j in range(joint_count):
            frame[6 + j] = dof_pos[j] - action.default_angles[j]
        frame[6 + joint_count : 6 + 2 * joint_count] = ctx.sim["robot_dof_vel"]
        frame[6 + 2 * joint_count :] = action.action_queue[action.action_ptr[0]]
        if reset:
            for i in range(num_history_frames - 1):
                history[i] = frame
            state.seen_reset[ctx.env_id] = motion.reset_counter[0]
        feature_start = 0
        for feature_width in (3, 3, joint_count, joint_count, joint_count):
            for i in range(num_history_frames):
                start = offset + i * feature_width
                out[start : start + feature_width] = history[i, feature_start : feature_start + feature_width]
            offset += num_history_frames * feature_width
            feature_start += feature_width


@configclass(kw_only=True)
class SonicCriticObservationCfg(ObservationTermCfg):
    num_history_frames: int = SONIC_DEFAULT_NUM_HISTORY_FRAMES

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        motion = env.command_terms["motion"]
        joint_count = motion.clip.joint_pos.shape[1]
        body_count = motion.clip.tracked_bodies_pos_w.shape[1]
        # One proprioceptive history frame is [lin_vel | ang_vel | joint_pos |
        # joint_vel | action] — the same 6 + 3J width as an actor frame but a
        # different feature mix (base linear velocity replaces gravity).
        history_frame_dim = 6 + 3 * joint_count
        size = (
            2 * int(motion.num_future_frames) * joint_count
            + 9
            + 9 * body_count
            + self.num_history_frames * history_frame_dim
        )
        state = SonicCriticObservation(
            np.zeros((env.num_envs, self.num_history_frames, history_frame_dim), np.float32),
            np.full(env.num_envs, -1, np.int64),
        )
        return ObsTerm(size, SonicCriticObservation.compute, state)


@kernel_data
class SonicAntiShakeReward:
    body_indices: SharedArray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAntiShakeReward) -> float:
        v = ctx.sim["tracked_body_angular_velocity"]
        total = 0.0
        for i in state.body_indices:
            norm = math.sqrt(v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2])
            excess = max(norm - 1.5, 0.0)
            total += excess * excess
        return total / state.body_indices.shape[0]


@configclass(kw_only=True)
class SonicAntiShakeRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
        tracked = env.cfg.commands.motion.tracked_body_names
        state = SonicAntiShakeReward(np.asarray([tracked.index(name) for name in names], dtype=np.int64))
        return RewardTerm(SonicAntiShakeReward.compute, state)


@kernel_data
class SonicFeetAccelerationReward:
    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext) -> float:
        acceleration = ctx.commands["motion"].foot_joint_acceleration
        total = 0.0
        for index in range(acceleration.shape[0]):
            total += acceleration[index] * acceleration[index]
        return total


@configclass(kw_only=True)
class SonicFeetAccelerationRewardCfg(RewardTermCfg):
    def __call__(self, env: ManagerEnv) -> RewardTerm:
        return RewardTerm(SonicFeetAccelerationReward.compute)


@kernel_data
class SonicAnchorPosTermination:
    threshold: np.float32
    low_reference_threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAnchorPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"][m.reference_index]
        threshold = state.low_reference_threshold if m.low_reference[0] else state.threshold
        return abs(m.clip.root_body_pos_w[m.steps[0], 2] - p[2]) > threshold


@configclass(kw_only=True)
class SonicAnchorPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.15
    low_reference_threshold: float = 0.75

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicAnchorPosTermination(np.float32(self.threshold), np.float32(self.low_reference_threshold))
        return TerminationTerm(SonicAnchorPosTermination.compute, state)


@kernel_data
class SonicAnchorOriTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicAnchorOriTermination) -> bool:
        m = ctx.commands["motion"]
        q = ctx.sim["tracked_body_quat"][m.reference_index]
        error = rotation_distance(m.clip.root_body_quat_w[m.steps[0]], q)
        return error * error > state.threshold


@configclass(kw_only=True)
class SonicAnchorOriTerminationCfg(TerminationTermCfg):
    threshold: float = 0.2

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicAnchorOriTermination(np.float32(self.threshold))
        return TerminationTerm(SonicAnchorOriTermination.compute, state)


@kernel_data
class SonicFeetPosTermination:
    threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicFeetPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        # Compare against the per-step re-anchored reference (upstream
        # ``body_pos_relative_w``), not the raw clip world arrays: the robot
        # follows the re-anchored path, which diverges from the clip's recorded
        # world path as soon as the motion turns.
        target = m.target_body_position_relative
        for i in (3, 6):
            dx = p[i, 0] - target[i, 0]
            dy = p[i, 1] - target[i, 1]
            dz = p[i, 2] - target[i, 2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) > state.threshold:
                return True
        return False


@configclass(kw_only=True)
class SonicFeetPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.2

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicFeetPosTermination(np.float32(self.threshold))
        return TerminationTerm(SonicFeetPosTermination.compute, state)


@kernel_data
class SonicEeBodyPosTermination:
    threshold: np.float32
    low_reference_threshold: np.float32

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicEeBodyPosTermination) -> bool:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        step = m.steps[0]
        threshold = state.low_reference_threshold if m.low_reference[0] else state.threshold
        for i in (3, 6, 10, 13):
            if abs(p[i, 2] - m.clip.tracked_bodies_pos_w[step, i, 2]) > threshold:
                return True
        return False


@configclass(kw_only=True)
class SonicEeBodyPosTerminationCfg(TerminationTermCfg):
    threshold: float = 0.15
    low_reference_threshold: float = 0.75

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        state = SonicEeBodyPosTermination(np.float32(self.threshold), np.float32(self.low_reference_threshold))
        return TerminationTerm(SonicEeBodyPosTermination.compute, state)


@kernel_data
class SonicTrackingReward:
    point_indices: SharedArray
    point_offsets: SharedArray
    std: np.float32
    scratch_ref_point: np.ndarray
    scratch_robot_point: np.ndarray
    scratch_ref_local: np.ndarray
    scratch_robot_local: np.ndarray

    @staticmethod
    @dispatch
    def compute(ctx: ManagerContext, state: SonicTrackingReward) -> float:
        m = ctx.commands["motion"]
        p = ctx.sim["tracked_body_pos"]
        q = ctx.sim["tracked_body_quat"]
        step = m.steps[0]
        anchor = m.reference_index
        ref_anchor_p = m.clip.tracked_bodies_pos_w[step, anchor]
        robot_anchor_p = p[anchor]
        ref_anchor_q = m.clip.tracked_bodies_quat_w[step, anchor]
        robot_anchor_q = q[anchor]
        total = 0.0
        for point_id in range(state.point_indices.shape[0]):
            body = state.point_indices[point_id]
            ref_point = state.scratch_ref_point[ctx.env_id]
            robot_point = state.scratch_robot_point[ctx.env_id]
            rotate_vector(m.clip.tracked_bodies_quat_w[step, body], state.point_offsets[point_id], ref_point)
            rotate_vector(q[body], state.point_offsets[point_id], robot_point)
            ref_point += m.clip.tracked_bodies_pos_w[step, body] - ref_anchor_p
            robot_point += p[body] - robot_anchor_p
            ref_local = state.scratch_ref_local[ctx.env_id]
            robot_local = state.scratch_robot_local[ctx.env_id]
            rotate_inverse(ref_anchor_q, ref_point, ref_local)
            rotate_inverse(robot_anchor_q, robot_point, robot_local)
            dx = ref_local[0] - robot_local[0]
            dy = ref_local[1] - robot_local[1]
            dz = ref_local[2] - robot_local[2]
            total += dx * dx + dy * dy + dz * dz
        mean_error = total / max(state.point_indices.shape[0], 1)
        return math.exp(-mean_error / (state.std * state.std))


@configclass(kw_only=True)
class SonicTrackingRewardCfg(RewardTermCfg):
    std: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("SONIC tracking reward std must be positive and finite")

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        motion = env.cfg.commands.motion
        indices = np.asarray(
            [motion.tracked_body_names.index(name) for name in motion.reward_point_body_names], dtype=np.int64
        )
        offsets = np.asarray(motion.reward_point_body_offsets, dtype=np.float32)
        scratch = np.zeros((env.num_envs, 3), dtype=np.float32)
        state = SonicTrackingReward(
            indices, offsets, np.float32(self.std), scratch.copy(), scratch.copy(), scratch.copy(), scratch.copy()
        )
        return RewardTerm(SonicTrackingReward.compute, state)

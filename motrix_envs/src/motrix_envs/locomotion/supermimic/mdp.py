# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic task terms: motion command, goal observation, clip-end termination.

The task reproduces the SuperMimic G1 whole-body tracking MDP (arXiv
2610.28480, as reference-implemented by UniSuperMimic) on the MotrixSim
manager stack: a goal-conditioned policy tracks the LAFAN1 corpus, episodes
sample their start frame from SuperMimic's return-based one-second-bin
mixture (see :mod:`motrix_envs.locomotion.supermimic.sampling`), and end on
the owning clip's final frame instead of resampling mid-episode.

Upstream mappings kept here:

- Reward: the WBT tracking recipe already matches SuperMimic's six-kernel
  tracking sum (four unit-weight body terms + two half-weight root terms);
  the task cfg assigns the upstream weights. The manager scales rewards by
  ``ctrl_dt``, so the per-step return scale equals upstream's ``* dt``.
- Termination: ``bad_ref_z``/``bad_ref_ori`` are SuperMimic's height/tilt
  checks (upstream thresholds 0.25 / 0.8); ``motion_clip_end`` ends the
  episode on the owning clip's final frame.
- Reset: the WBT teleport reset terms carry SuperMimic's noise magnitudes.
"""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerContext,
    ManagerEnv,
    ObservationTermCfg,
    ObsTerm,
    ResetTerm,
    ResetTermCfg,
    TerminationTerm,
    TerminationTermCfg,
    kernel_data,
    njit,
)
from motrix_env_core.manager.math.quaternion import inverse as quat_inverse
from motrix_env_core.manager.math.quaternion import mul as quat_mul
from motrix_env_core.manager.math.quaternion import rotate_inverse, rotate_vector, to_matrix_first_two_rows
from motrix_env_core.mdp.action import JointPositionActionState
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.numba.kernel_data import Map, SharedArray
from motrix_env_core.numba.manager.commands import ResetContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import JointPositionWrite, JointVelocityWrite
from motrix_envs.locomotion.supermimic.sampling import SuperMimicStartSampler
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand, WbtMotionCommandCfg
from motrix_envs.locomotion.wbt.mdp.observations import RelativePositionParams
from motrix_envs.motion import MotionLibrary

# One-second bins at the 50 Hz control rate, matching upstream BIN_FRAMES = 50.
SUPERMIMIC_BIN_SECONDS = 1.0
# Upstream goal_frames (conf/supermimic/config.yaml): 10 future frames.
SUPERMIMIC_DEFAULT_GOAL_FRAMES = 10
# Upstream gamma for episode returns credited to start bins (Table A.5).
SUPERMIMIC_GAMMA = 0.99
# Simulation-context layout published by reset terms and consumed by the world
# model / critic: robot friction, joint offsets, and torso CoM offset.
SUPERMIMIC_CONTEXT_DIM = 33
SUPERMIMIC_JOINT_OFFSET_START = 1
SUPERMIMIC_COM_OFFSET_START = 30
# Per-step reward ceiling of the SuperMimic reward group: four unit-weight
# tracking kernels plus two half-weighted root kernels (upstream
# MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * dt). The manager applies the dt
# factor, so the sampler receives the dt-free constant here.
SUPERMIMIC_MAX_STEP_REWARD = 5.0


def build_start_bins(frame_clip_end: np.ndarray, bin_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a concatenated motion corpus into per-clip uniform frame bins.

    Args:
        frame_clip_end: Per-frame global index of the owning clip's final
            frame (``WbtMotionClip.frame_clip_end``).
        bin_frames: Bin width in frames (one second at the control rate).

    Returns:
        ``(bin_start, bin_end, frame_bin)`` — inclusive global frame bounds of
        every bin and, per frame, the index of the bin that owns it. The last
        bin of each clip is clipped to the clip's final frame.
    """
    clip_ends = np.unique(frame_clip_end)
    starts: list[int] = []
    ends: list[int] = []
    clip_start = 0
    for clip_end in clip_ends.tolist():
        clip_end = int(clip_end)
        for bin_start in range(clip_start, clip_end + 1, bin_frames):
            starts.append(bin_start)
            ends.append(min(bin_start + bin_frames - 1, clip_end))
        clip_start = clip_end + 1
    bin_start = np.asarray(starts, dtype=np.int64)
    bin_end = np.asarray(ends, dtype=np.int64)
    frame_bin = np.searchsorted(bin_end, np.arange(frame_clip_end.shape[0], dtype=np.int64), side="left")
    return bin_start, bin_end, frame_bin


@njit(inline="always")
def _seed_state_from_frame(
    state: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> None:
    """Seed the published state buffer with a reference frame's state."""
    for axis in range(3):
        state[axis] = root_pos[axis]
        state[7 + axis] = root_lin_vel[axis]
        state[10 + axis] = root_ang_vel[axis]
    state[3] = root_quat[0]
    state[4] = root_quat[1]
    state[5] = root_quat[2]
    state[6] = root_quat[3]
    num_joints = joint_pos.shape[0]
    for joint_id in range(num_joints):
        state[13 + joint_id] = joint_pos[joint_id]
        state[13 + num_joints + joint_id] = joint_vel[joint_id]


@dispatch
def _reset_supermimic_dof_pos(
    ctx: ManagerContext, sim_writes: Map[np.ndarray], params: tuple[np.float32, np.float32]
) -> None:
    """Reset joints and publish the same offsets used by the upstream MDP."""
    position = sim_writes["position"]
    velocity = sim_writes["velocity"]
    motion: SuperMimicMotionCommand = ctx.commands["motion"]
    action: JointPositionActionState = ctx.actions["joint_position"]
    step = motion.steps[0]
    position[:] = motion.clip.joint_pos[step]
    velocity[:] = motion.clip.joint_vel[step]
    noise, enabled = params
    for index in range(position.shape[0]):
        offset = ctx.rand.next_uniform() * noise if enabled > 0.5 else np.float32(0.0)
        motion.context[SUPERMIMIC_JOINT_OFFSET_START + index] = offset
        value = position[index] + offset
        position[index] = min(max(value, action.joint_lower[index]), action.joint_upper[index])
        motion.state[13 + index] = position[index]


@configclass(kw_only=True)
class SuperMimicDofPosResetCfg(ResetTermCfg):
    """Reset joint positions and publish their per-reset offsets in ``xi``."""

    noise: float = 0.01
    noise_scale: float = 1.0
    eval_mode: bool = False

    def __call__(self, ctx) -> ResetTerm:
        joints = ctx.cfg.commands.motion.joint_names
        return ResetTerm(
            _reset_supermimic_dof_pos,
            (np.float32(self.noise * self.noise_scale), np.float32(not self.eval_mode)),
            writes={"position": JointPositionWrite(joints), "velocity": JointVelocityWrite(joints)},
        )


@dispatch
def supermimic_dof_pos_rel_obs(ctx: ManagerContext, out: np.ndarray, params: RelativePositionParams) -> None:
    """Joint positions relative to default pose and the reset joint offsets."""
    dof_pos = ctx.sim["robot_dof_pos"]
    motion: SuperMimicMotionCommand = ctx.commands["motion"]
    np.subtract(dof_pos, params.reference, out)
    for index in range(out.shape[0]):
        out[index] -= motion.context[SUPERMIMIC_JOINT_OFFSET_START + index]
    add_uniform_noise(out, params.noise_amplitude, ctx.rand.state)


@kernel_data
class SuperMimicMotionCommand(WbtMotionCommand):
    """WBT motion command with SuperMimic start-bin sampling and goal frames.

    The re-anchoring ``update`` and the clip-end ``advance`` semantics are
    inherited from :class:`~motrix_envs.locomotion.wbt.mdp.command.WbtMotionCommand`;
    ``hold_at_clip_end`` freezes a lane on its clip's final frame, where the
    ``motion_clip_end`` termination ends the episode. ``reset_env`` replaces
    the corpus-wide adaptive draw with SuperMimic's per-clip one-second-bin
    draw: a bin is selected from the shared ``bin_cdf`` (published host-side
    by :class:`~motrix_envs.locomotion.supermimic.sampling.SuperMimicStartSampler`),
    then a frame uniformly inside the bin.

    Attributes:
        bin_cdf: Cumulative bin-selection probabilities over all corpus bins.
        bin_start: Inclusive first global frame index of every bin.
        bin_end: Inclusive final global frame index of every bin.
        start_bin: Bin index the lane's current episode started from; written
            by ``reset_env`` and re-anchored host-side between episodes.
        reset_counter: Per-lane episode reset count; the host sampler detects
            episode boundaries through changes in this counter.
        num_goal_frames: Number of future reference frames in the goal
            observation (future frames are clamped at the owning clip's end).
        state: Flattened robot state ``(13 + 2N,)`` published every control
            step for the differentiable world model: root position (3), root
            quaternion xyzw (4), root linear/angular velocity (3+3), joint
            positions and velocities (N+N), all in the world frame.
        context: Simulation context ``(33,)`` — friction scale, per-reset
            joint offsets, and torso CoM offset — written by reset terms and
            read by the world model / critic.
        requested_frames: Per-lane frame pin for the next reset (host-set,
            ``-1`` = draw from the bin mixture); consumed by ``reset_env``,
            which draws the owning bin for the pinned frame. Per-clip
            evaluation uses it to start every lane at its clip's first frame.
    """

    bin_cdf: SharedArray = SharedArray(())
    bin_start: SharedArray = SharedArray(())
    bin_end: SharedArray = SharedArray(())
    requested_frames: np.ndarray = np.full((1, 1), -1, dtype=np.int64)
    start_bin: np.ndarray = np.zeros((1, 1), dtype=np.int64)
    reset_counter: np.ndarray = np.zeros((1, 1), dtype=np.int64)
    state: np.ndarray = np.zeros((1, 15), dtype=np.float32)
    endpoint_state: np.ndarray = np.zeros((1, 15), dtype=np.float32)
    context: np.ndarray = np.zeros((1, SUPERMIMIC_CONTEXT_DIM), dtype=np.float32)
    num_goal_frames: np.int64 = np.int64(1)

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        """Re-anchor the reference targets and publish the lane's state.

        The anchoring block mirrors :meth:`WbtMotionCommand.update
        <motrix_envs.locomotion.wbt.mdp.command.WbtMotionCommand.update>`
        (inherited dispatch methods cannot be layered inside the fused
        kernel); the tail additionally writes the flattened robot state the
        differentiable world model consumes.
        """
        # Select the current motion frame for this environment lane.
        step = self.steps[0]
        motion_reference_body_pos_w = self.clip.reference_body_pos_w[step]
        motion_reference_body_quat_w = self.clip.reference_body_quat_w[step]
        motion_tracked_bodies_pos_w = self.clip.tracked_bodies_pos_w[step]
        motion_tracked_bodies_quat_w = self.clip.tracked_bodies_quat_w[step]
        # Anchor the motion to the current simulated pose of the configured
        # reference body, rather than to the clip's absolute world transform.
        reference_index = self.reference_index
        robot_reference_body_pos_w = ctx.sim["tracked_body_pos"][reference_index]
        robot_reference_body_quat_w = ctx.sim["tracked_body_quat"][reference_index]
        out_body_pos = self.target_body_position_relative
        out_body_quat = self.target_body_orientation_relative

        # Keep only the relative yaw between motion and robot. Roll and pitch
        # remain those of each body in the source motion after composition.
        delta_quat = out_body_quat[0]
        quat_inverse(motion_reference_body_quat_w, delta_quat)
        quat_mul(robot_reference_body_quat_w, delta_quat, delta_quat)
        dx, dy, dz, dw = delta_quat
        half_yaw = np.float32(0.5 * math.atan2(2.0 * (dw * dz + dx * dy), 1.0 - 2.0 * (dy * dy + dz * dz)))
        yaw_z = math.sin(half_yaw)
        yaw_w = math.cos(half_yaw)
        height_delta = motion_reference_body_pos_w[2] - robot_reference_body_pos_w[2]

        # Apply the same planar rigid transform to every tracked body, then
        # compose the yaw alignment with its source-motion orientation.
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

        # Publish the flattened post-physics state for the world model.
        state = self.state
        root_lin_vel = ctx.sim["tracked_body_linear_velocity"][reference_index]
        root_ang_vel = ctx.sim["tracked_body_angular_velocity"][reference_index]
        for axis in range(3):
            state[axis] = robot_reference_body_pos_w[axis]
            state[7 + axis] = root_lin_vel[axis]
            state[10 + axis] = root_ang_vel[axis]
        state[3] = robot_reference_body_quat_w[0]
        state[4] = robot_reference_body_quat_w[1]
        state[5] = robot_reference_body_quat_w[2]
        state[6] = robot_reference_body_quat_w[3]
        dof_pos = ctx.sim["robot_dof_pos"]
        dof_vel = ctx.sim["robot_dof_vel"]
        for joint_id in range(dof_pos.shape[0]):
            state[13 + joint_id] = dof_pos[joint_id]
            state[13 + dof_pos.shape[0] + joint_id] = dof_vel[joint_id]
        self.endpoint_state[:] = state

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        """The task env advances the reference before evaluating the endpoint."""
        if self.sequential_clips:
            clip_end = self.clip.frame_clip_end[self.steps[0]]
            step = self.steps[0] + 1
            self.clip_ended[0] = step > clip_end
            if step < self.clip.joint_pos.shape[0]:
                self.steps[0] = step
                if self.clip_ended[0]:
                    ctx.sim_reset_requested[0] = True
            elif self.hold_at_clip_end:
                self.steps[0] = self.clip.joint_pos.shape[0] - 1
            else:
                self.steps[0] = 0
                ctx.sim_reset_requested[0] = True
        else:
            self.clip_ended[0] = self.steps[0] >= self.clip.frame_clip_end[self.steps[0]]

    def reset(self, ctx: ResetContext) -> None:
        """Count episode resets for the host-side return sampler.

        SuperMimic credits episode returns at reset time on the host; the
        WBT adaptive-sampling statistics are not used (the bin mixture
        replaces them), so this hook only advances the reset counter.
        """
        self.reset_counter[ctx.env_ids] += 1

    def on_transition(self) -> None:
        """Nothing to fold host-side: bin tables are static and the mixture is
        republished by the env-owned return sampler after each step."""

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        """Draw the lane's episode start from the SuperMimic bin mixture.

        Sequential playback (play mode) keeps the current frame, matching the
        inherited WBT contract. Every path also seeds the published state
        buffer with the teleport target: the fused evaluate kernel is the only
        other writer, so without this seed the first transition after
        construction would read a zero-initialized state.
        """
        if self.sequential_clips:
            _seed_state_from_frame(
                self.state,
                self.clip.root_body_pos_w[self.steps[0]],
                self.clip.root_body_quat_w[self.steps[0]],
                self.clip.root_body_lin_vel_w[self.steps[0]],
                self.clip.root_body_ang_vel_w[self.steps[0]],
                self.clip.joint_pos[self.steps[0]],
                self.clip.joint_vel[self.steps[0]],
            )
            return
        requested = int(self.requested_frames[0])
        if requested >= 0:
            # Host-pinned frame (per-clip evaluation): adopt it and record
            # the owning bin so host-side bookkeeping stays consistent.
            self.steps[0] = requested
            bin_id = np.int64(0)
            while self.bin_end[bin_id] < requested:
                bin_id += 1
            self.start_bin[0] = bin_id
            self.requested_frames[0] = np.int64(-1)
            _seed_state_from_frame(
                self.state,
                self.clip.root_body_pos_w[self.steps[0]],
                self.clip.root_body_quat_w[self.steps[0]],
                self.clip.root_body_lin_vel_w[self.steps[0]],
                self.clip.root_body_ang_vel_w[self.steps[0]],
                self.clip.joint_pos[self.steps[0]],
                self.clip.joint_vel[self.steps[0]],
            )
            return
        unit = (ctx.rand.next_uniform() + np.float32(1.0)) * np.float32(0.5)
        bin_id = np.int64(0)
        while bin_id + 1 < self.bin_cdf.size and unit > self.bin_cdf[bin_id]:
            bin_id += 1
        span = self.bin_end[bin_id] - self.bin_start[bin_id] + np.int64(1)
        phase = (ctx.rand.next_uniform() + np.float32(1.0)) * np.float32(0.5)
        offset = min(np.int64(phase * np.float32(span)), span - np.int64(1))
        self.steps[0] = self.bin_start[bin_id] + offset
        self.start_bin[0] = bin_id
        _seed_state_from_frame(
            self.state,
            self.clip.root_body_pos_w[self.steps[0]],
            self.clip.root_body_quat_w[self.steps[0]],
            self.clip.root_body_lin_vel_w[self.steps[0]],
            self.clip.root_body_ang_vel_w[self.steps[0]],
            self.clip.joint_pos[self.steps[0]],
            self.clip.joint_vel[self.steps[0]],
        )


@configclass(kw_only=True)
class SuperMimicMotionCommandCfg(WbtMotionCommandCfg):
    """Configuration for :class:`SuperMimicMotionCommand`.

    The inherited adaptive-sampling parameters are inert: SuperMimic's
    return-based bin mixture replaces the failure-rate curriculum, and the
    host-side sampler is owned by the task env. ``adaptive_sampling_enabled``
    must stay False so the inherited hooks stay out of the way.
    """

    num_goal_frames: int = SUPERMIMIC_DEFAULT_GOAL_FRAMES
    supermimic_gamma: float = SUPERMIMIC_GAMMA
    # Upstream return EMA: bin_G <- 0.8 * bin_G + 0.2 * credited (App. D.1).
    supermimic_return_ema: float = 0.2
    supermimic_sampling: bool = True
    # The return-based bin mixture replaces the failure-rate curriculum.
    adaptive_sampling_enabled: bool = False

    def __call__(self, env: ManagerEnv) -> SuperMimicMotionCommand:
        robot = env.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}")
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("WBT motion joint_names must be non-empty and unique.")
        body_joint_names = env.model.bodies["robot"].joint_names
        if body_joint_names != tuple(self.joint_names):
            raise ValueError("WBT robot body joint order must match commands.motion.joint_names order.")
        if robot.resolved_base_link_name not in self.tracked_body_names:
            raise ValueError(f"tracked_body_names must include the robot base link {robot.resolved_base_link_name!r}")
        try:
            reference_index = self.tracked_body_names.index(self.reference_body_name)
        except ValueError:
            raise ValueError(
                f"tracked_body_names must include the reference body {self.reference_body_name!r}"
            ) from None
        if self.adaptive_sampling_enabled:
            raise ValueError(
                "SuperMimic sampling replaces the WBT adaptive sampler; keep adaptive_sampling_enabled=False."
            )
        env_fps = max(int(round(1.0 / env.cfg.ctrl_dt)), 1)
        clip = MotionLibrary(
            self.motion_files,
            joint_names=list(self.joint_names),
            tracked_body_names=self.tracked_body_names,
            reference_body_name=self.reference_body_name,
            root_body_name=robot.base_link_name,
            fps=env_fps,
            extension_channels=self.extension_channels,
        ).assemble()
        bin_frames = max(int(round(SUPERMIMIC_BIN_SECONDS * env_fps)), 1)
        bin_start, bin_end, _ = build_start_bins(clip.frame_clip_end, bin_frames)
        uniform_cdf = np.cumsum(np.full(bin_start.shape[0], 1.0 / bin_start.shape[0], dtype=np.float32))
        uniform_cdf[-1] = 1.0
        if self.supermimic_sampling:
            env._pending_supermimic_sampler = SuperMimicStartSampler(
                num_envs=env.num_envs,
                bin_start=bin_start,
                bin_end=bin_end,
                gamma=self.supermimic_gamma,
                return_ema=self.supermimic_return_ema,
                ctrl_dt=env.cfg.ctrl_dt,
                max_step_reward=SUPERMIMIC_MAX_STEP_REWARD,
            )
        tracked_shape = (len(self.tracked_body_names), 3)
        return SuperMimicMotionCommand(
            clip=clip,
            reference_index=np.int64(reference_index),
            steps=np.zeros((env.num_envs, 1), dtype=np.int64),
            clip_ended=np.zeros((env.num_envs, 1), dtype=bool),
            command=np.empty((env.num_envs, 2 * env.num_actuators), dtype=np.float32),
            target_body_position_relative=np.empty((env.num_envs, *tracked_shape), dtype=np.float32),
            target_body_orientation_relative=np.empty((env.num_envs, tracked_shape[0], 4), dtype=np.float32),
            adaptive_bin_failed_count=np.zeros((0,), dtype=np.float32),
            adaptive_current_bin_failed_count=np.zeros((0,), dtype=np.float32),
            sampling_cdf=np.empty((0,), dtype=np.float32),
            start_at_timestep_zero_prob=np.float32(self.start_at_timestep_zero_prob),
            hold_at_clip_end=self.hold_at_clip_end,
            sequential_clips=self.sequential_clips,
            uniform_ratio=np.float32(self.uniform_ratio),
            alpha=np.float32(self.alpha),
            kernel_size=np.int64(self.kernel_size),
            kernel_lambda=np.float32(self.kernel_lambda),
            reset_counter=np.zeros((env.num_envs, 1), dtype=np.int64),
            requested_frames=np.full((env.num_envs, 1), -1, dtype=np.int64),
            bin_cdf=uniform_cdf,
            bin_start=bin_start,
            bin_end=bin_end,
            start_bin=np.zeros((env.num_envs, 1), dtype=np.int64),
            state=np.zeros((env.num_envs, 13 + 2 * clip.joint_pos.shape[1]), dtype=np.float32),
            endpoint_state=np.zeros((env.num_envs, 13 + 2 * clip.joint_pos.shape[1]), dtype=np.float32),
            context=np.zeros((env.num_envs, SUPERMIMIC_CONTEXT_DIM), dtype=np.float32),
            num_goal_frames=np.int64(self.num_goal_frames),
        )


@njit(inline="always")
def _write_yaw_quaternion(quaternion: np.ndarray, out: np.ndarray) -> None:
    """Write the yaw-only quaternion (xyzw) of ``quaternion`` into ``out``."""
    x, y, z, w = quaternion
    half_yaw = np.float32(0.5 * math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    out[0] = 0.0
    out[1] = 0.0
    out[2] = math.sin(half_yaw)
    out[3] = math.cos(half_yaw)


@dispatch
def supermimic_goal_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    """Future reference frames in the current reference heading frame.

    Per future frame (clamped at the owning clip's final frame), writes the
    upstream 73-per-frame goal layout for the 29-DoF G1: root position delta
    (3), relative root rotation 6D (6), local root linear/angular velocity
    (3+3), reference joint positions (N) and scaled joint velocities (N),
    all expressed in the yaw frame of the reference root at the current step.
    """
    motion: SuperMimicMotionCommand = ctx.commands["motion"]
    clip = motion.clip
    step = motion.steps[0]
    clip_end = clip.frame_clip_end[step]
    joint_count = clip.joint_pos.shape[1]
    frame_dim = 15 + 2 * joint_count
    heading = np.empty(4, np.float32)
    _write_yaw_quaternion(clip.root_body_quat_w[step], heading)
    relative = np.empty(4, np.float32)
    for frame_id in range(int(motion.num_goal_frames)):
        future = min(step + frame_id + 1, clip_end)
        offset = frame_id * frame_dim
        root_pos = clip.root_body_pos_w[future]
        current_pos = clip.root_body_pos_w[step]
        rotate_inverse(
            heading,
            (
                root_pos[0] - current_pos[0],
                root_pos[1] - current_pos[1],
                root_pos[2] - current_pos[2],
            ),
            out[offset : offset + 3],
        )
        quat_inverse(heading, relative)
        quat_mul(relative, clip.root_body_quat_w[future], relative)
        to_matrix_first_two_rows(relative, out[offset + 3 : offset + 9])
        rotate_inverse(heading, clip.root_body_lin_vel_w[future], out[offset + 9 : offset + 12])
        rotate_inverse(heading, clip.root_body_ang_vel_w[future], out[offset + 12 : offset + 15])
        out[offset + 15 : offset + 15 + joint_count] = clip.joint_pos[future]
        out[offset + 15 + joint_count : offset + frame_dim] = clip.joint_vel[future]
        out[offset + 15 + joint_count : offset + frame_dim] *= np.float32(0.1)
    return None


@configclass(kw_only=True)
class SuperMimicGoalObsCfg(ObservationTermCfg):
    """Goal-conditioned future-reference observation.

    The width derives from the motion command's goal-frame count and joint
    count; the reference itself is exact, so the term adds no noise.
    """

    def __call__(self, ctx) -> ObsTerm:
        motion = ctx.command_terms["motion"]
        if not isinstance(motion, SuperMimicMotionCommand):
            raise TypeError(
                f"SuperMimic goal observation requires SuperMimicMotionCommand, got {type(motion).__name__}."
            )
        joint_count = motion.clip.joint_pos.shape[1]
        size = int(motion.num_goal_frames) * (15 + 2 * joint_count)
        return ObsTerm(size, supermimic_goal_obs)


@dispatch
def supermimic_tracking_error_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    """Privileged tracking errors for the value group.

    Writes the anchored tracked-body position errors (the same per-step
    re-anchored targets the relative rewards track) followed by the reference
    root position error, all in the world frame.
    """
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: SuperMimicMotionCommand = ctx.commands["motion"]
    num_bodies = tracked_body_pos.shape[0]
    for body_id in range(num_bodies):
        offset = body_id * 3
        target = motion.target_body_position_relative[body_id]
        for axis in range(3):
            out[offset + axis] = target[axis] - tracked_body_pos[body_id, axis]
    root_offset = num_bodies * 3
    reference_pos = motion.clip.reference_body_pos_w[motion.steps[0]]
    robot_pos = tracked_body_pos[motion.reference_index]
    for axis in range(3):
        out[root_offset + axis] = reference_pos[axis] - robot_pos[axis]
    return None


@configclass(kw_only=True)
class SuperMimicTrackingErrorObsCfg(ObservationTermCfg):
    """Anchored body errors (K×3) plus the reference root error (3)."""

    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(3 * num_tracked + 3, supermimic_tracking_error_obs)


@dispatch
def motion_clip_end_termination(ctx: ManagerContext) -> bool:
    """End the episode while the lane's own clip is exhausted.

    Fires on the owning clip's final frame: ``advance`` holds the step index
    there (``hold_at_clip_end``), so the final frame receives exactly one
    reward step before the episode ends. Upstream treats clip end as a
    time-out; the return sampler bootstraps these episodes like truncations.
    """
    motion: SuperMimicMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    return bool(step >= motion.clip.frame_clip_end[step])


@configclass(kw_only=True)
class MotionClipEndTerminationCfg(TerminationTermCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(motion_clip_end_termination)


__all__ = [
    "SUPERMIMIC_BIN_SECONDS",
    "SUPERMIMIC_DEFAULT_GOAL_FRAMES",
    "SUPERMIMIC_GAMMA",
    "SUPERMIMIC_MAX_STEP_REWARD",
    "MotionClipEndTerminationCfg",
    "SuperMimicGoalObsCfg",
    "SuperMimicMotionCommand",
    "SuperMimicMotionCommandCfg",
    "SuperMimicTrackingErrorObsCfg",
    "build_start_bins",
]

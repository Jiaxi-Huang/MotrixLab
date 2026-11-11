# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch contract tests for the WBT goal-window observation terms.

These lock the upstream supermimic goal-window semantics of
``wbt/mdp/observations.py``: stride-offset future windows that clamp at the
current clip's end frame (never bleeding into the next concatenated clip),
the per-frame ``[qj(N), vj(N)]`` block layout, anchor-relative rot6d and
position windows, and the anchor-frame projected gravity direction.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.rand import RandValue, initialize_rand_states
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand
from motrix_envs.locomotion.wbt.mdp.observations import (
    MotionFutureJointObsCfg,
    MotionFutureReferenceOrientationObsCfg,
    MotionFutureReferencePositionObsCfg,
    ProjectedGravityObsCfg,
    motion_future_joint_obs,
    motion_future_reference_orientation_obs,
    motion_future_reference_position_obs,
    projected_gravity_obs,
)
from motrix_envs.motion import WbtMotionClip
from motrix_envs.motion.sampler import AdaptiveTimestepsSampler

_NUM_JOINTS = 2
_NUM_TRACKED = 3
_HALF_SQRT_TWO = math.sqrt(0.5)
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
# Yaw rotations by quarter turns: exact 0/±1 rotation-matrix entries.
_YAW_QUARTER = {
    0: _IDENTITY_QUAT,
    1: (0.0, 0.0, _HALF_SQRT_TWO, _HALF_SQRT_TWO),
    2: (0.0, 0.0, 1.0, 0.0),
    3: (0.0, 0.0, -_HALF_SQRT_TWO, _HALF_SQRT_TWO),
}


def _quat_xyzw(xyzw: tuple[float, float, float, float], num_envs: int, row: int = 0) -> torch.Tensor:
    """Tracked-body quats ``(num_envs, _NUM_TRACKED, 4)`` with one anchor row set."""
    quats = torch.zeros((num_envs, _NUM_TRACKED, 4), dtype=torch.float32)
    quats[:, :, 3] = 1.0
    quats[:, row] = torch.tensor(xyzw, dtype=torch.float32)
    return quats


def _clip(
    *,
    joint_pos: torch.Tensor | None = None,
    joint_vel: torch.Tensor | None = None,
    ref_pos: torch.Tensor | None = None,
    ref_quat: torch.Tensor | None = None,
) -> WbtMotionClip:
    """Two concatenated clips: frames 0..9 (end 9) and 10..19 (end 19)."""
    num_frames = 20
    frame = torch.arange(num_frames, dtype=torch.float32)
    return WbtMotionClip(
        joint_pos=frame[:, None].repeat(1, _NUM_JOINTS) if joint_pos is None else joint_pos,
        joint_vel=(100.0 + frame)[:, None].repeat(1, _NUM_JOINTS) if joint_vel is None else joint_vel,
        tracked_bodies_pos_w=torch.zeros((num_frames, _NUM_TRACKED, 3), dtype=torch.float32),
        tracked_bodies_quat_w=torch.zeros((num_frames, _NUM_TRACKED, 4), dtype=torch.float32),
        tracked_bodies_lin_vel_w=torch.zeros((num_frames, _NUM_TRACKED, 3), dtype=torch.float32),
        tracked_bodies_ang_vel_w=torch.zeros((num_frames, _NUM_TRACKED, 3), dtype=torch.float32),
        root_body_pos_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_quat_w=torch.zeros((num_frames, 4), dtype=torch.float32),
        root_body_lin_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_ang_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        reference_body_pos_w=ref_pos
        if ref_pos is not None
        else torch.stack((frame, -2.0 * frame, 0.25 * frame), dim=-1),
        reference_body_quat_w=ref_quat
        if ref_quat is not None
        else torch.tensor([_IDENTITY_QUAT] * num_frames, dtype=torch.float32),
        frame_clip_end=torch.tensor([9] * 10 + [19] * 10, dtype=torch.int64),
        extensions=Map({}),
    )


def _ctx(clip: WbtMotionClip, steps: torch.Tensor, tracked_quat: torch.Tensor, tracked_pos: torch.Tensor):
    num_envs = steps.shape[0]
    return ManagerContext(
        env_id=torch.arange(num_envs, dtype=torch.int64),
        actions=Map({}),
        commands=Map({"motion": SimpleNamespace(steps=steps, clip=clip, reference_index=0)}),
        metrics=Map({}),
        rand=RandValue(initialize_rand_states(num_envs, 7)),
        sim=Map({"tracked_body_quat": tracked_quat, "tracked_body_pos": tracked_pos}),
        dt=0.02,
        sim_reset_requested=torch.zeros((num_envs, 1), dtype=torch.bool),
        reset_ids=torch.zeros(0, dtype=torch.int64),
        reset_episode=torch.ones(0, dtype=torch.bool),
        terminated=torch.zeros(num_envs, dtype=torch.bool),
        truncated=torch.zeros(num_envs, dtype=torch.bool),
        reward=torch.zeros(num_envs, dtype=torch.float32),
    )


def _command(clip: WbtMotionClip) -> WbtMotionCommand:
    return WbtMotionCommand(
        clip=clip,
        reference_index=0,
        clip_end=torch.full((1, 1), 19, dtype=torch.int64),
        target_body_position_relative=torch.empty((1, _NUM_TRACKED, 3), dtype=torch.float32),
        target_body_orientation_relative=torch.empty((1, _NUM_TRACKED, 4), dtype=torch.float32),
        sampler=AdaptiveTimestepsSampler.from_clip(
            clip, 1, uniform_ratio=0.1, alpha=0.001, kernel_size=1, kernel_lambda=0.8
        ),
        start_at_timestep_zero_prob=0.0,
        hold_at_clip_end=False,
        sequential_clips=False,
        steps=torch.zeros((1, 1), dtype=torch.int64),
        steps_gauge=torch.zeros(1, dtype=torch.float32),
        clip_ended=torch.zeros((1, 1), dtype=torch.bool),
        command=torch.empty((1, 2), dtype=torch.float32),
    )


# -- future joint-target window -------------------------------------------------


def test_future_joint_window_blocks_frames_and_clamps_at_clip_end():
    clip = _clip()
    steps = torch.tensor([[3], [8], [15]], dtype=torch.int64)
    tracked = _quat_xyzw(_IDENTITY_QUAT, 3)
    ctx = _ctx(clip, steps, tracked, torch.zeros((3, _NUM_TRACKED, 3), dtype=torch.float32))
    out = torch.full((3, 2 * 2 * _NUM_JOINTS), float("nan"), dtype=torch.float32)

    motion_future_joint_obs(ctx, out, (0, 5))

    expected_frames = [(3, 8), (8, 9), (15, 19)]  # lane 1 clamps 13 -> 9 (clip A's end)
    for lane, (first, second) in enumerate(expected_frames):
        # Per-frame block layout: [qj(N), vj(N)] per window frame.
        torch.testing.assert_close(out[lane, 0:2], clip.joint_pos[first])
        torch.testing.assert_close(out[lane, 2:4], clip.joint_vel[first])
        torch.testing.assert_close(out[lane, 4:6], clip.joint_pos[second])
        torch.testing.assert_close(out[lane, 6:8], clip.joint_vel[second])
    # The clamp keeps the window inside clip A: frame 9's content, not clip B's frame 13.
    assert not torch.equal(out[1, 4:6], clip.joint_pos[13])


def test_future_joint_cfg_derives_upstream_offsets_and_width():
    command = _command(_clip())
    build_ctx = SimpleNamespace(command_terms={"motion": command})

    default_term = MotionFutureJointObsCfg().__call__(build_ctx)
    assert default_term.size == 10 * 2 * _NUM_JOINTS
    assert default_term.dispatch is motion_future_joint_obs
    assert default_term.args[0] == tuple(range(0, 50, 5))

    custom_term = MotionFutureJointObsCfg(window=3, stride=2).__call__(build_ctx)
    assert custom_term.size == 3 * 2 * _NUM_JOINTS
    assert custom_term.args[0] == (0, 2, 4)


def test_future_joint_cfg_rejects_non_wbt_command():
    build_ctx = SimpleNamespace(command_terms={"motion": object()})
    with pytest.raises(TypeError, match="WbtMotionCommand"):
        MotionFutureJointObsCfg().__call__(build_ctx)


# -- anchor-relative reference windows ------------------------------------------


def test_future_reference_orientation_window_is_anchor_relative_rot6d():
    ref_quat = torch.tensor([_YAW_QUARTER[1]] * 20, dtype=torch.float32)
    clip = _clip(ref_quat=ref_quat)
    steps = torch.tensor([[0], [1]], dtype=torch.int64)
    # Lane 0: identity anchor sees the +90° yaw target. Lane 1: anchor already
    # yawed +90°, so the relative orientation is identity.
    tracked = torch.zeros((2, _NUM_TRACKED, 4), dtype=torch.float32)
    tracked[:, :, 3] = 1.0
    tracked[1, 0] = torch.tensor(_YAW_QUARTER[1], dtype=torch.float32)
    ctx = _ctx(clip, steps, tracked, torch.zeros((2, _NUM_TRACKED, 3), dtype=torch.float32))
    out = torch.full((2, 2 * 6), float("nan"), dtype=torch.float32)

    motion_future_reference_orientation_obs(ctx, out, (0, 1))

    yaw_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    torch.testing.assert_close(out[0], yaw_90.repeat(2))
    torch.testing.assert_close(out[1], identity.repeat(2))


def test_future_reference_position_window_is_anchor_frame_and_clamps():
    clip = _clip()
    steps = torch.tensor([[3], [8]], dtype=torch.int64)
    # Lane 0: identity anchor at the origin sees world positions directly.
    # Lane 1: anchor yawed +90° at (1, 1, 0); rotate_inverse maps (a, b, c) -> (b, -a, c).
    tracked_pos = torch.zeros((2, _NUM_TRACKED, 3), dtype=torch.float32)
    tracked_pos[1, 0] = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    tracked_quat = _quat_xyzw(_IDENTITY_QUAT, 2)
    tracked_quat[1, 0] = torch.tensor(_YAW_QUARTER[1], dtype=torch.float32)
    ctx = _ctx(clip, steps, tracked_quat, tracked_pos)
    out = torch.full((2, 2 * 3), float("nan"), dtype=torch.float32)

    motion_future_reference_position_obs(ctx, out, (0, 5))

    torch.testing.assert_close(out[0], clip.reference_body_pos_w[[3, 8]].reshape(-1))
    anchor = tracked_pos[1, 0]
    rel_world = clip.reference_body_pos_w[[8, 9]] - anchor  # 13 clamps to clip A's end 9
    expected = torch.stack((rel_world[:, 1], -rel_world[:, 0], rel_world[:, 2]), dim=-1).reshape(-1)
    torch.testing.assert_close(out[1], expected)


def test_future_reference_cfg_sizes_and_offsets():
    orientation = MotionFutureReferenceOrientationObsCfg(window=2, stride=1).__call__(SimpleNamespace())
    position = MotionFutureReferencePositionObsCfg(window=2, stride=1).__call__(SimpleNamespace())
    assert orientation.size == 2 * 6
    assert orientation.dispatch is motion_future_reference_orientation_obs
    assert orientation.args[0] == (0, 1)
    assert position.size == 2 * 3
    assert position.dispatch is motion_future_reference_position_obs
    assert position.args[0] == (0, 1)


# -- projected gravity -----------------------------------------------------------


def test_projected_gravity_is_expressed_in_the_anchor_frame():
    steps = torch.tensor([[0], [0], [0]], dtype=torch.int64)
    tracked = torch.zeros((3, _NUM_TRACKED, 4), dtype=torch.float32)
    tracked[:, :, 3] = 1.0
    tracked[1, 0] = torch.tensor(_YAW_QUARTER[1], dtype=torch.float32)  # yaw leaves gravity alone
    tracked[2, 0] = torch.tensor((_HALF_SQRT_TWO, 0.0, 0.0, _HALF_SQRT_TWO), dtype=torch.float32)  # +90° roll
    ctx = _ctx(_clip(), steps, tracked, torch.zeros((3, _NUM_TRACKED, 3), dtype=torch.float32))
    out = torch.full((3, 3), float("nan"), dtype=torch.float32)

    projected_gravity_obs(ctx, out, 0.0)

    torch.testing.assert_close(out[0], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(out[1], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(out[2], torch.tensor([0.0, -1.0, 0.0]))


def test_projected_gravity_cfg_width_and_default_noise():
    term = ProjectedGravityObsCfg().__call__(SimpleNamespace())
    assert term.size == 3
    assert term.dispatch is projected_gravity_obs
    assert term.args[0] == 0.05

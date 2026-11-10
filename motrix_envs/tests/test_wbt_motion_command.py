# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch contract tests for the WBT motion command state machine.

These protect the behavioral contracts of ``WbtMotionCommand``: start-frame
sampling delegated to the nested sampler (uniform mapping, adaptive-CDF
search-left binning, clamping, start-at-zero draws, fixed per-mode draw
counts), the ``advance`` wrap/hold/sequential-clip branches, and the reset
pass's curriculum fold plus reset-lane-only RNG consumption.
"""

import dataclasses

import pytest
import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.rand import RandValue, draw_uniform, initialize_rand_states
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand
from motrix_envs.motion import WbtMotionClip
from motrix_envs.motion.sampler import (
    AdaptiveTimestepsSampler,
    UniformTimestepsSampler,
)


def _clip(num_frames: int) -> WbtMotionClip:
    """Minimal numeric clip corpus: only the frame axis and clip ends matter."""
    return WbtMotionClip(
        joint_pos=torch.zeros((num_frames, 1), dtype=torch.float32),
        joint_vel=torch.zeros((num_frames, 1), dtype=torch.float32),
        tracked_bodies_pos_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        tracked_bodies_quat_w=torch.zeros((num_frames, 1, 4), dtype=torch.float32),
        tracked_bodies_lin_vel_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        tracked_bodies_ang_vel_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        root_body_pos_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_quat_w=torch.zeros((num_frames, 4), dtype=torch.float32),
        root_body_lin_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_ang_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        reference_body_pos_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        reference_body_quat_w=torch.zeros((num_frames, 4), dtype=torch.float32),
        frame_clip_end=torch.full((num_frames,), num_frames - 1, dtype=torch.int64),
        extensions=Map({}),
    )


def _adaptive_sampler(num_frames: int, env_fps: int = 1) -> AdaptiveTimestepsSampler:
    return AdaptiveTimestepsSampler.from_clip(
        _clip(num_frames),
        env_fps,
        uniform_ratio=0.1,
        alpha=0.001,
        kernel_size=1,
        kernel_lambda=0.8,
    )


def _global_axis_sample_reference(
    draws: torch.Tensor,
    sampling_cdf: torch.Tensor,
    num_frames: int,
    start_zero_prob: float,
) -> torch.Tensor:
    """The former global-axis sampling math, kept verbatim as the regression
    reference the per-clip sampler must reproduce on a single-clip corpus."""
    unit = (draws[:, :1] + 1.0) * 0.5
    bin_id = (sampling_cdf[None, :] < unit).sum(dim=1, dtype=torch.int64)
    bin_unit = (draws[:, 1:2] + 1.0) * 0.5
    phase = (bin_id.to(torch.float32) + bin_unit[:, 0]) / sampling_cdf.numel()
    step_value = phase * num_frames
    step_value = torch.clamp(step_value, min=0.0, max=float(num_frames - 2))
    steps = step_value.to(torch.int64)
    zero_unit = (draws[:, -1:] + 1.0) * 0.5
    return torch.where(zero_unit[:, 0] < start_zero_prob, torch.zeros_like(steps), steps)


def _command(
    num_envs: int,
    num_frames: int,
    *,
    sampler=None,
    hold_at_clip_end: bool = False,
    sequential_clips: bool = False,
    start_zero_prob: float = 0.0,
    steps: torch.Tensor | None = None,
    seed: int = 7,
) -> tuple[WbtMotionCommand, torch.Tensor]:
    state = initialize_rand_states(num_envs, seed)
    command = WbtMotionCommand(
        clip=_clip(num_frames),
        reference_index=0,
        target_body_position_relative=torch.empty((num_envs, 1, 3), dtype=torch.float32),
        target_body_orientation_relative=torch.empty((num_envs, 1, 4), dtype=torch.float32),
        sampler=sampler if sampler is not None else _adaptive_sampler(num_frames),
        start_at_timestep_zero_prob=start_zero_prob,
        hold_at_clip_end=hold_at_clip_end,
        sequential_clips=sequential_clips,
        steps=steps if steps is not None else torch.zeros((num_envs, 1), dtype=torch.int64),
        clip_ended=torch.zeros((num_envs, 1), dtype=torch.bool),
        command=torch.empty((num_envs, 2), dtype=torch.float32),
    )
    return command, state


def _ctx(
    state: torch.Tensor,
    *,
    reset_ids: torch.Tensor | None = None,
    reset_episode: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
) -> ManagerContext:
    num_envs = state.shape[0]
    return ManagerContext(
        env_id=torch.arange(num_envs, dtype=torch.int64),
        actions=Map({}),
        commands=Map({}),
        metrics=Map({}),
        rand=RandValue(state),
        sim=Map({}),
        dt=0.02,
        sim_reset_requested=torch.zeros((num_envs, 1), dtype=torch.bool),
        reset_ids=torch.zeros(0, dtype=torch.int64) if reset_ids is None else reset_ids,
        # Compact reset-pass flags: one entry per lane in ``reset_ids``.
        reset_episode=torch.ones(0, dtype=torch.bool) if reset_episode is None else reset_episode,
        terminated=torch.zeros(num_envs, dtype=torch.bool) if terminated is None else terminated,
        reward=torch.zeros(num_envs, dtype=torch.float32),
    )


# -- sampler.sample ------------------------------------------------------------


def test_uniform_sampler_maps_clamps_and_zero_draws():
    sampler = UniformTimestepsSampler(num_frames=10)
    # unit = (draw + 1) / 2 in [0, 1); uniform step = unit * (T - 1), clamped
    # to [0, T - 2].
    draws = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [0.9998, 0.0]])
    steps = sampler.sample(draws, torch.tensor(0.0))
    torch.testing.assert_close(steps, torch.tensor([0, 4, 8], dtype=torch.int64))
    assert sampler.draw_count == 2


def test_adaptive_sampler_search_left_and_within_bin_phase():
    # 100 frames at env_fps=33 gives 100 // 33 + 1 = 4 bins.
    sampler = _adaptive_sampler(100, env_fps=33)
    sampler.sampling_cdf.copy_(torch.tensor([0.25, 0.5, 0.75, 1.0]))
    # unit == 0.5 lands in bin 1 (search-left: equality selects the equal
    # bin); unit == 0.25 lands in bin 0. Within-bin phase comes from the
    # second draw: step = (bin + bin_unit) / 4 * 100, bin_unit = 0.5 here.
    draws = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    steps = sampler.sample(draws, torch.tensor(0.0))
    torch.testing.assert_close(steps, torch.tensor([37, 12], dtype=torch.int64))
    assert sampler.draw_count == 3


def test_adaptive_sampler_matches_global_axis_math_on_single_clip():
    """Per-clip binning reproduces the former global-axis math bit for bit."""
    num_frames, env_fps = 373, 20  # 373 // 20 + 1 = 19 bins
    sampler = _adaptive_sampler(num_frames, env_fps)
    assert sampler.failed_count.numel() == num_frames // env_fps + 1
    generator = torch.Generator().manual_seed(3)
    draws = torch.rand((64, 3), generator=generator) * 2.0 - 1.0
    for zero_prob in (0.0, 0.3, 1.0):
        reference = _global_axis_sample_reference(draws, sampler.sampling_cdf, num_frames, zero_prob)
        torch.testing.assert_close(sampler.sample(draws, zero_prob), reference)


def test_sample_start_at_zero():
    sampler = UniformTimestepsSampler(num_frames=10)
    draws = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9]])
    # zero_unit = (draw + 1) / 2; prob 1.0 sends every lane to frame zero,
    # prob 0.0 sends none (zero_unit is always in [0, 1)).
    steps = sampler.sample(draws, torch.tensor(1.0))
    torch.testing.assert_close(steps, torch.tensor([0, 0], dtype=torch.int64))
    steps = sampler.sample(draws, torch.tensor(0.0))
    torch.testing.assert_close(steps, torch.tensor([4, 4], dtype=torch.int64))


# -- advance ------------------------------------------------------------------


def test_advance_wraps_ended_lanes_with_sampled_frames():
    command, state = _command(2, 5, steps=torch.tensor([[0], [4]]))
    ctx = _ctx(state)
    draws, _ = draw_uniform(state.clone(), command.sampler.draw_count)
    expected_wrap = command.sampler.sample(draws, torch.tensor(command.start_at_timestep_zero_prob))
    command.advance(ctx)
    assert command.steps[0, 0].item() == 1
    assert command.steps[1, 0].item() == expected_wrap[1].item()
    assert not command.clip_ended[0, 0].item()
    assert command.clip_ended[1, 0].item()
    assert not ctx.sim_reset_requested[0, 0].item()
    assert ctx.sim_reset_requested[1, 0].item()


def test_advance_wrap_consumes_only_ended_lane_rng():
    command, state = _command(2, 5, steps=torch.tensor([[0], [4]]))
    ctx = _ctx(state)
    before = state.clone()
    _, next_state = draw_uniform(state.clone(), command.sampler.draw_count)
    command.advance(ctx)
    torch.testing.assert_close(ctx.rand.state[0], before[0])
    torch.testing.assert_close(ctx.rand.state[1], next_state[1])


def test_advance_hold_at_clip_end_freezes_lane():
    command, state = _command(2, 5, hold_at_clip_end=True, steps=torch.tensor([[2], [4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([3, 4]))
    assert not command.clip_ended[0, 0].item()
    assert command.clip_ended[1, 0].item()
    assert not ctx.sim_reset_requested.any()


def test_advance_sequential_wraps_corpus_end():
    command, state = _command(2, 5, sequential_clips=True, steps=torch.tensor([[3], [4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([4, 0]))
    assert not ctx.sim_reset_requested[0, 0].item()
    assert ctx.sim_reset_requested[1, 0].item()


def test_advance_sequential_hold_at_corpus_end():
    command, state = _command(1, 5, sequential_clips=True, hold_at_clip_end=True, steps=torch.tensor([[4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([4]))
    assert command.clip_ended[0, 0].item()
    assert not ctx.sim_reset_requested.any()


def test_advance_folds_failure_ema_every_step():
    command, state = _command(1, 5)
    ctx = _ctx(state)
    sampler = command.sampler
    sampler.current_failed[0] = 1.0
    command.advance(ctx)
    assert sampler.failed_count[0].item() == pytest.approx(0.001, rel=1e-6)
    assert sampler.current_failed.sum().item() == 0.0
    # Without new failures the next step decays the EMA toward zero.
    command.advance(ctx)
    assert sampler.failed_count[0].item() == pytest.approx(0.001 * 0.999, rel=1e-6)


# -- reset --------------------------------------------------------------------


def test_reset_sequential_keeps_current_frames():
    steps = torch.tensor([[2], [3]])
    command, state = _command(2, 5, sequential_clips=True, steps=steps)
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 1]),
        reset_episode=torch.tensor([True, True]),
    )
    command.reset(ctx)
    torch.testing.assert_close(command.steps, steps)


def test_reset_rebuilds_adaptive_cdf_from_failures():
    # Lane 0 terminated inside bin 1 (steps=4 of 8 frames, 2 bins at
    # env_fps=5); lane 1 reset without terminating contributes no mass.
    command, state = _command(2, 8, sampler=_adaptive_sampler(8, env_fps=5), steps=torch.tensor([[4], [0]]))
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 1]),
        reset_episode=torch.tensor([True, True]),
        terminated=torch.tensor([True, False]),
    )
    command.reset(ctx)
    sampler = command.sampler
    assert sampler.current_failed.tolist() == [0.0, 1.0]
    cdf = sampler.sampling_cdf
    assert cdf[-1].item() == 1.0
    assert torch.all(cdf[:-1] <= cdf[1:])
    # With zero failure history and uniform_ratio mass spread evenly, the
    # rebuilt distribution stays uniform over the reachable bins.
    torch.testing.assert_close(cdf, torch.tensor([0.5, 1.0]))


def test_reset_consumes_only_reset_lane_rng():
    command, state = _command(4, 5, steps=torch.tensor([[0], [1], [2], [3]]))
    before = state.clone()
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([1, 3]),
        reset_episode=torch.tensor([True, True]),
    )
    _, next_state = draw_uniform(state.clone(), command.sampler.draw_count)
    command.reset(ctx)
    torch.testing.assert_close(ctx.rand.state[0], before[0])
    torch.testing.assert_close(ctx.rand.state[2], before[2])
    torch.testing.assert_close(ctx.rand.state[1], next_state[1])
    torch.testing.assert_close(ctx.rand.state[3], next_state[3])


def test_reset_resamples_steps_in_valid_range():
    command, state = _command(3, 6, start_zero_prob=0.5, steps=torch.tensor([[5], [5], [5]]))
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 2]),
        reset_episode=torch.tensor([True, True]),
    )
    command.reset(ctx)
    steps = command.steps[:, 0]
    assert steps[1].item() == 5
    for lane in (0, 2):
        assert 0 <= steps[lane].item() <= 4


# -- multi-clip per-clip binning ------------------------------------------------


def _multi_clip(clip_lengths: tuple[int, ...]) -> WbtMotionClip:
    """Corpus of concatenated clips: per-frame boundaries at each clip's end."""
    frame_clip_end = torch.cat(
        [
            torch.full((length,), sum(clip_lengths[:index]) + length - 1, dtype=torch.int64)
            for index, length in enumerate(clip_lengths)
        ]
    )
    clip = _clip(sum(clip_lengths))
    return dataclasses.replace(clip, frame_clip_end=frame_clip_end)


def test_per_clip_bins_never_straddle_clip_boundaries():
    # Two clips of 40 and 13 frames at env_fps=10: 5 + 2 bins, each bin's
    # frames inside exactly one clip.
    sampler = AdaptiveTimestepsSampler.from_clip(
        _multi_clip((40, 13)), 10, uniform_ratio=0.1, alpha=0.001, kernel_size=1, kernel_lambda=0.8
    )
    assert sampler.clip_bins.tolist() == [5, 2]
    assert sampler.bin_clip_offset.tolist() == [0, 5]
    assert sampler.clip_frame_start.tolist() == [0, 40]
    assert sampler.clip_frame_end.tolist() == [39, 52]
    assert sampler.failed_count.numel() == 7

    # A failure at the last frame of clip 0 and the first frame of clip 1
    # lands in their own clips' tail/head bins — never a shared one.
    steps = torch.tensor([39, 40], dtype=torch.int64)
    sampler.fold_episode_stats(
        torch.tensor([0, 1]),
        reset_episode=torch.ones(2, dtype=torch.bool),
        terminated=torch.ones(2, dtype=torch.bool),
        steps=steps,
    )
    # Clip 0: bin 39*5//40 = 4 (flat 4); clip 1: bin (40-40)*2//13 = 0 (flat 5).
    assert sampler.current_failed.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


def test_per_clip_sample_stays_inside_the_drawn_clip():
    sampler = AdaptiveTimestepsSampler.from_clip(
        _multi_clip((40, 13)), 10, uniform_ratio=0.1, alpha=0.001, kernel_size=1, kernel_lambda=0.8
    )
    # Force all sampling mass onto the last bin (clip 1's second bin).
    cdf = torch.zeros(7, dtype=torch.float32)
    cdf[6] = 1.0
    sampler.sampling_cdf.copy_(cdf)
    generator = torch.Generator().manual_seed(11)
    draws = torch.rand((32, 3), generator=generator) * 2.0 - 1.0
    steps = sampler.sample(draws, 0.0)
    assert torch.all(steps >= 46), steps  # inside clip 1, past its first bin
    assert torch.all(steps <= 51), steps  # non-terminal frame of clip 1


def test_uniform_sampler_draw_count_and_metric_gauges_rebuild():
    command, state = _command(2, 8, sampler=UniformTimestepsSampler(num_frames=8), steps=torch.tensor([[4], [0]]))
    assert command.sampler.draw_count == 2
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 1]),
        reset_episode=torch.tensor([True, True]),
        terminated=torch.tensor([True, False]),
    )
    command.reset(ctx)  # uniform sampler: folds and rebuilds are no-ops
    assert command.steps[1, 0].item() in range(0, 7)

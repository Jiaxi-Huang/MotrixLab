# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Quaternion / resampling / velocity helpers shared by the lafan4sonic pathways.

All quaternions are unit ``xyzw``. The velocity constructions mirror the
BONES-SEED converter (upstream gear_sonic ``_compute_velocity`` contract) so
LAFAN corpora stay interchangeable with BONES-SEED corpora.
"""

from __future__ import annotations

import numpy as np

from motrix_env_core.math import quaternion

# Upstream smooths pose-derived body velocities with an edge-replicated
# gaussian filter (sigma = 2 frames) before storing them.
VELOCITY_SMOOTH_SIGMA = 2.0
_VELOCITY_SMOOTH_TRUNCATE = 4.0


def slerp_track(quat: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Slerp a trailing-4-axis xyzw track onto ``target_times``.

    ``quat`` is ``(T, 4)`` or ``(T, K, 4)``; interpolation happens along axis 0
    with per-slot shortest-arc handling, so ``K`` tracks slerp in one call.
    """
    upper = np.clip(np.searchsorted(source_times, target_times, side="right"), 1, len(source_times) - 1)
    lower = upper - 1
    q0 = quat[lower]
    q1 = quat[upper].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    span = (target_times - source_times[lower]) / (source_times[upper] - source_times[lower])
    fraction = span.reshape(span.shape + (1,) * (quat.ndim - 1))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    safe = sin_theta > 1.0e-7
    scale0 = np.where(safe, np.sin((1.0 - fraction) * theta) / np.where(safe, sin_theta, 1.0), 1.0 - fraction)
    scale1 = np.where(safe, np.sin(fraction * theta) / np.where(safe, sin_theta, 1.0), fraction)
    out = scale0 * q0 + scale1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample_track(
    root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray, source_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample (lerp pos/dof, slerp quat) onto the half-open target grid.

    The output grid is ``arange(0, (T-1)/source_fps, 1/target_fps)`` — the same
    grid the BVH pathway uses, so both sides of a clip stay frame-aligned.
    """
    times = np.arange(len(root_pos), dtype=np.float64) / source_fps
    target = np.arange(0.0, times[-1], 1.0 / target_fps, dtype=np.float64)
    pos = np.stack([np.interp(target, times, root_pos[:, axis]) for axis in range(3)], axis=-1)
    quat = slerp_track(root_quat, times, target)
    joints = np.stack([np.interp(target, times, dof[:, axis]) for axis in range(dof.shape[1])], axis=-1)
    return pos, quat, joints


def target_times(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    """The shared resampling grid for one clip (both pathways must agree)."""
    last = (num_frames - 1) / source_fps
    return np.arange(0.0, last, 1.0 / target_fps, dtype=np.float64)


def quat_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw unit quats -> (..., 3) rotation vectors (atan2 form, w >= 0)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz, w = q[..., :3], q[..., 3:4]
    norms = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half = np.arctan2(norms, w)
    axis = np.where(norms > 1.0e-12, xyz / np.where(norms > 1.0e-12, norms, 1.0), 0.0)
    return axis * (2.0 * half)


def axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vectors -> (..., 4) xyzw unit quats."""
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1.0e-8, np.sin(half) / np.where(angle > 1.0e-8, angle, 1.0), 0.5 - angle * angle / 48.0)
    return np.concatenate((axis_angle * scale, np.cos(half)), axis=-1)


def angular_velocity_world(quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity of a (T, ...) xyzw sequence via shortest-arc central diffs."""
    result = np.asarray(quat_xyzw, dtype=np.float64).copy()
    for i in range(1, len(result)):
        flip = np.sum(result[i - 1] * result[i], axis=-1) < 0.0
        result[i] = np.where(flip[..., None], -result[i], result[i])
    omega = np.zeros(result.shape[:-1] + (3,), dtype=np.float64)
    if len(result) > 2:
        q_rel = quaternion.mul(result[2:], quaternion.conjugate(result[:-2]))
        omega[1:-1] = quat_to_rotvec(q_rel) / (2.0 * dt)
        omega[0] = omega[1]
        omega[-1] = omega[-2]
    return omega


def gaussian_smooth1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with edge replication."""
    radius = int(_VELOCITY_SMOOTH_TRUNCATE * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [np.repeat(values[:1], radius, axis=0), values, np.repeat(values[-1:], radius, axis=0)],
        axis=0,
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=0)
    return windows @ kernel


def forward_difference_track(values: np.ndarray, dt: float) -> np.ndarray:
    """Forward differences with the final frame duplicated (SONIC dof-vel contract)."""
    velocity = np.empty(values.shape, dtype=np.float64)
    velocity[:-1] = (values[1:] - values[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def body_velocities_from_poses(
    body_pos_w: np.ndarray, body_quat_w: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame body velocities by differentiating FK poses (sigma-2 smoothing)."""
    lin = np.gradient(np.asarray(body_pos_w, dtype=np.float64), dt, axis=0)
    ang = angular_velocity_world(body_quat_w, dt)
    return gaussian_smooth1d(lin, VELOCITY_SMOOTH_SIGMA), gaussian_smooth1d(ang, VELOCITY_SMOOTH_SIGMA)

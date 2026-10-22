# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched SO(3) helpers for the SuperMimic world model.

Quaternions are xyzw everywhere, matching the environment's published state
layout; states are ``(position, rotation matrix, v, w, q, qd)`` tuples.
"""

from __future__ import annotations

import torch

from .contracts import State

_STATE_HEAD = 13


def quat_xyzw_to_mat(quat: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def mat_to_quat_xyzw(mat: torch.Tensor) -> torch.Tensor:
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack(((m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw), qw), -1),
            torch.stack((qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx), (m21 - m12) / (4 * qx)), -1),
            torch.stack(((m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy), (m02 - m20) / (4 * qy)), -1),
            torch.stack(((m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz, (m10 - m01) / (4 * qz)), -1),
        ),
        dim=-2,
    )
    indices = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(candidates, -2, indices[..., None, None].expand(indices.shape + (1, 4))).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return q * torch.where(q[..., 3:4] < 0, -1.0, 1.0)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(vector.shape[:-1] + (3, 3))


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    """Rotation matrices of rotation vectors (axis * angle)."""
    theta2 = (vector * vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    skew_matrix = _skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(skew_matrix.shape)
    return eye + a[..., None] * skew_matrix + b[..., None] * (skew_matrix @ skew_matrix)


def quat_xyzw_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    q = quat * torch.where(quat[..., 3:4] < 0, -1.0, 1.0)
    vector = q[..., :3]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., 3:4])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(q[..., 3:4], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return vector * scale


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    return quat_xyzw_to_rotvec(mat_to_quat_xyzw(rotation))


def yaw_mat(rotation: torch.Tensor) -> torch.Tensor:
    forward_x, forward_y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(forward_x * forward_x + forward_y * forward_y + 1e-9)
    cos, sin = forward_x / norm, forward_y / norm
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    return torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1).reshape(rotation.shape[:-2] + (3, 3))


def flatten_state(state: State) -> torch.Tensor:
    """``[pos 3 | quat xyzw 4 | v 3 | w 3 | q N | qd N]`` — the env's layout."""
    position, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
    return torch.cat((position, mat_to_quat_xyzw(rotation), lin_vel, ang_vel, joint_pos, joint_vel), -1)


def unflatten_state(value: torch.Tensor) -> State:
    num_joints = (value.shape[-1] - _STATE_HEAD) // 2
    joint_end = _STATE_HEAD + num_joints
    return (
        value[..., 0:3],
        quat_xyzw_to_mat(value[..., 3:7]),
        value[..., 7:10],
        value[..., 10:13],
        value[..., 13:joint_end],
        value[..., joint_end:],
    )


def invariant_features(state: State, mdp) -> torch.Tensor:
    """Yaw-invariant state summary; the world-model input head."""
    _, rotation, lin_vel, ang_vel, joint_pos, joint_vel = state
    heading = yaw_mat(rotation)
    rotation_t = rotation.transpose(-1, -2)
    return torch.cat(
        (
            state[0][..., 2:3],
            _mat6(heading.transpose(-1, -2) @ rotation),
            (rotation_t @ lin_vel.unsqueeze(-1)).squeeze(-1),
            (rotation_t @ ang_vel.unsqueeze(-1)).squeeze(-1),
            joint_pos,
            joint_vel,
        ),
        -1,
    )


def _mat6(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[..., :2, :].reshape(rotation.shape[:-2] + (6,))


__all__ = [
    "flatten_state",
    "invariant_features",
    "mat_to_quat_xyzw",
    "quat_xyzw_to_mat",
    "so3_exp",
    "so3_log",
    "unflatten_state",
    "yaw_mat",
]

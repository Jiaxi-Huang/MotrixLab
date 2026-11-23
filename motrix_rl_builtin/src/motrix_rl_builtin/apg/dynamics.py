# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""GenCoord velocity-delta model and analytic pose integration.

Equations adapted from SuperMimic abc124c, ETH Zurich (MIT).
All quaternions here use MotrixLab's xyzw convention.
"""

import torch
from torch import nn

from motrix_env_core.math import quaternion as quat
from motrix_rl_builtin.apg.networks import Normalizer, mlp, run_mlp


def represent(aux, velocity, angular_velocity, joints, joint_velocity):
    rotation = aux[..., 3:7]
    down = torch.zeros_like(velocity)
    down[..., 2] = -1
    return torch.cat(
        [
            aux[..., 2:3],
            quat.rotate_inverse(rotation, down),
            quat.rotate_inverse(rotation, velocity),
            quat.rotate_inverse(rotation, angular_velocity),
            joints,
            joint_velocity,
        ],
        dim=-1,
    )


def integrate(aux, rep, prediction, dt, substeps):
    joint_count = (rep.shape[-1] - 10) // 2
    v, w = rep[..., 4:7], rep[..., 7:10]
    q, qd = rep[..., 10 : 10 + joint_count], rep[..., 10 + joint_count :]
    dv, dw, dqd = prediction[..., :3], prediction[..., 3:6], prediction[..., 6:]
    coefficient = (substeps + 1) / (2 * substeps)
    rotation = aux[..., 3:7]
    next_p = aux[..., :3] + quat.rotate_vector(rotation, v + coefficient * dv) * dt
    omega = quat.rotate_vector(rotation, w + coefficient * dw) * dt
    angle_sq = omega.square().sum(dim=-1, keepdim=True)
    small = angle_sq < 1e-4
    # Avoid sinc's CUDA runtime compilation. Taylor expansions also retain
    # smooth derivatives at zero; keep the unused division branch finite.
    angle = angle_sq.clamp_min(1e-4).sqrt()
    scale = torch.where(small, 0.5 - angle_sq / 48 + angle_sq.square() / 3840, torch.sin(angle * 0.5) / angle)
    real = torch.where(small, 1 - angle_sq / 8 + angle_sq.square() / 384, torch.cos(angle * 0.5))
    delta = torch.cat([omega * scale, real], dim=-1)
    next_rotation = nn.functional.normalize(quat.mul(delta, rotation), dim=-1)
    next_aux = torch.cat([next_p, next_rotation], dim=-1)
    next_rep = represent(
        next_aux,
        quat.rotate_vector(rotation, v + dv),
        quat.rotate_vector(rotation, w + dw),
        q + (qd + coefficient * dqd) * dt,
        qd + dqd,
    )
    return next_aux, next_rep


def state_loss(aux, rep, target_aux, target_rep, cfg):
    joint_count = (rep.shape[-1] - 10) // 2
    errors = [
        (aux[..., :3] - target_aux[..., :3]) / cfg.root_position_error,
        (quat.to_matrix(aux[..., 3:7]) - quat.to_matrix(target_aux[..., 3:7])).flatten(-2)
        / (2**0.5 * cfg.root_orientation_error),
        (
            quat.rotate_vector(aux[..., 3:7], rep[..., 4:7])
            - quat.rotate_vector(target_aux[..., 3:7], target_rep[..., 4:7])
        )
        / cfg.root_linear_velocity_error,
        (
            quat.rotate_vector(aux[..., 3:7], rep[..., 7:10])
            - quat.rotate_vector(target_aux[..., 3:7], target_rep[..., 7:10])
        )
        / cfg.root_angular_velocity_error,
        (rep[..., 10 : 10 + joint_count] - target_rep[..., 10 : 10 + joint_count]) / cfg.joint_position_error,
        (rep[..., 10 + joint_count :] - target_rep[..., 10 + joint_count :]) / cfg.joint_velocity_error,
    ]
    return torch.cat(errors, dim=-1).square().mean(-1)


class Dynamics(nn.Module):
    def __init__(self, joint_count, hidden_dims, dt, substeps, mixed_precision=False):
        super().__init__()
        self.dt, self.substeps = dt, substeps
        self.mixed_precision = mixed_precision
        self.output_normalizer = Normalizer(6 + joint_count, eps=1e-4)
        self.state_normalizer = Normalizer(10 + 2 * joint_count)
        self.action_normalizer = Normalizer(joint_count)
        self.net = mlp(10 + 3 * joint_count, hidden_dims, 6 + joint_count, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, aux, rep, control):
        prediction = run_mlp(
            self.net,
            torch.cat([self.state_normalizer(rep), self.action_normalizer(control)], dim=-1),
            self.mixed_precision,
        )
        prediction = self.output_normalizer.inverse(prediction)
        return integrate(aux, rep, prediction, self.dt, self.substeps)


def velocity_increment(aux, rep, next_aux, next_rep):
    """Successor velocity increments expressed in the initial body frame."""
    rotation, next_rotation = aux[..., 3:7], next_aux[..., 3:7]
    joint_count = (rep.shape[-1] - 10) // 2
    return torch.cat(
        [
            quat.rotate_inverse(rotation, quat.rotate_vector(next_rotation, next_rep[..., 4:7])) - rep[..., 4:7],
            quat.rotate_inverse(rotation, quat.rotate_vector(next_rotation, next_rep[..., 7:10])) - rep[..., 7:10],
            next_rep[..., 10 + joint_count :] - rep[..., 10 + joint_count :],
        ],
        -1,
    )

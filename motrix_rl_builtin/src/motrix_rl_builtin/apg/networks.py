# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dynamics, actor, and critic networks for the APG world-model chain.

Task-agnostic: dimensions come from the assembly layout and observation
group widths, no layer carries task-specific slicing. The so(3) integrator,
invariant features, error floors, and running norms are recovered from the
prior ``motrix.mbrl`` attempt; the embedded exogenous advance is NOT —
exogenous channels advance by rule in :class:`ModelingProgram
<motrix_env_core.manager.modeling.ModelingProgram>`, and the dynamics
predicts only the physical block.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from motrix_rl_builtin.apg.assemble import matrix_to_quat_xyzw


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    act: type[nn.Module] = nn.ELU,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden:
        layers.append(nn.Linear(last, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(act())
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor, amp_dtype: torch.dtype | None) -> torch.Tensor:
    """MLP forward under optional bf16 autocast, output back in fp32.

    Only the MLP matmuls drop precision (upstream ``run_mlp``): state
    integration, FK, rewards, and collection stay fp32 either way.
    """
    if amp_dtype is None:
        return net(x)
    with torch.autocast(device_type=x.device.type, dtype=amp_dtype):
        return net(x).float()


def quat_xyzw_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions ``(..., 4)`` to rotation matrices ``(..., 3, 3)``."""
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def yaw_mat(rotation: torch.Tensor) -> torch.Tensor:
    """Heading-only rotation of ``(..., 3, 3)`` orientations."""
    forward_x, forward_y = rotation[..., 0, 0], rotation[..., 1, 0]
    norm = torch.sqrt(forward_x * forward_x + forward_y * forward_y + 1e-9)
    cos, sin = forward_x / norm, forward_y / norm
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    return torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1).reshape(rotation.shape[:-2] + (3, 3))


def _quat_wxyz_to_rotvec(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = quat_wxyz * torch.where(quat_wxyz[..., :1] < 0, -1.0, 1.0)
    vector = quat_wxyz[..., 1:]
    sin_half = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, quat_wxyz[..., :1])
    scale = torch.where(
        sin_half < 1e-6,
        2.0 / torch.clamp(quat_wxyz[..., :1], min=1e-6),
        2 * half / (sin_half + 1e-12),
    )
    return vector * scale


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Rotation matrices to rotation vectors via the quaternion branch selection."""
    quat_wxyz = matrix_to_quat_xyzw(rotation)
    quat_wxyz = torch.cat((quat_wxyz[..., 3:], quat_wxyz[..., :3]), -1)
    return _quat_wxyz_to_rotvec(quat_wxyz)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(vector.shape[:-1] + (3, 3))


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    theta2 = (vector * vector).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    skew_matrix = _skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype).expand(skew_matrix.shape)
    return eye + a[..., None] * skew_matrix + b[..., None] * (skew_matrix @ skew_matrix)


def _mat_to_quat_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    quat_xyzw = matrix_to_quat_xyzw(rotation)
    return torch.cat((quat_xyzw[..., 3:], quat_xyzw[..., :3]), -1)


class _RunningNorm(nn.Module):
    """Running mean/variance normalization, frozen once ``max_count`` is seen."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var.copy_(
            (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total) / total
        )
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()


class Dynamics(nn.Module):
    """World-model supervision contract, with flat-state MSE defaults."""

    amp_dtype: torch.dtype | None = None

    def prediction_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).square().mean(dim=-1)

    def reduce_window_loss(self, loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return loss / valid.sum().clamp(min=1.0)

    def loss_metric(self, loss: torch.Tensor, window: int) -> torch.Tensor:
        return loss

    def update_statistics(self, state, action, successor) -> None:
        """Update model-specific statistics from real transitions, when needed."""


class MlpDynamics(Dynamics):
    """Residual flat-state world model: ``next = state + f(state, action)``."""

    def __init__(self, flat_width: int, action_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _mlp(flat_width + action_dim, hidden, flat_width)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = run_mlp(self.net, torch.cat([state, action], dim=-1), self.amp_dtype)
        return state + delta


class FloatingBaseDynamics(Dynamics):
    """Frame-local world model for the physical block of floating-base states.

    Block convention: ``[0:3]`` root position, ``[3:7]`` root quaternion
    (xyzw), ``[7:10]``/``[10:13]`` root linear/angular velocity, then joint
    positions/velocities in tree order. The MLP sees only heading/translation
    -invariant features plus the action and predicts frame-local increments
    (yaw-frame position, so(3) rotation, body-frame velocities, joint deltas)
    that integrate in fp32 back into world coordinates, as the SuperMimic
    recipe. Exogenous channels are NOT part of the block — they advance by
    rule in the modeling program.
    """

    def __init__(self, num_joints: int, action_dim: int, hidden: Sequence[int]):
        super().__init__()
        self.num_joints = num_joints
        joints = 13 + num_joints
        self._pos, self._quat = slice(0, 3), slice(3, 7)
        self._lin, self._ang = slice(7, 10), slice(10, 13)
        self._q, self._qd = slice(13, joints), slice(joints, 13 + 2 * num_joints)
        invariant_dim = 1 + 6 + 3 + 3 + 2 * num_joints
        self.out_dim = 12 + 2 * num_joints
        self.net = _mlp(invariant_dim + action_dim, hidden, self.out_dim, act=nn.SiLU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.in_norm = _RunningNorm(invariant_dim + action_dim)
        self.out_norm = _RunningNorm(self.out_dim, eps=1e-4)
        self.register_buffer(
            "error_floor",
            torch.cat(
                [
                    torch.full((width,), floor)
                    for width, floor in (
                        (3, 0.003),
                        (3, 0.005),
                        (3, 0.02),
                        (3, 0.05),
                        (num_joints, 0.003),
                        (num_joints, 0.05),
                    )
                ]
            ),
            persistent=False,
        )

    def inputs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        position = state[..., self._pos]
        rotation = quat_xyzw_to_mat(state[..., self._quat])
        lin_vel = state[..., self._lin]
        ang_vel = state[..., self._ang]
        joint_pos = state[..., self._q]
        joint_vel = state[..., self._qd]

        rotation_t = rotation.transpose(-1, -2)
        heading = yaw_mat(rotation)
        tilt_6d = (heading.transpose(-1, -2) @ rotation)[..., :, :2].reshape(rotation.shape[:-2] + (6,))
        features = torch.cat(
            (
                position[..., 2:3],
                tilt_6d,
                (rotation_t @ lin_vel.unsqueeze(-1)).squeeze(-1),
                (rotation_t @ ang_vel.unsqueeze(-1)).squeeze(-1),
                joint_pos,
                joint_vel,
            ),
            -1,
        )
        return torch.cat((features, action), -1)

    def targets(self, state: torch.Tensor, successor: torch.Tensor) -> torch.Tensor:
        """Frame-local physical increment."""
        rotation = quat_xyzw_to_mat(state[..., self._quat])
        rotation_t = rotation.transpose(-1, -2)
        heading_t = yaw_mat(rotation).transpose(-1, -2)
        return torch.cat(
            (
                (heading_t @ (successor[..., self._pos] - state[..., self._pos]).unsqueeze(-1)).squeeze(-1),
                so3_log(rotation_t @ quat_xyzw_to_mat(successor[..., self._quat])),
                (rotation_t @ (successor[..., self._lin] - state[..., self._lin]).unsqueeze(-1)).squeeze(-1),
                (rotation_t @ (successor[..., self._ang] - state[..., self._ang]).unsqueeze(-1)).squeeze(-1),
                successor[..., self._q] - state[..., self._q],
                successor[..., self._qd] - state[..., self._qd],
            ),
            -1,
        )

    def reduce_window_loss(self, loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return loss / valid.shape[0]

    def loss_metric(self, loss: torch.Tensor, window: int) -> torch.Tensor:
        return loss / window

    def prediction_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scaled physical error in the target state's local coordinates."""
        scale = torch.maximum(self.out_norm.std(), self.error_floor)
        return (self.targets(target, prediction) / scale).square().mean(-1)

    @torch.no_grad()
    def update_statistics(self, state, action, successor) -> None:
        self.in_norm.update(self.inputs(state, action))
        self.out_norm.update(self.targets(state, successor))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        position = state[..., self._pos]
        rotation = quat_xyzw_to_mat(state[..., self._quat])
        heading = yaw_mat(rotation)
        lin_vel, ang_vel = state[..., self._lin], state[..., self._ang]
        joint_pos, joint_vel = state[..., self._q], state[..., self._qd]
        prediction = run_mlp(self.net, self.in_norm(self.inputs(state, action)), self.amp_dtype)
        increment = prediction * self.out_norm.std() + self.out_norm.mean
        d_position, d_rotation, d_lin_vel, d_ang_vel, d_joint_pos, d_joint_vel = increment.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        next_state = state.clone()
        next_state[..., self._pos] = position + (heading @ d_position.unsqueeze(-1)).squeeze(-1)
        next_quat = _mat_to_quat_wxyz(rotation @ so3_exp(d_rotation))
        next_state[..., self._quat] = torch.cat((next_quat[..., 1:], next_quat[..., :1]), -1)
        next_state[..., self._lin] = lin_vel + (rotation @ d_lin_vel.unsqueeze(-1)).squeeze(-1)
        next_state[..., self._ang] = ang_vel + (rotation @ d_ang_vel.unsqueeze(-1)).squeeze(-1)
        next_state[..., self._q] = joint_pos + d_joint_pos
        next_state[..., self._qd] = joint_vel + d_joint_vel
        return next_state


def _output_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown actor output activation {name!r}")


class MlpActor(nn.Module):
    """Deterministic actor over the flattened normalized observation history.

    The single action-boundary transform (raw output -> command) is applied
    by the caller as ``action_limit * raw``, consistently in collection,
    imagination, and export.
    """

    amp_dtype: torch.dtype | None = None

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (128, 128),
        *,
        output_activation: str = "tanh",
    ):
        super().__init__()
        self.net = _mlp(input_dim, hidden, action_dim)
        self.output_activation = _output_activation(output_activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output_activation(run_mlp(self.net, features, self.amp_dtype))


class MlpCritic(nn.Module):
    """State-value head ``V(obs) -> (..., 1)``."""

    amp_dtype: torch.dtype | None = None

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, obs, self.amp_dtype)

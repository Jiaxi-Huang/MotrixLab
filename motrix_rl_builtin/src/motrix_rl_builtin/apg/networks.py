# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dynamics, actor, and critic networks for the APG world-model chain.

Task-agnostic: dimensions come from the assembly layout and observation
group widths, no layer carries task-specific slicing. The floating-base
dynamics mirrors upstream supermimic's gencoord recipe (invariant rep-state
features in, acceleration-level deltas out, semi-implicit integration,
Bryson-weighted channel-group supervision); the embedded exogenous advance
of the prior ``motrix.mbrl`` attempt is NOT kept — exogenous channels
advance by rule in :class:`ModelingProgram
<motrix_env_core.manager.modeling.ModelingProgram>`, and the dynamics
predicts only the physical block.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


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


def quat_integrate_xyzw(quaternion: torch.Tensor, world_omega: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate xyzw quaternions one step by a world-frame angular velocity.

    ``q <- normalize(q + dt/2 · (0, ω) ⊗ q)`` — the same first-order update
    upstream supermimic's ``quat_integrate`` applies.
    """
    x, y, z, w = quaternion.unbind(-1)
    ox, oy, oz = world_omega.unbind(-1)
    half = 0.5 * dt
    return torch.nn.functional.normalize(
        torch.stack(
            (
                x + half * (ox * w + oy * z - oz * y),
                y + half * (oy * w + oz * x - ox * z),
                z + half * (oz * w + ox * y - oy * x),
                w - half * (ox * x + oy * y + oz * z),
            ),
            dim=-1,
        ),
        dim=-1,
    )


class FloatingBaseDynamics(Dynamics):
    """Acceleration-delta world model for the physical block (upstream gencoord).

    Block convention: ``[0:3]`` root position, ``[3:7]`` root quaternion
    (xyzw), ``[7:10]``/``[10:13]`` root linear/angular velocity (world
    frame), then joint positions/velocities in tree order. Mirrors upstream
    supermimic's ``GenCoordRepresentation``: the MLP consumes the invariant
    rep-state features ``[z, g_b, v_b, w_b, qj, vj]`` plus the action and
    predicts acceleration-level deltas ``[Δv_b, Δw_b, Δvj]`` in the current
    root body frame; poses integrate analytically with the semi-implicit
    substep coefficient ``c = (decimation + 1) / (2 · decimation)`` —
    velocity states advance by the full delta, pose updates weight it by
    ``c`` to match how the sim composes its substeps. The supervision error
    is Bryson-weighted per channel group of the integrated state. Exogenous
    channels are NOT part of the block — they advance by rule in the
    modeling program.
    """

    def __init__(
        self,
        num_joints: int,
        action_dim: int,
        hidden: Sequence[int],
        dt: float,
        decimation: int,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.dt = dt
        joints = 13 + num_joints
        self._pos, self._quat = slice(0, 3), slice(3, 7)
        self._lin, self._ang = slice(7, 10), slice(10, 13)
        self._q, self._qd = slice(13, joints), slice(joints, 13 + 2 * num_joints)
        rep_dim = 10 + 2 * num_joints
        self.out_dim = 6 + num_joints
        self.net = _mlp(rep_dim + action_dim, hidden, self.out_dim, act=nn.GELU, layer_norm=True)
        output = self.net[-1]
        assert isinstance(output, nn.Linear)
        # Zero acceleration at init: rollouts start as constant-velocity
        # continuation. Kept as an evidence-backed deviation from upstream's
        # default init: with no WM-only warmup (upstream-faithful) the
        # default-initialized rollout diverges on the first update at full
        # width — policy_grad_norm NaN at iteration 1 (2026-11-11 GPU run),
        # NaN weights then panic the sim backend. The zero output keeps the
        # first unroll physically sane; its implicit short WM warmup is the
        # price.
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.state_norm = _RunningNorm(rep_dim)
        self.action_norm = _RunningNorm(action_dim)
        self.register_buffer("substep_coeff", torch.tensor((decimation + 1) / (2.0 * decimation)))
        # Bryson's rule: each channel group weights 1 / nominal_error^2 on
        # upstream's gencoord nominal errors.
        self.register_buffer(
            "bryson_weight",
            torch.cat(
                [
                    torch.full((width,), 1.0 / nominal**2)
                    for width, nominal in (
                        (3, 0.01),  # root position (world)
                        (9, 0.02),  # root orientation matrix, scaled by sqrt(1/2)
                        (num_joints, 0.1),  # joint position
                        (num_joints, 3.14),  # joint velocity
                    )
                ]
            ),
            persistent=False,
        )

    def _rep_features(self, state: torch.Tensor) -> torch.Tensor:
        """Invariant features ``[z(1), g_b(3), v_b(3), w_b(3), qj, vj]``."""
        rotation = quat_xyzw_to_mat(state[..., self._quat])
        rotation_t = rotation.transpose(-1, -2)
        lin_b = (rotation_t @ state[..., self._lin].unsqueeze(-1)).squeeze(-1)
        ang_b = (rotation_t @ state[..., self._ang].unsqueeze(-1)).squeeze(-1)
        gravity = state.new_zeros(state.shape[:-1] + (3,))
        gravity[..., 2] = -1.0
        g_b = (rotation_t @ gravity.unsqueeze(-1)).squeeze(-1)
        return torch.cat(
            (
                state[..., 2:3],
                g_b,
                lin_b,
                ang_b,
                state[..., self._q],
                state[..., self._qd],
            ),
            -1,
        )

    def reduce_window_loss(self, loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return loss / valid.shape[0]

    def loss_metric(self, loss: torch.Tensor, window: int) -> torch.Tensor:
        return loss / window

    def _loss_channels(self, state: torch.Tensor) -> torch.Tensor:
        """Bryson-scored channels: world root pos, scaled root rotation
        matrix (so a plain squared difference is a nominal in radians), and
        joint positions/velocities."""
        rotation = quat_xyzw_to_mat(state[..., self._quat]).flatten(-2, -1) * math.sqrt(0.5)
        return torch.cat(
            (
                state[..., self._pos],
                rotation,
                state[..., self._q],
                state[..., self._qd],
            ),
            -1,
        )

    def prediction_error(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bryson-weighted channel-group MSE of the integrated states."""
        difference = (self._loss_channels(prediction) - self._loss_channels(target)).square()
        width = difference.shape[-1]
        return (difference * self.bryson_weight).sum(-1) / width

    @torch.no_grad()
    def update_statistics(self, state, action, successor) -> None:
        self.state_norm.update(self._rep_features(state))
        self.action_norm.update(action)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rotation = quat_xyzw_to_mat(state[..., self._quat])
        rotation_t = rotation.transpose(-1, -2)
        lin_b = (rotation_t @ state[..., self._lin].unsqueeze(-1)).squeeze(-1)
        ang_b = (rotation_t @ state[..., self._ang].unsqueeze(-1)).squeeze(-1)
        joint_pos, joint_vel = state[..., self._q], state[..., self._qd]
        prediction = run_mlp(
            self.net,
            torch.cat(
                (self.state_norm(self._rep_features(state)), self.action_norm(action)),
                -1,
            ),
            self.amp_dtype,
        )
        d_lin, d_ang, d_joint_vel = prediction.split((3, 3, self.num_joints), -1)
        c = self.substep_coeff
        dt = self.dt

        next_joint_vel = joint_vel + d_joint_vel
        next_joint_pos = joint_pos + (joint_vel + c * d_joint_vel) * dt
        # Next twist in world coordinates; pose increments weight the delta
        # by the semi-implicit substep coefficient.
        next_lin = (rotation @ (lin_b + d_lin).unsqueeze(-1)).squeeze(-1)
        next_ang = (rotation @ (ang_b + d_ang).unsqueeze(-1)).squeeze(-1)
        pose_lin = (rotation @ (lin_b + c * d_lin).unsqueeze(-1)).squeeze(-1)
        pose_ang = (rotation @ (ang_b + c * d_ang).unsqueeze(-1)).squeeze(-1)
        next_state = state.clone()
        next_state[..., self._pos] = state[..., self._pos] + pose_lin * dt
        next_state[..., self._quat] = quat_integrate_xyzw(state[..., self._quat], pose_ang, dt)
        next_state[..., self._lin] = next_lin
        next_state[..., self._ang] = next_ang
        next_state[..., self._q] = next_joint_pos
        next_state[..., self._qd] = next_joint_vel
        return next_state


class MlpActor(nn.Module):
    """Deterministic actor over the flattened normalized observation history.

    The single action-boundary transform (raw output -> command) is applied
    by the caller as ``action_limit * raw``, consistently in collection,
    imagination, and export. The output is unsquashed, as upstream.
    """

    amp_dtype: torch.dtype | None = None

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.net = _mlp(input_dim, hidden, action_dim, act=nn.GELU, layer_norm=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, features, self.amp_dtype)


class GoalConditionedActor(nn.Module):
    """Multi-goal-encoder actor (upstream MultiEncoderAE, single-policy form).

    Input is the concatenated actor contract: normalized observation history
    followed by the normalized goal blocks of every configured encoder, in
    encoder order (upstream stacks proprioception only). Each goal block
    passes its own encoder MLP to a tanh-bounded latent (upstream's
    bottleneck, 32 by default), and the decoder consumes the concatenated
    latents alongside the history — so the decoder width matches upstream's
    (latents + stacked proprioception), not the raw goal widths. One encoder
    degenerates to the single-goal actor.
    """

    amp_dtype: torch.dtype | None = None

    def __init__(
        self,
        input_dim: int,
        goal_slices: Sequence[int],
        action_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
    ):
        super().__init__()
        self.goal_slices = tuple(goal_slices)
        if not self.goal_slices or any(width <= 0 for width in self.goal_slices):
            raise ValueError(f"Goal-conditioned policies require positive goal slices, got {self.goal_slices}")
        if latent_dim <= 0:
            raise ValueError("Goal-conditioned policies require a positive latent dimension")
        self.goal_dim = sum(self.goal_slices)
        self.encoders = nn.ModuleList(
            _mlp(width, hidden, latent_dim, act=nn.GELU, layer_norm=True) for width in self.goal_slices
        )
        self.decoder = _mlp(
            input_dim - self.goal_dim + latent_dim * len(self.goal_slices),
            hidden,
            action_dim,
            act=nn.GELU,
            layer_norm=True,
        )
        # Near-zero decoder output init (weights x0.01, bias 0): the collector
        # multiplies the raw output by the task action limit (~5 rad-scale for
        # G1), so a default-init output already commands multi-radian targets
        # on iteration 1 — action-rate penalties explode (rewards ~-8/step,
        # NaN gradients; 2026-11-11 GPU run). Upstream needs no guard because
        # it applies no action-limit multiplier.
        output = self.decoder[-1]
        assert isinstance(output, nn.Linear)
        output.weight.data.mul_(0.01)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        history, goal = features[..., : -self.goal_dim], features[..., -self.goal_dim :]
        latents = [
            torch.tanh(run_mlp(encoder, block, self.amp_dtype))
            for encoder, block in zip(self.encoders, goal.split(self.goal_slices, dim=-1), strict=True)
        ]
        # Unsquashed decoder output, as upstream.
        return run_mlp(self.decoder, torch.cat((*latents, history), dim=-1), self.amp_dtype)


class MlpCritic(nn.Module):
    """State-value head ``V(obs) -> (..., 1)`` (upstream critic: LN + GELU MLP)."""

    amp_dtype: torch.dtype | None = None

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1, act=nn.GELU, layer_norm=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, obs, self.amp_dtype)

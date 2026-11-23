# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Motion encoder/decoder and differentiable normalization.

Motion-only networks with initialization, normalization and precision choices
benchmarked against Uni-SuperMimic e8da48f.
"""

import torch
from torch import nn

from motrix_env_core.observations import append_observation, repeat_observation
from motrix_rl.normalize import EmpiricalNormalization


def mlp(input_dim, hidden_dims, output_dim, *, layer_norm=False):
    layers = []
    for width in hidden_dims:
        layers.append(nn.Linear(input_dim, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU() if layer_norm else nn.ELU())
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Normalizer(EmpiricalNormalization):
    """Explicit statistics updates; frozen statistics never detach the input.

    Maintains the shared NORM_KEYS surface. Uses population parallel variance
    with the original mean delta, including between-batch variance.
    """

    def __init__(self, width, eps=1e-2):
        super().__init__(width, "cpu", eps=eps)

    def forward(self, value):
        return (value - self._mean) / self.std()

    def std(self):
        return (self._var + self.eps**2).sqrt()

    def inverse(self, value):
        return value * self.std() + self._mean

    @torch.no_grad()
    def update(self, value):
        value = value.reshape(-1, self._mean.shape[-1])
        count = value.shape[0]
        mean = value.mean(0, keepdim=True)
        var = value.var(0, unbiased=False, keepdim=True)
        total = self.count + count
        delta = mean - self._mean
        m2 = self._var * self.count + var * count + delta.square() * self.count * count / total
        self._mean.add_(delta * count / total)
        self._var.copy_(m2 / total)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(total)


class MotionActor(nn.Module):
    def __init__(self, proprio_dim, goal_dim, action_dim, cfg):
        super().__init__()
        self.mixed_precision = cfg.amp
        self.proprio_dim = proprio_dim
        self.goal_dim = goal_dim
        self.history_length = cfg.history_length
        self.obs_normalizer = Normalizer(proprio_dim)
        self.goal_normalizer = Normalizer(goal_dim)
        self.encoder = mlp(goal_dim, cfg.actor_hidden_dims, cfg.latent_dim)
        self.decoder = mlp(cfg.latent_dim + cfg.history_length * proprio_dim, cfg.actor_hidden_dims, action_dim)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(0.01)
            self.decoder[-1].bias.zero_()

    def forward(self, history, goal):
        latent = torch.tanh(run_mlp(self.encoder, self.goal_normalizer(goal), self.mixed_precision))
        return run_mlp(
            self.decoder, torch.cat([latent, self.obs_normalizer(history).flatten(1)], dim=-1), self.mixed_precision
        )


class Policy(nn.Module):
    """Flat export contract: frame-major history, followed by actor goal."""

    def __init__(self, actor, low, high):
        super().__init__()
        self.actor = actor
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32).clone())
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32).clone())

    @property
    def input_dim(self):
        return self.actor.proprio_dim * self.actor.history_length + self.actor.goal_dim

    def forward(self, obs):
        width = self.actor.proprio_dim * self.actor.history_length
        history = obs[:, :width].reshape(-1, self.actor.history_length, self.actor.proprio_dim)
        return self.actor(history, obs[:, width:]).clamp(self.low, self.high)


class History:
    """Collector/play history, oldest first; reset rows repeat their first frame."""

    def __init__(self, frame, length):
        self.value = repeat_observation(frame, length).clone()

    def advance(self, frame, done):
        self.value = append_observation(self.value, frame, done)
        return self.value


def run_mlp(network, value, mixed_precision):
    # Physics, normalization and loss arithmetic stay float32; only GEMMs use bf16.
    with torch.autocast(
        value.device.type, dtype=torch.bfloat16, enabled=mixed_precision and value.device.type == "cuda"
    ):
        return network(value).float()


class Critic(nn.Module):
    def __init__(self, width, hidden_dims, mixed_precision):
        super().__init__()
        self.input_normalizer = Normalizer(width)
        self.return_normalizer = Normalizer(1)
        self.net = mlp(width, hidden_dims, 1)
        self.mixed_precision = mixed_precision

    def normalized(self, value):
        return run_mlp(self.net, self.input_normalizer(value), self.mixed_precision)

    def forward(self, value):
        return self.return_normalizer.inverse(self.normalized(value))

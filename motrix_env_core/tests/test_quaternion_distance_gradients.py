# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Rotation distance remains finite and differentiable at coincident orientations."""

import torch

from motrix_env_core.math.quaternion import rotation_distance


def test_coincident_and_opposite_sign_quaternion_gradients():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True)
    for target in (identity.detach(), -identity.detach()):
        distance = rotation_distance(identity, target)
        torch.testing.assert_close(distance, torch.zeros(1))
        (gradient,) = torch.autograd.grad(distance.sum(), identity)
        assert torch.isfinite(gradient).all()


def test_half_turn_distance_and_gradient():
    current = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    distance = rotation_distance(current, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    torch.testing.assert_close(distance, torch.tensor([torch.pi]))
    (gradient,) = torch.autograd.grad(distance.sum(), current)
    assert torch.isfinite(gradient).all()

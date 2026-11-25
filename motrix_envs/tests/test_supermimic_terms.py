# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Motion window boundaries and deployed versus privileged input contracts."""

from types import SimpleNamespace

import torch

from motrix_envs.locomotion.wbt.mdp.supermimic import clip_end, motion_goal


def test_goal_window_clamps_each_clip_and_uses_column_rotation_layout():
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    yaw90 = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    clip = SimpleNamespace(
        joint_pos=torch.arange(6.0).reshape(6, 1),
        joint_vel=torch.arange(6.0).reshape(6, 1) + 10,
        root_body_pos_w=torch.arange(18.0).reshape(6, 3),
        root_body_quat_w=yaw90.repeat(6, 1),
    )
    command = SimpleNamespace(steps=torch.tensor([[1], [4]]), frame_end=torch.tensor([[2], [5]]), clip=clip)
    ctx = SimpleNamespace(
        commands={"motion": command}, sim={"root_pos": torch.zeros(2, 3), "root_quat": identity.repeat(2, 1)}
    )
    actor, privileged = torch.empty(2, 24), torch.empty(2, 33)
    offsets = torch.tensor([0, 2, 4])
    motion_goal(ctx, actor, offsets, False)
    motion_goal(ctx, privileged, offsets, True)
    expected = torch.tensor([[1.0, 11.0, 2.0, 12.0, 2.0, 12.0], [4.0, 14.0, 5.0, 15.0, 5.0, 15.0]])
    torch.testing.assert_close(actor[:, :6], expected)
    torch.testing.assert_close(
        actor[:, 6:], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]).repeat(2, 3), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(privileged[:, :6], actor[:, :6])
    torch.testing.assert_close(privileged[:, -18:], actor[:, -18:])
    assert not clip_end(ctx).any()
    command.steps.copy_(command.frame_end)
    assert clip_end(ctx).all()


def test_goal_orientation_is_differentiable_without_privileged_position():
    rotation = torch.tensor([[0.1, 0.2, 0.3, 0.9]], requires_grad=True)
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    position = torch.zeros(1, 3, requires_grad=True)
    clip = SimpleNamespace(
        joint_pos=torch.zeros(1, 1),
        joint_vel=torch.zeros(1, 1),
        root_body_pos_w=torch.ones(1, 3),
        root_body_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    ctx = SimpleNamespace(
        commands={
            "motion": SimpleNamespace(
                steps=torch.zeros(1, 1, dtype=torch.long), frame_end=torch.zeros(1, 1, dtype=torch.long), clip=clip
            )
        },
        sim={"root_pos": position, "root_quat": rotation},
    )
    out = torch.empty(1, 8)
    motion_goal(ctx, out, torch.tensor([0]), False)
    gradient = torch.autograd.grad(out.sum(), (rotation, position), allow_unused=True)
    assert gradient[0].abs().sum() > 0
    assert gradient[1] is None

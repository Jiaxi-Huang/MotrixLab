# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract tests for the manager-based humanoid walk presets."""

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.config.scene import ProceduralHFieldAssetCfg
from motrix_env_core.input import ConstantPlanarVelocitySourceCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_motrixsim.input import RendererKeyboardPlanarVelocitySourceCfg
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg
from motrix_envs.locomotion.humanoid.dex_evt import (
    make_dex_evt_walk_flat_cfg,
    make_dex_evt_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.g1 import (
    make_g129dof_walk_flat_cfg,
    make_g129dof_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.k1 import (
    make_k1_walk_flat_cfg,
    make_k1_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.microduck import (
    make_microduck_walk_flat_cfg,
    make_microduck_walk_rough_cfg,
)


@pytest.mark.parametrize(
    ("env_name", "action_dim", "policy_dim", "value_dim"),
    [
        ("g1-walk-flat", 29, 100, 103),
        ("dex-evt-walk-flat", 23, 82, 85),
        ("k1-walk-flat", 22, 79, 82),
        ("microduck-walk-flat", 14, 55, 58),
    ],
)
def test_walk_presets_use_shared_manager_env(env_name, action_dim, policy_dim, value_dim):
    env = registry.make(env_name, num_envs=2)
    state = env.init_state()

    assert type(env) is ManagerEnv
    assert isinstance(env.cfg, HumanoidVelocityTrackingManagerEnvCfg)
    assert env.action_space.shape == (action_dim,)
    assert state.obs.policy.shape == (2, policy_dim)
    assert state.obs.value.shape == (2, value_dim)


@pytest.mark.parametrize(
    ("make_flat_cfg", "make_rough_cfg"),
    [
        (make_g129dof_walk_flat_cfg, make_g129dof_walk_rough_cfg),
        (make_dex_evt_walk_flat_cfg, make_dex_evt_walk_rough_cfg),
        (make_k1_walk_flat_cfg, make_k1_walk_rough_cfg),
        (make_microduck_walk_flat_cfg, make_microduck_walk_rough_cfg),
    ],
)
def test_walk_rough_only_overrides_scene_spawn_range_and_render_spacing(make_flat_cfg, make_rough_cfg):
    flat_cfg = make_flat_cfg()
    rough_cfg = make_rough_cfg()

    assert rough_cfg.rewards == flat_cfg.rewards
    assert rough_cfg.commands == flat_cfg.commands
    assert rough_cfg.actions == flat_cfg.actions
    assert rough_cfg.terminations == flat_cfg.terminations
    assert rough_cfg.sim == flat_cfg.sim
    assert flat_cfg.sim_reset.humanoid_state.spawn_xy_range == 0.0
    assert rough_cfg.sim_reset.humanoid_state.spawn_xy_range > flat_cfg.sim_reset.humanoid_state.spawn_xy_range
    assert flat_cfg.render_spacing > 0.0
    assert rough_cfg.render_spacing == 0.0
    assert isinstance(rough_cfg.scene.assets.terrain, ProceduralHFieldAssetCfg)


def test_humanoid_walk_rejects_incomplete_joint_preset():
    cfg = make_g129dof_walk_flat_cfg()
    cfg.scene.objs.robot.key_pose.joint_names.pop(0)
    cfg.scene.objs.robot.key_pose.poses["default"].pop(0)

    # The backend rejects incomplete key poses at model compile time.
    with pytest.raises(ValueError, match="must cover every joint"):
        ManagerEnv(cfg, num_envs=2)


def test_microduck_walk_command_source_drives_command_buffer():
    cfg = make_microduck_walk_flat_cfg()
    cfg.commands.walk.source = ConstantPlanarVelocitySourceCfg(value=(0.5, 0.0, 0.0))
    env = ManagerEnv(cfg, num_envs=2)
    state = env.init_state()

    env.compute_transition(state)

    walk = env.command_terms["walk"]
    # Whole batch shares one command (teleop repeat semantics) ...
    np.testing.assert_allclose(walk.command, [[0.5, 0.0, 0.0]] * 2)
    # ... while the kernel-side lifecycle (step clock) keeps running.
    assert np.all(walk.steps >= 1.0)


def test_microduck_walk_play_cfg_drives_command_from_keyboard():
    # for_play() swaps the walk command source to the renderer keyboard; the
    # command range mirrors the training vel_limit scaled by 0.5 (teleop stays
    # in the well-trained low-speed range), and without a renderer the device
    # stays neutral (zero command) while the lifecycle keeps running.
    env_cfg = registry.make_env_config("microduck-walk-flat", mode="play")
    assert isinstance(env_cfg.commands.walk.source, RendererKeyboardPlanarVelocitySourceCfg)
    assert env_cfg.max_episode_seconds is None  # play episodes have no length cap
    assert env_cfg.commands.walk.source.command_lower == tuple(v * 0.5 for v in env_cfg.commands.walk.vel_limit[0])
    assert env_cfg.commands.walk.source.command_upper == tuple(v * 0.5 for v in env_cfg.commands.walk.vel_limit[1])
    # Training keeps the random-sampling path.
    assert registry.make_env_config("microduck-walk-flat").commands.walk.source is None

    env = registry.make("microduck-walk-flat", mode="play", num_envs=2)
    state = env.init_state()

    env.compute_transition(state)

    walk = env.command_terms["walk"]
    np.testing.assert_allclose(walk.command, np.zeros((2, 3), dtype=np.float32))
    assert np.all(walk.steps >= 1.0)

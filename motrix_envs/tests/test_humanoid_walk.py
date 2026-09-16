# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract tests for the manager-based humanoid walk presets."""

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg, HumanoidWalkSceneCfg
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
    make_microduck_walk_stairs_cfg,
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
    assert rough_cfg.sim_reset.humanoid_state.spawn_xy_range == 4.0
    assert flat_cfg.render_spacing > 0.0
    assert rough_cfg.render_spacing == 0.0
    assert rough_cfg.scene.assets.terrain.size == (32.0, 32.0)


@pytest.mark.parametrize(
    ("make_cfg", "camera_distance", "camera_lookat"),
    [
        (make_g129dof_walk_flat_cfg, 6.0, None),
        (make_g129dof_walk_rough_cfg, 6.0, None),
        (make_dex_evt_walk_flat_cfg, 6.0, None),
        (make_dex_evt_walk_rough_cfg, 6.0, None),
        (make_k1_walk_flat_cfg, 6.0, None),
        (make_k1_walk_rough_cfg, 6.0, None),
        (make_microduck_walk_flat_cfg, 0.35, (0.0, 0.0, 0.12)),
        (make_microduck_walk_rough_cfg, 0.35, (0.0, 0.0, 0.12)),
    ],
)
def test_walk_presets_use_shared_system_camera(make_cfg, camera_distance, camera_lookat):
    scene = make_cfg().scene

    assert isinstance(scene, HumanoidWalkSceneCfg)
    assert scene.system_camera.lookat == camera_lookat
    assert scene.system_camera.distance == pytest.approx(camera_distance)
    assert scene.system_camera.elevation == pytest.approx(-20.0)
    assert scene.system_camera.azimuth == pytest.approx(180.0)


def test_dex_evt_and_k1_walk_use_shared_reward_weights_and_named_contact_termination():
    g1_cfg = make_g129dof_walk_flat_cfg()
    dex_cfg = make_dex_evt_walk_flat_cfg()
    k1_cfg = make_k1_walk_flat_cfg()

    for term_name in ("tracking_lin_vel", "tracking_ang_vel", "penalty_action_rate", "pose"):
        assert getattr(dex_cfg.rewards, term_name).weight == getattr(g1_cfg.rewards, term_name).weight
        assert getattr(k1_cfg.rewards, term_name).weight == getattr(g1_cfg.rewards, term_name).weight
    assert len(dex_cfg.terminations.colliding.termination_geoms) == 14
    assert len(k1_cfg.terminations.colliding.termination_geoms) == 18
    assert len(make_microduck_walk_flat_cfg().terminations.colliding.termination_geoms) == 1


def test_humanoid_walk_rejects_incomplete_joint_preset():
    cfg = make_g129dof_walk_flat_cfg()
    cfg.scene.objs.robot.key_pose.joint_names.pop(0)
    cfg.scene.objs.robot.key_pose.poses["default"].pop(0)

    # The backend rejects incomplete key poses at model compile time.
    with pytest.raises(ValueError, match="must cover every joint"):
        ManagerEnv(cfg, num_envs=2)


def test_microduck_walk_stairs_env_registers_and_builds():
    env = registry.make("microduck-walk-stairs", num_envs=1)
    state = env.init_state()

    assert type(env) is ManagerEnv
    assert isinstance(env.cfg, HumanoidVelocityTrackingManagerEnvCfg)
    assert env.action_space.shape == (14,)
    assert state.obs.policy.shape == (1, 55)
    env.step(np.zeros((1, 14), dtype=np.float32))


def test_microduck_walk_stairs_step_height_is_configurable():
    default_cfg = make_microduck_walk_stairs_cfg()
    default_cells = default_cfg.scene.assets.terrain.generator.regions
    assert all(cell.generator.step_height == pytest.approx(0.04) for cell in default_cells)
    # Convex (mound) and concave (pit) tiles alternate in a checkerboard.
    assert default_cells[0].generator.profile == "descending"
    assert default_cells[1].generator.profile == "ascending"

    tall_cfg = make_microduck_walk_stairs_cfg(step_height=0.08)
    tall_cells = tall_cfg.scene.assets.terrain.generator.regions
    assert all(cell.generator.step_height == pytest.approx(0.08) for cell in tall_cells)
    # Mound tiles sit on the raised base plane, pit tiles reach down to the floor.
    assert tall_cells[0].generator.base_level == pytest.approx(0.5)
    assert tall_cells[1].generator.base_level == pytest.approx(0.0)

    # Seamless tiling: every tile rim is at the shared base plane, so no vertical
    # wall larger than a single step exists anywhere in the field.
    heights = tall_cfg.scene.assets.terrain.generator.generate((32.0, 32.0), (320, 320))
    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)
    step_jump = 0.08 / tall_cfg.scene.assets.terrain.generator.height_scale
    assert np.abs(np.diff(heights, axis=0)).max() <= step_jump + 1e-6
    assert np.abs(np.diff(heights, axis=1)).max() <= step_jump + 1e-6


def test_humanoid_walk_rejects_incomplete_joint_preset():
    cfg = make_g129dof_walk_flat_cfg()
    cfg.scene.objs.robot.key_pose.joint_names.pop(0)
    cfg.scene.objs.robot.key_pose.poses["default"].pop(0)

    # The backend rejects incomplete key poses at model compile time.
    with pytest.raises(ValueError, match="must cover every joint"):
        ManagerEnv(cfg, num_envs=2)

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for quadruped flat-, rough-, and stairs-terrain walk tasks."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from omegaconf import MISSING

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    CompositeTerrainGeneratorCfg,
    ContactSensorCfg,
    ContactSensorReduce,
    FlatTerrainGeneratorCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SceneSensorsCfg,
    StairsTerrainGeneratorCfg,
    TerrainRegionCfg,
    grid_terrain,
)
from motrix_env_core.direct.env import DirectEnvCfg
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneCfg
from motrix_envs.robot import QuadrupedRobotCfg


def _contact_sensor(geom_name: str | None) -> ContactSensorCfg:
    if geom_name is None:
        raise ValueError("Quadruped task requires a contact geom name for every leg")
    return ContactSensorCfg(geom1="floor", geom2=geom_name, reduce=ContactSensorReduce.mindist)


@configclass
class QuadrupedTaskSensorsCfg(SceneSensorsCfg):
    """Required contact sensors for the four quadruped task legs."""

    front_left_contact: ContactSensorCfg = MISSING
    front_right_contact: ContactSensorCfg = MISSING
    rear_left_contact: ContactSensorCfg = MISSING
    rear_right_contact: ContactSensorCfg = MISSING


@configclass
class QuadrupedSceneCfg(StandardSceneCfg):
    """A standard scene whose foot contact sensors are derived from its quadruped robot."""

    sensors: QuadrupedTaskSensorsCfg = QuadrupedTaskSensorsCfg()

    def __post_init__(self) -> None:
        robot = self.objs.robot
        if not isinstance(robot, QuadrupedRobotCfg):
            raise TypeError(f"QuadrupedSceneCfg robot must be QuadrupedRobotCfg, got {type(robot).__name__}")
        front_left, front_right, rear_left, rear_right = robot.foot_contact_geom_names
        self.sensors = QuadrupedTaskSensorsCfg(
            front_left_contact=_contact_sensor(front_left),
            front_right_contact=_contact_sensor(front_right),
            rear_left_contact=_contact_sensor(rear_left),
            rear_right_contact=_contact_sensor(rear_right),
        )


@configclass
class QuadrupedWalkRoughSceneAssetsCfg(StandardSceneAssetsCfg):
    """Standard scene assets plus the procedural rough height field used by rough-walk tasks."""

    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(
            seed=0,
            height_scale=0.1,
            flip_y=True,
        ),
        size=(64.0, 64.0),
        shape=(320, 320),
    )


# Stairs field layout: a centered 4x4 checkerboard of stair cells on a flat
# corridor covering the rest of the field.
_STAIRS_FIELD_CELLS = 8
_STAIRS_BLOCK_CELLS = 4
_STAIRS_STEP_HEIGHT = 0.08
_STAIRS_STEP_WIDTH = 0.3
_STAIRS_PLATFORM_WIDTH = 2.0
# Structure difficulty by block ring: the inner 2x2 platforms and pits span
# three risers, the outer twelve six.
_STAIRS_INNER_STEPS = 4
_STAIRS_OUTER_STEPS = 7


def _stairs_rise(step_count: int) -> float:
    """Corridor-to-plateau (platform) or rim-to-floor (pit) height."""
    return (step_count - 1) * _STAIRS_STEP_HEIGHT


def _stairs_corridor() -> float:
    """Corridor height above the height-field floor; also the deepest pit depth.

    Pits need headroom below their rim, so the corridor sits at half the field
    span: platforms climb above it and pits sink below it, both anchored to the
    cell-edge corridor through ``base_level``.
    """
    return _stairs_rise(_STAIRS_OUTER_STEPS)


def _stairs_field_scale() -> float:
    """Full height-field span: corridor plus the tallest platform rise."""
    return _stairs_corridor() + _stairs_rise(_STAIRS_OUTER_STEPS)


def _stairs_cell(step_count: int, *, descending: bool) -> StairsTerrainGeneratorCfg:
    """One pyramid-stairs platform (``descending``) or pit (``ascending``).

    A central plateau or pit floor with one exact ``step_height`` riser per
    ``step_width`` ring back to the cell-edge corridor. Both scales span to
    fraction 1.0, so the composite output covers [0, 1] exactly and the
    engine's min-max height-field normalization stays the identity.
    """
    corridor = _stairs_corridor()
    rise = _stairs_rise(step_count)
    if descending:
        scale = corridor + rise
        base_level = corridor / scale
    else:
        scale = corridor
        base_level = (corridor - rise) / scale
    return StairsTerrainGeneratorCfg(
        axis="radial",
        profile="descending" if descending else "ascending",
        step_count=step_count,
        step_height=_STAIRS_STEP_HEIGHT,
        step_width=_STAIRS_STEP_WIDTH,
        platform_width=_STAIRS_PLATFORM_WIDTH,
        base_level=base_level,
        height_scale=scale,
    )


def _stairs_grid() -> CompositeTerrainGeneratorCfg:
    """Alternating platforms and pits with difficulty by block ring."""

    def step_count(i: int, j: int) -> int:
        ring = max(abs(2 * i - (_STAIRS_BLOCK_CELLS - 1)), abs(2 * j - (_STAIRS_BLOCK_CELLS - 1)))
        return _STAIRS_INNER_STEPS if ring == 1 else _STAIRS_OUTER_STEPS

    cells = [
        [
            _stairs_cell(
                step_count(i, j),
                descending=(i + j) % 2 == 0,
            )
            for j in range(_STAIRS_BLOCK_CELLS)
        ]
        for i in range(_STAIRS_BLOCK_CELLS)
    ]
    return grid_terrain(
        cells,
        height_scale=_stairs_field_scale(),
        base=FlatTerrainGeneratorCfg(
            height=_stairs_corridor() / _stairs_field_scale(),
            height_scale=_stairs_field_scale(),
        ),
    )


def _stairs_terrain() -> CompositeTerrainGeneratorCfg:
    """The stairs-walk field: one stair grid surrounded by flat corridor."""
    corridor = FlatTerrainGeneratorCfg(
        height=_stairs_corridor() / _stairs_field_scale(),
        height_scale=_stairs_field_scale(),
    )
    block_fraction = _STAIRS_BLOCK_CELLS / _STAIRS_FIELD_CELLS
    return CompositeTerrainGeneratorCfg(
        base=corridor,
        regions=(
            TerrainRegionCfg(
                generator=_stairs_grid(),
                center=(0.5, 0.5),
                size=(block_fraction, block_fraction),
            ),
        ),
        height_scale=_stairs_field_scale(),
    )


@configclass
class QuadrupedWalkStairsSceneAssetsCfg(StandardSceneAssetsCfg):
    """Standard scene assets plus the procedural stairs terrain used by stairs-walk tasks.

    Like the rough-walk field this is one bounded 64 m square. A centered 32 m
    block carries a 4x4 grid of pyramid-stairs platforms and pits:
    2.0 m plateaus and pit floors with exact 0.08 m risers on 0.3 m treads,
    three risers in the inner 2x2 and six in the outer twelve, every structure
    anchored to the cell-edge corridor through ``base_level``. Pits need
    headroom below their rim, so the corridor sits at 0.48 m - half the field
    span - and the deepest pit floors bottom out at the height-field floor.
    Outside the block, the field is flat corridor.
    """

    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=_stairs_terrain(),
        size=(64.0, 64.0),
        shape=(641, 641),
    )

    def spawn_points(self) -> tuple[tuple[float, float], ...]:
        """Return fixed spawn slots at the stairs structures' centers.

        Platform plateaus and pit floors are both slots, so every episode
        starts on a stair structure and traverses it.
        """
        generator = self.terrain.generator
        size = np.asarray(self.terrain.size, dtype=np.float64)
        block = generator.regions[0]
        centers = (
            (
                block.center[0] - 0.5 * block.size[0] + (i + 0.5) / _STAIRS_BLOCK_CELLS * block.size[0],
                block.center[1] - 0.5 * block.size[1] + (j + 0.5) / _STAIRS_BLOCK_CELLS * block.size[1],
            )
            for i in range(_STAIRS_BLOCK_CELLS)
            for j in range(_STAIRS_BLOCK_CELLS)
        )
        return tuple((float((x - 0.5) * size[0]), float((y - 0.5) * size[1])) for x, y in centers)


@configclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@configclass
class ControlConfig:
    # action scale: target angle = action_scale * action + default_angle
    action_scale: float = 0.25
    simulate_action_latency: bool = False


@configclass
class QuadrupedWalkRandomizationCfg:
    """Episode-level domain randomization for quadruped walking."""

    enabled: bool = False
    joint_pos_noise: float = 0.0
    joint_vel_noise: float = 0.0
    base_lin_vel_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_ang_vel_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)
    action_delay_steps: tuple[int, int] = (0, 0)
    kp_scale_range: tuple[float, float] = (1.0, 1.0)
    damping_scale_range: tuple[float, float] = (1.0, 1.0)
    sliding_friction_range: tuple[float, float] | None = None
    base_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    base_com_offset_noise: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        scalar_noise = {
            "joint_pos_noise": self.joint_pos_noise,
            "joint_vel_noise": self.joint_vel_noise,
        }
        vector_noise = {
            "base_lin_vel_noise": self.base_lin_vel_noise,
            "base_ang_vel_noise": self.base_ang_vel_noise,
            "base_com_offset_noise": self.base_com_offset_noise,
        }
        for name, value in scalar_noise.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name, value in vector_noise.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must contain three finite non-negative values, got {value}")

        delay_low, delay_high = self.action_delay_steps
        if (
            isinstance(delay_low, bool)
            or isinstance(delay_high, bool)
            or not isinstance(delay_low, int)
            or not isinstance(delay_high, int)
            or delay_low < 0
            or delay_low > delay_high
            or delay_high > 1
        ):
            raise ValueError(
                "action_delay_steps must be an ordered integer range within the currently supported [0, 1], "
                f"got {self.action_delay_steps}"
            )

        for name, value in {
            "kp_scale_range": self.kp_scale_range,
            "damping_scale_range": self.damping_scale_range,
            "base_mass_scale_range": self.base_mass_scale_range,
        }.items():
            low, high = value
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(f"{name} must be a finite positive ordered range, got {value}")
        if self.sliding_friction_range is not None:
            low, high = self.sliding_friction_range
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(
                    f"sliding_friction_range must be a finite positive ordered range, got {self.sliding_friction_range}"
                )


@configclass
class VelocityCommandCfg:
    lower: NDArray[np.float32] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    upper: NDArray[np.float32] = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    standing_probability: float = 0.0
    standing_threshold: float = 0.05
    resampling_seconds_range: tuple[float, float] | None = None

    def validate(self) -> None:
        if self.resampling_seconds_range is not None:
            low, high = self.resampling_seconds_range
            if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0 or low > high:
                raise ValueError(
                    "resampling_seconds_range must be a finite positive ordered range, "
                    f"got {self.resampling_seconds_range}"
                )


@configclass
class Commands:
    velocity: VelocityCommandCfg = VelocityCommandCfg()


@configclass
class Sensor:
    # General-purpose sensors exposed by the assembled scene.
    local_linvel: str = "local_linvel"
    gyro: str = "gyro"
    upvector: str = "upvector"
    foot_positions: tuple[str, str, str, str] = ("FL_pos", "FR_pos", "RL_pos", "RR_pos")


@configclass
class RewardScales:
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 1.0
    lin_vel_z: float = -5.0
    ang_vel_xy: float = -0.1
    base_height: float = -100.0
    action_rate: float = -0.1
    similar_to_default: float = -0.1
    contact: float = 0.24
    swing_feet_z: float = 2.0
    swing_contact: float = -1.0


@configclass
class RewardConfig:
    scales: RewardScales = RewardScales()
    tracking_lin_vel_sigma: float = 0.25
    tracking_ang_vel_sigma: float = 0.25
    target_foot_height: float = 0.1
    swing_feet_height_sigma: float = 0.05
    base_height_target: float = 0.3


@configclass
class QuadrupedWalkEnvCfg(DirectEnvCfg):
    """Base configuration for quadruped walk tasks."""

    max_episode_seconds: float = 20.0
    scene: QuadrupedSceneCfg = MISSING
    noise_config: NoiseConfig = NoiseConfig()
    control_config: ControlConfig = ControlConfig()
    randomization: QuadrupedWalkRandomizationCfg = QuadrupedWalkRandomizationCfg()
    commands: Commands = Commands()
    sensor: Sensor = Sensor()
    reward_config: RewardConfig = RewardConfig()
    key_pose_name: str = "default"
    ground_geom_name: str = "floor"
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, 0.3)
    spawn_xy_range: float = 0.0
    # Fixed spawn slots (world xy); when non-empty, resets pick one slot at
    # random instead of uniform sampling inside spawn_xy_range.
    spawn_points: tuple[tuple[float, float], ...] = ()
    trot_pairs: tuple[tuple[int, int], ...] = ((0, 3), (1, 2))
    gait_frequency: float = 2.0
    sim: SimCfg = SimCfg(dt=0.01, solver_iterations=1)
    ctrl_dt: float = 0.02

    def validate(self) -> None:
        super().validate()
        self.randomization.validate()
        self.commands.velocity.validate()
        if (
            self.randomization.enabled
            and any(self.randomization.action_delay_steps)
            and self.control_config.simulate_action_latency
        ):
            raise ValueError("random action delay cannot be combined with simulate_action_latency")

    def for_play(self) -> "QuadrupedWalkEnvCfg":
        """Disable domain randomization and command changes for play/evaluation."""

        return replace(
            self,
            randomization=replace(self.randomization, enabled=False),
            commands=replace(
                self.commands,
                velocity=replace(self.commands.velocity, resampling_seconds_range=None),
            ),
        )


__all__ = [
    "Commands",
    "ControlConfig",
    "NoiseConfig",
    "QuadrupedWalkRandomizationCfg",
    "QuadrupedSceneCfg",
    "QuadrupedTaskSensorsCfg",
    "QuadrupedWalkEnvCfg",
    "QuadrupedWalkStairsSceneAssetsCfg",
    "QuadrupedWalkRoughSceneAssetsCfg",
    "RewardConfig",
    "RewardScales",
    "Sensor",
    "VelocityCommandCfg",
]

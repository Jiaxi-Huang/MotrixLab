# Config Overrides and Tuning

`QuadrupedWalkEnvCfg` provides defaults for the shared velocity-tracking logic. A robot config usually subclasses this
config and overrides only its robot, sensor mapping, action scale, reward targets, and terrain. Define one complete
flat-ground config first and derive terrain variants from it so the tasks differ only where intended.

## 1. Define a config

This skeleton shows the common override points. Fields not declared here retain the `QuadrupedWalkEnvCfg` defaults:

```python
@registry.envcfg("<robot>-walk-flat")
@configclass
class RobotWalkCfg(QuadrupedWalkEnvCfg):
    scene: QuadrupedSceneCfg = ...
    control_config: ControlConfig = ControlConfig(action_scale=0.25)
    commands: Commands = Commands(...)
    sensor: Sensor = Sensor(...)
    noise_config: NoiseConfig = NoiseConfig(...)
    reward_config: RewardConfig = RewardConfig(...)
    key_pose_name: str = "default"
    initial_base_position: tuple[float, float, float] = (0.0, 0.0, 0.3)
    spawn_xy_range: float = 4.0
```

See [Add a Robot Task](adding_robot.md) for the full `scene` and `sensor` integration contract.

## 2. `control_config`: action scale and latency

The policy action is a joint-position residual around the selected key pose:

$$
q_{target}=q_{key\_pose}+action\_scale\cdot a
$$

| Field                     | Meaning                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `action_scale`            | Target-angle change produced by a unit policy action; action-space numerical bounds scale inversely |
| `simulate_action_latency` | When `true`, physics control uses the previous control-step action, creating a fixed one-step delay |

Increasing `action_scale` makes the same network output produce a larger target-angle change while automatically reducing
the numerical action-space bounds. First verify that the default pose lies within every actuator control range, then
choose a scale according to joint range and PD response. Latency is not randomized; changing it alters the control
contract, so training and playback must use the same setting.

## 3. `commands`: velocity distribution

`Commands.velocity` groups the body-frame `[vx, vy, yaw_rate]` distribution and standing semantics. `lower` and `upper`
use that axis order:

```python
commands = Commands(
    velocity=VelocityCommandCfg(
        lower=np.array([-0.5, -0.3, -0.8], dtype=np.float32),
        upper=np.array([1.0, 0.3, 0.8], dtype=np.float32),
        standing_probability=0.1,
        standing_threshold=0.05,
    )
)
```

The three axes are sampled independently at reset and remain fixed for the episode. After sampling, the command is set
exactly to zero with `standing_probability`. A command whose Euclidean norm is below `standing_threshold` holds the gait
phase at zero so every foot uses the stance reference.

The shared config defaults to fixed `[0.5, 0.0, 0.0]`. Go2 overrides it with forward/backward, lateral, and turning ranges
and a nonzero standing probability. When changing those ranges, also check `tracking_ang_vel_sigma`, terrain boundaries,
spawn range, and whether runtime play/deploy command bindings stay inside the trained range.

## 4. `noise_config`: observation noise

The environment transforms every affected component as

$$
x_{obs}=x_{raw}+u\cdot level\cdot scale,\qquad u\sim U(-1,1)
$$

| Field               | Applies to                                                                  |
| ------------------- | --------------------------------------------------------------------------- |
| `level`             | Global multiplier for all observation noise; set it to `0` to disable noise |
| `scale_joint_angle` | Joint-position residual                                                     |
| `scale_joint_vel`   | Joint velocity                                                              |
| `scale_gyro`        | Body-frame angular velocity                                                 |
| `scale_gravity`     | Up-vector                                                                   |
| `scale_linvel`      | Body-frame linear velocity in the critic observation                        |

These are absolute amplitudes added directly to raw observations; there is no separate observation-normalization scale.
Actor and critic fields share the same noisy values, and the critic's local linear velocity is also noisy, so it is not a
fully noise-free privileged state. A useful integration sequence is to verify the model and rewards with `level=0`, then
increase noise gradually.

## 5. `gait_frequency` and `trot_pairs`: gait reference

`gait_frequency` specifies cycle frequency in Hz; increasing it shortens stance and swing. The default `2.0` is used by
the built-in configs. `trot_pairs` uses zero-based indices in front-left, front-right, rear-left, rear-right order. Its
default `((0, 3), (1, 2))` defines the two diagonal pairs.

These fields only change the reward reference; they do not generate a foot trajectory or low-level control signal. When
tuning frequency, also check `target_foot_height`, contact rewards, and whether the policy can lift a foot within the new
swing duration. Keep the default pairing unless the intended reference gait changes.

## 6. `reward_config`: tracking, posture, and gait

```python
reward_config = RewardConfig(
    scales=RewardScales(
        tracking_lin_vel=1.0,
        tracking_ang_vel=1.0,
        base_height=-100.0,
        # Override other shared terms as needed.
    ),
    tracking_lin_vel_sigma=0.25,
    tracking_ang_vel_sigma=0.05,
    target_foot_height=0.1,
    swing_feet_height_sigma=0.05,
    base_height_target=0.3,
)
```

| Field                     | Tuning consideration                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `scales`                  | Weights for raw reward terms; constraints use negative values, and the final sum is scaled by `ctrl_dt` |
| `tracking_lin_vel_sigma`  | Denominator of the XY velocity-error exponential kernel; smaller is stricter                            |
| `tracking_ang_vel_sigma`  | Denominator of the yaw-rate-error exponential kernel; smaller is stricter                               |
| `target_foot_height`      | Target swing-foot lift relative to stance, in meters                                                    |
| `swing_feet_height_sigma` | Foot-height exponential-kernel scale; the computation uses its square                                   |
| `base_height_target`      | Target body height above local ground, in meters                                                        |

`base_height_target` and `initial_base_position[2]` should normally be close to the robot's default standing height.
Choose `target_foot_height` according to leg length and terrain variation: too little encourages dragging, while too much
can require unreasonable joint motion. When tuning a reward weight, inspect the matching value in `state.reward_terms`
rather than comparing configuration numbers alone.

## 7. `sensor`: sensor-name mapping

`Sensor` maps state required by the shared task to sensor names in the assembled scene:

| Field            | Expected output                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `local_linvel`   | Three-dimensional body-frame linear velocity                                                              |
| `gyro`           | Three-dimensional body-frame angular velocity                                                             |
| `upvector`       | Three-dimensional body up-vector                                                                          |
| `foot_positions` | Four three-dimensional foot-position sensor names in front-left, front-right, rear-left, rear-right order |

`foot_positions` must contain four non-empty, unique names. The swing-height reward expects positions in a body reference
frame. When a robot asset lacks these sensors, add `FrameSensorCfg` entries in the scene config instead of copying the
environment implementation.

## 8. Terrain, spawn range, and episode length

| Field                   | Meaning                                                                         |
| ----------------------- | ------------------------------------------------------------------------------- |
| `ground_geom_name`      | Plane or height-field geom used for height queries; built-in scenes use `floor` |
| `initial_base_position` | Default world position of the floating base, in meters                          |
| `spawn_xy_range`        | Half-width of uniform reset sampling along x and y; `0` fixes the position      |
| `max_episode_seconds`   | Episode duration that produces truncation                                       |
| `ctrl_dt`               | Policy control period and the timestep scale applied to the reward sum          |

Built-in flat and rough configs both use `spawn_xy_range=4.0`. Terrain variants inherit the complete flat config and
override only the scene and terrain-specific fields: rough replaces `FlatTerrainCfg` with
`QuadrupedWalkRoughSceneAssetsCfg` and `HFieldTerrainCfg`, while stairs additionally uses fixed stair-structure spawn
slots. At reset, the environment raises the base above the highest of nine terrain samples near the spawn point; the
body-height reward also remains relative to local terrain during the episode.

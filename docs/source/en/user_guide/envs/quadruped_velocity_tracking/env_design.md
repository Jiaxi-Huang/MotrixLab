# Task Environment Design

`QuadrupedWalkTask` defines tracking body-frame planar velocity $v_x$, $v_y$, and yaw rate $\omega_z$ as a
robot-independent task. The shared environment owns actor and critic observations, joint-position residual control,
four-foot contacts and diagonal-trot phase, rewards, reset, and termination. A concrete robot supplies its model, default
pose, base link, and four foot-contact geoms through the `QuadrupedRobotCfg` in `QuadrupedWalkEnvCfg.scene`; the task
config supplies sensor names and numerical parameters.

## Action space

Let $A$ be the actuator count. The action space is an $A$-dimensional continuous `Box`, with one dimension per
joint-targeting actuator. The policy output is a joint-position residual around the pose selected by `key_pose_name`:

$$
q_{target}=q_{key\_pose}+action\_scale\cdot a
$$

The lower and upper bounds of each action dimension are computed separately from the actuator control range:

$$
a_{min,i}=\frac{q_{min,i}-q_{key\_pose,i}}{action\_scale},\qquad
a_{max,i}=\frac{q_{max,i}-q_{key\_pose,i}}{action\_scale}
$$

Action bounds are therefore usually asymmetric and can differ between joints. With `simulate_action_latency=false`, the
current action is applied immediately. When it is `true`, physics control uses the previous control-step action, while
the action field in the observation remains the current policy output.

## Observations

The environment uses asymmetric actor-critic observations. Let $F$ be the foot count; a quadruped config supplies four
feet in front-left, front-right, rear-left, and rear-right order, so $F=4$.

| Observation                               | Actor | Critic | Notes                                                        |
| ----------------------------------------- | ----: | -----: | ------------------------------------------------------------ |
| Body-frame angular velocity               |     3 |      3 | Read from `sensor.gyro`; optional uniform noise              |
| Negative up-vector                        |     3 |      3 | Negated value from `sensor.upvector`; optional uniform noise |
| Joint-position residual from the key pose |   $A$ |    $A$ | Optional uniform noise                                       |
| Joint velocity                            |   $A$ |    $A$ | Optional uniform noise                                       |
| Current action                            |   $A$ |    $A$ | Current policy output                                        |
| Velocity command                          |     3 |      3 | $v_x$, $v_y$, and $\omega_z$                                 |
| Four-foot gait phase                      |   $F$ |    $F$ | One phase value in $[0,1)$ per foot                          |
| Body-frame linear velocity                |     — |      3 | Privileged critic input; optional uniform noise              |

The actor dimension is $9+3A+F$, and the critic adds 3 dimensions. Every built-in robot has 12 actuators and four feet,
giving 49 actor dimensions and 52 critic dimensions.

`noise_config` adds separate zero-mean uniform noise to the gyro, up-vector, joint position, joint velocity, and critic
linear velocity. Fields shared by the actor and critic use the same noisy values. The current implementation has no
additional observation scaling or history stacking.

## Diagonal-trot reference

The environment advances a normalized cycle at `gait_frequency`. At the default `2 Hz`, the phase increases by
`ctrl_dt * gait_frequency` each control step. `trot_pairs=((0, 3), (1, 2))` groups the front-left with the rear-right
foot and the front-right with the rear-left foot. Feet in one pair are synchronized, and the two pairs are half a cycle
apart. A phase below `0.6` requests stance, while a phase at or above `0.6` requests swing.

When the Euclidean norm of the velocity command is below `commands.velocity.standing_threshold`, the shared and per-foot
phases are held at zero. All feet then use the stance reference instead of being forced through a trot cycle, which gives
explicit zero commands a consistent standing objective.

Four contact sensors determine whether the feet touch the ground, and four foot-position sensors report foot positions in
a body reference frame. Contact matching, swing clearance, and early swing-contact rewards all consume these states.

## Reward design

| Reward term          | Computation                                                                                          | Design purpose                              |
| -------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `tracking_lin_vel`   | Apply an exponential kernel to squared error between commanded and actual body-frame XY velocity     | Track forward and lateral velocity commands |
| `tracking_ang_vel`   | Apply an exponential kernel to squared yaw-rate tracking error                                       | Track turning commands                      |
| `lin_vel_z`          | Square body-frame vertical velocity                                                                  | Suppress unnecessary vertical motion        |
| `ang_vel_xy`         | Sum squared roll and pitch angular velocities                                                        | Suppress rapid body oscillation             |
| `base_height`        | Square error between body height above local ground and the target height                            | Maintain a robot-appropriate body height    |
| `action_rate`        | Sum squared action differences between consecutive control steps                                     | Reduce abrupt commands and joint jitter     |
| `similar_to_default` | Sum absolute joint-angle deviations from the key pose                                                | Keep the gait near a stable posture         |
| `contact`            | Measure the fraction of feet whose contact matches the stance or swing reference                     | Establish diagonal-trot contact timing      |
| `swing_feet_z`       | During non-contact swing, apply an exponential kernel to foot-height error and average over all feet | Lift swing feet toward the target clearance |
| `swing_contact`      | Measure the fraction of feet still touching the ground during swing                                  | Penalize dragging and early touchdown       |

Each raw term is first multiplied by its `RewardScales` weight; negative weights turn non-negative measurements into
penalties. `state.reward_terms` stores these weighted values before timestep scaling. Their sum is multiplied by `ctrl_dt` to
produce the reward returned to the training algorithm. The current environment has no reward curriculum.

On rough terrain, `base_height` uses terrain height beneath the robot as its zero point. `swing_feet_z` consumes foot
positions in a body reference frame and targets `target_foot_height - base_height_target`, making it invariant to a
uniform change in the robot's world-frame height.

## Termination conditions

| Type                  | Status       | Condition                                          | Meaning                                                |
| --------------------- | ------------ | -------------------------------------------------- | ------------------------------------------------------ |
| Tip-over termination  | `terminated` | The up-vector z component is no greater than `0.5` | The robot has tipped substantially or inverted         |
| Time-limit truncation | `truncated`  | Episode time reaches `max_episode_seconds`         | Normal completion at the time limit rather than a fall |

Foot, torso, or ground contact does not itself cause failure termination; foot contacts are used only by the gait rewards.

## Reset logic

Reset restores model state, places the floating base at `initial_base_position`, restores actuator joints to the pose
selected by `key_pose_name`, and clears all velocities together with current and previous actions. It then sets actuator
targets, refreshes kinematics, reads foot contacts and positions, samples a velocity command, and initializes the shared
and per-foot phases to zero.

| Randomized quantity  | Sampling                                                                                                                                                | Timing                                                    | Purpose                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Velocity command     | Sample each axis from `commands.velocity.lower` to `commands.velocity.upper`, then replace the result with zero with probability `standing_probability` | At every reset; not resampled within an episode           | Cover translation, turning, and explicit standing targets |
| Initial x/y position | When `spawn_xy_range > 0`, sample each coordinate uniformly from `[-spawn_xy_range, spawn_xy_range]`                                                    | At reset; raise z above the highest nearby terrain sample | Cover terrain regions without spawning inside the ground  |
| Observation noise    | Add uniform `[-1,1] * noise_config.level * scale_*` noise to each configured component                                                                  | Whenever observations are built                           | Improve robustness to sensor error                        |

These are task-command, initial-state, and observation randomization. The environment does not currently randomize mass,
inertia, friction, actuator parameters, or control latency, so it has no physical domain randomization.
`simulate_action_latency` is a fixed control-mode choice in each config rather than a per-episode random variable.

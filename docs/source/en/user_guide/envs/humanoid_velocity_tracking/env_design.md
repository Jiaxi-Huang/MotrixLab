# Task Environment Design

`HumanoidVelocityTrackingEnv` defines tracking body-frame planar velocity $v_x$, $v_y$, and yaw rate $\omega_z$ as a
robot-independent task. The shared environment builds actor and critic observations, converts policy actions into joint
position targets, maintains the biped gait, computes rewards and the penalty curriculum, and handles reset and termination.
It contains no concrete robot names and assumes no fixed joint count.

A concrete robot supplies model semantics and numerical parameters through `HumanoidVelocityTrackingEnvCfg`. The
`HumanoidRobotCfg` in `scene` identifies the robot model, default pose, base link, and left and right foot links; the
task-level `asset` identifies sole sites, the ground geom, and terminating-contact rules. Other values tune action scaling,
reward weights, and spawn ranges, allowing the same task logic to support humanoids with different sizes and joint layouts.

## Action space

Let $A$ be the number of position actuators. The action space is an $A$-dimensional continuous `Box`, with one dimension
per actuator. The policy outputs joint-position residuals around the default standing pose, and the environment computes
position-actuator targets as

$$
q_{target}=q_{default}+action\_scale\cdot a
$$

`control_config.action_scale` determines the target-angle change produced by a unit action. Each action dimension uses a
zero-centered symmetric bound:

$$
b_i=\frac{\max\left(\left|q_{min,i}-q_{default,i}\right|,
\left|q_{max,i}-q_{default,i}\right|\right)}{action\_scale},
\qquad a_i\in[-b_i,b_i]
$$

Here $q_{min,i}$ and $q_{max,i}$ come from the corresponding position actuator's `ctrl_range`, either declared directly
or inherited from its target joint. Action bounds can therefore differ between joints, and the default pose need not lie
at the midpoint of a control range.

## Observations

The environment uses asymmetric actor-critic observations. Let $J$ be the number of robot joint DoFs and $A$ the number
of actuators:

| Observation                         | Actor | Critic | Notes                               |
| ----------------------------------- | ----: | -----: | ----------------------------------- |
| Body-frame linear velocity          |     — |      3 | Privileged critic input             |
| Body-frame angular velocity         |     3 |      3 | Scaled by configuration             |
| Projected gravity                   |     3 |      3 | Gravity direction in the body frame |
| Velocity command                    |     3 |      3 | $v_x$, $v_y$, and $\omega_z$        |
| Joint-position residual             |   $J$ |    $J$ | Uniform noise on the actor input    |
| Joint velocity                      |   $J$ |    $J$ | Uniform noise on the actor input    |
| Current action                      |   $A$ |    $A$ | Current policy output               |
| Sine and cosine of both foot phases |     4 |      4 | One phase encoding per foot         |

With one actuator per joint, the actor dimension is $13+3J$ and the critic dimension is $16+3J$.

## Reward design

The environment advances both foot phases using `gait.period` and generates expected sole height from
`gait.swing_height`. A standing command pins the phase. On height-field terrain, ground height is sampled beneath each
sole every step, so clearance remains relative to the local surface.

| Reward term             | Computation                                                                                        | Design purpose                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `tracking_lin_vel`      | Apply an exponential kernel to squared body-frame planar-velocity tracking error                   | Track forward and lateral velocity commands accurately                            |
| `tracking_ang_vel`      | Apply an exponential kernel to squared yaw-rate tracking error                                     | Respond accurately to turning commands                                            |
| `penalty_ang_vel_xy`    | Sum squared roll and pitch angular velocities                                                      | Suppress rapid body oscillation                                                   |
| `penalty_orientation`   | Sum squared horizontal components of projected gravity                                             | Keep the torso upright                                                            |
| `penalty_action_rate`   | Sum squared action differences between consecutive control steps                                   | Reduce abrupt commands and joint jitter                                           |
| `feet_phase`            | Apply an exponential kernel to squared error between actual foot clearance and the phase reference | Produce support and swing timing consistent with the gait phase                   |
| `pose`                  | Compute a joint-weighted squared deviation from the default pose                                   | Constrain the torso and arms while retaining freedom in primary locomotion joints |
| `penalty_close_feet_xy` | Check whether lateral foot separation in the body-yaw frame is below a threshold                   | Prevent the feet from becoming too close or crossing                              |
| `penalty_feet_ori`      | Measure the horizontal gravity component in each foot frame                                        | Keep the soles approximately level                                                |
| `alive`                 | Return a constant value at each step                                                               | Make an early fall reduce the obtainable episode return                           |

Each raw term is multiplied by its `RewardScales` weight and the control timestep; negative weights turn constraint
measurements such as `penalty_*` and `pose` into penalties. Curriculum-selected penalties are also multiplied by the
current `penalty_scale` according to completed episode length. This factor is exposed through
`state.metrics["penalty_scale"]`, and the final weighted terms through `state.reward_terms`.

## Termination conditions

| Type                  | Status       | Condition                                                                                   | Meaning                                                        |
| --------------------- | ------------ | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Contact termination   | `terminated` | Any robot collision geom listed in `asset.terminate_contact_geom_names` contacts the ground | Normally detects non-foot contacts such as the pelvis or torso |
| Time-limit truncation | `truncated`  | Episode time reaches `max_episode_seconds`                                                  | Normal completion at the time limit rather than a fall         |

## Reset logic

Reset restores the configured default pose; clears joint and base velocities together with current and previous actions;
and then recomputes kinematic state. An uneven-terrain config also adjusts base height using terrain samples around the
spawn point.

The environment uses the following randomization:

| Randomized quantity           | Sampling                                                                                                                                                   | Timing                                                                 | Purpose                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Velocity command              | Sample each of the three dimensions uniformly within `commands.vel_limit`, then replace it with an all-zero command with probability `commands.stand_prob` | At reset and every `commands.resampling_time`                          | Cover translation, turning, and standing tasks                                 |
| Initial gait phase            | Sample one foot's phase offset uniformly from $[-\pi,\pi]$ and place the other foot half a cycle apart                                                     | At reset                                                               | Prevent environments from always beginning at the same point in the gait cycle |
| Uneven-terrain spawn position | When `spawn_xy_range > 0`, sample X and Y independently from `[-spawn_xy_range, spawn_xy_range]`                                                           | At reset; raise the base above the highest nearby terrain point        | Cover different terrain regions                                                |
| Actor joint-observation noise | Add uniform noise controlled by `noise_dof_pos` and `noise_dof_vel` to scaled joint positions and velocities                                               | Whenever observations are built; critic observations remain noise-free | Improve robustness to observation error                                        |

These entries are task-command, initial-state, and observation randomization. The environment does not currently
randomize physical parameters such as mass, friction, or PD gains, so it has no additional physical domain-randomization
terms.

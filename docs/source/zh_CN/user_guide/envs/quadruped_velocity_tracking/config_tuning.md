# 配置覆盖与调参

`QuadrupedWalkEnvCfg` 为共享速度跟踪逻辑提供默认参数。机器人配置通常继承该配置类，只覆盖机器人模型、
传感器映射、动作缩放、奖励目标和地形。先建立一份完整的平地配置，再让粗糙地形配置继承它，可以确保两个任务只在
场景上产生预期差异。

## 1. 定义配置

下面的骨架展示常用覆盖入口；未声明的字段沿用 `QuadrupedWalkEnvCfg` 默认值：

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

`scene` 与 `sensor` 的完整接入要求见[“新增机器人任务”](adding_robot.md)。

## 2. `control_config`：动作缩放与延迟

策略动作是相对于关键姿态的关节位置残差：

$$
q_{target}=q_{key\_pose}+action\_scale\cdot a
$$

| 字段                      | 含义                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `action_scale`            | 单位策略动作对应的目标关节角变化；动作空间边界会随之反向缩放 |
| `simulate_action_latency` | 为 `true` 时，物理控制使用上一控制步动作，形成固定一拍延迟   |

增大 `action_scale` 会让同样大小的网络输出产生更大的目标角变化，同时自动缩小动作空间数值边界。接入初期应先确认
默认姿态位于全部 actuator 控制范围内，再结合关节活动范围和 PD 响应选择缩放。动作延迟不会被随机采样；修改它会改变
控制契约，训练与回放必须使用相同配置。

## 3. `commands`：速度命令分布

`Commands.velocity` 集中配置机体坐标系 `[vx, vy, yaw_rate]` 的分布和站立语义；`lower` 和 `upper` 均按该轴顺序排列：

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

三个轴在 reset 时独立采样，并在整个回合内保持不变；采样完成后，以 `standing_probability` 的概率将命令精确设为
零。当命令的欧氏范数小于 `standing_threshold` 时，步态相位保持为 0，使所有脚使用支撑参考。

共享配置默认为固定 `[0.5, 0.0, 0.0]`；Go2 会覆盖为前后、横移和转向范围，并配置非零站立概率。修改这些范围时，
还应检查 `tracking_ang_vel_sigma`、地形边界、出生范围，以及 play/deploy 的 runtime command binding 是否位于训练
范围内。

## 4. `noise_config`：观察噪声

环境对每个受影响分量应用：

$$
x_{obs}=x_{raw}+u\cdot level\cdot scale,\qquad u\sim U(-1,1)
$$

| 字段                | 作用对象                                  |
| ------------------- | ----------------------------------------- |
| `level`             | 所有观察噪声的总乘数；设为 `0` 可关闭噪声 |
| `scale_joint_angle` | 关节位置偏差                              |
| `scale_joint_vel`   | 关节速度                                  |
| `scale_gyro`        | 机体局部角速度                            |
| `scale_gravity`     | Up-vector                                 |
| `scale_linvel`      | Critic 的机体局部线速度                   |

这些值是直接叠加到原始观察上的绝对幅度；当前配置没有单独的观察归一化缩放。Actor 与 Critic 共享的字段都会使用
同一份带噪值，critic 的局部线速度也会加噪，因此不要将其描述为完全无噪的 privileged state。通常先以
`level=0` 检查模型和奖励，再逐步增加噪声。

## 5. `gait_frequency` 与 `trot_pairs`：步态参考

`gait_frequency` 以 Hz 指定周期频率；增大它会缩短支撑和摆动时间。默认 `2.0` 适用于当前内置配置。
`trot_pairs` 使用前左、前右、后左、后右的零基索引，默认值 `((0, 3), (1, 2))` 定义两组对角腿。

这两个字段只改变奖励参考，不会直接生成脚部轨迹或低层控制信号。调节频率时应同步检查
`target_foot_height`、接触奖励和策略是否能在新的摆动时间内完成抬脚。除非要改变参考步态，否则应保留默认配对。

## 6. `reward_config`：跟踪、姿态与步态

```python
reward_config = RewardConfig(
    scales=RewardScales(
        tracking_lin_vel=1.0,
        tracking_ang_vel=1.0,
        base_height=-100.0,
        # 按需覆盖其余共享项
    ),
    tracking_lin_vel_sigma=0.25,
    tracking_ang_vel_sigma=0.05,
    target_foot_height=0.1,
    swing_feet_height_sigma=0.05,
    base_height_target=0.3,
)
```

| 字段                      | 调节重点                                                       |
| ------------------------- | -------------------------------------------------------------- |
| `scales`                  | 各原始奖励项的权重；约束项使用负值，所有项最终统一乘 `ctrl_dt` |
| `tracking_lin_vel_sigma`  | XY 线速度误差指数核的分母；越小越严格                          |
| `tracking_ang_vel_sigma`  | 偏航角速度误差指数核的分母；越小越严格                         |
| `target_foot_height`      | 摆动脚相对支撑位置的目标抬升高度，单位为米                     |
| `swing_feet_height_sigma` | 足端高度误差指数核的标准尺度，计算中使用其平方                 |
| `base_height_target`      | 机体相对局部地面的目标高度，单位为米                           |

`base_height_target` 和 `initial_base_position[2]` 通常应接近机器人的默认站立高度。`target_foot_height` 应结合
腿长和地形起伏设置；过低容易拖脚，过高可能要求超出合理关节范围。调节某个奖励权重时，应查看
`state.reward_terms` 中对应项的量级，而不只比较配置数值。

## 7. `sensor`：传感器名称映射

`Sensor` 将共享任务需要的状态映射到组装后场景中的 sensor 名称：

| 字段             | 期望输出                                                        |
| ---------------- | --------------------------------------------------------------- |
| `local_linvel`   | 3 维机体局部线速度                                              |
| `gyro`           | 3 维机体局部角速度                                              |
| `upvector`       | 3 维机体 Up-vector                                              |
| `foot_positions` | 按前左、前右、后左、后右顺序排列的四个 3 维足端位置 sensor 名称 |

`foot_positions` 必须包含四个非空且互不相同的名称。足端位置用于机体参考系下的摆动高度奖励，若机器人资产没有这些
sensor，应在场景配置中用 `FrameSensorCfg` 补充，而不是复制环境实现。

## 8. 地形、出生范围与回合长度

| 字段                    | 含义                                                     |
| ----------------------- | -------------------------------------------------------- |
| `ground_geom_name`      | 高度查询使用的平面或高度场 geom；内置 scene 使用 `floor` |
| `initial_base_position` | floating base 的默认世界坐标位置，单位为米               |
| `spawn_xy_range`        | reset 时 x/y 均匀采样的半宽；`0` 表示固定位置            |
| `max_episode_seconds`   | 达到该时长时产生 truncation                              |
| `ctrl_dt`               | 策略控制周期，同时也是奖励总和的时间缩放因子             |

内置平地与粗糙地形配置都使用 `spawn_xy_range=4.0`。粗糙地形配置继承完整的平地配置，仅覆盖
`scene`，将 `FlatTerrainCfg` 替换为 `QuadrupedWalkTerrainSceneAssetsCfg` 与 `HFieldTerrainCfg`。重置时环境在出生点
附近九个采样点中取最高地形高度并抬高基座；运行时的机体高度奖励也以局部地形为参考。

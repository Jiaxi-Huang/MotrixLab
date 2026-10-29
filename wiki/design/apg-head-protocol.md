# APG 解析头协议设计

## 摘要

**`motrix.apg`**：provider 只保留「世界模型想象 + 解析头协议」，全部任务知识下沉到任务包的
**解析头**（parsing head）。核心约束：**新增任务只改 `motrix_envs` + task yaml，
`motrix_rl_builtin/apg` 一行不动**；g1-supermimic（语料跟踪）与 go2-walk-flat（速度指令四足）
两个不同族任务是现行实证。

## 背景与动机

provider 若镜像任务语义（可微 FK、解析 reward/终止/观测/goal/critic head、context 布局、噪声表、
动作界限，以及对 motion command 缓冲与任务指标名的硬编码），新增任何任务都要改 provider。而
learner 机制（世界模型、策略、critic、想象 rollout、replay、sync/async trainer）
本身是维度注入的通用实现——任务知识集中在少数接触点，可以整体抽离到任务侧。
sync/async 采集与 play 回放三条路径统一消费 env 产出的 policy 观测（经 head 的
`split_observation` 拆分，采集期含 env 侧观测噪声）；解析 torch head 只服务想象期。

## 目标

- provider（`motrix_rl_builtin/apg`）零任务知识（grep 任务专名零命中，作为持续不变量）。
- 任务侧自持解析头：维度、可微 head、观测拆分、动作界限、env 缓冲读取、终止分类、课程采样状态。
- head 通过 task yaml 的点路径（`algo.head: "pkg.module:factory"`）声明，provider `importlib` 解析
  ——rslrl `class_name` 字符串先例；随 `task_config.yaml` 快照自然序列化，resume/play/WM worker
  全路径可用。
- head 对协议 duck-typing（不依赖 provider 的类型/训练逻辑）；共享的 torch 旋转/状态布局数学单一来源在
  `motrix_env_core.torch.geometry`（状态布局是 WM 与 head 的接口契约，不容双源），provider 与 head 都
  从它导入，依赖方向保持 env 包 → core，env 包不依赖 RL 插件。

## 非目标

- 不做 head registry / entry-point 机制：点路径字符串已满足零改动约束，不引入额外全局注册表。
- 不在 provider 内泛化 per-clip 评估：语料评估是任务语义，随任务 head 落在任务包。
- 不改 `motrix_rl` 控制平面契约（`AgentProvider`/`TrainerContext` 等照旧）。

## 协议：ApgTask

完整契约见 `motrix_rl_builtin/src/motrix_rl_builtin/apg/contracts.py` 的 `ApgTask` docstring。要点：

**维度与标量**（属性）：`num_joints`、`context_dim`、`obs_dim`、`goal_dim`（目标总宽——未来帧数或
准静态指令都是任务内部语义，算法侧只见宽度）、`max_step_reward`、`ctrl_dt`、`action_low/high`。

**可微 head**（torch，首维 batch，运行在想象 rollout 的 autograd 图内）：

- `obs(state, previous_action, context, frames)`：本体观测（噪声外置，见 `add_noise`）。
- `add_noise(obs)`：想象路径的观测噪声。
- `goal(frames, context)`：目标特征（宽 `goal_dim`）。context 携带重建想象期目标所需的一切（指令型
  任务的命令就藏在 context 里）；语料型任务按帧号索引、帧数是 head 内部语义。
- `reward(state, frames, action, previous_action, context)`：解析奖励。context 沿 rollout 恒定，
  provider 负责把它广播到「horizon × batch」的扁平想象批上。
- `soft_termination(state, frames)`：可微终止风险。
- `critic_features(state, frames, normalized_context)`：特权价值特征。
- `normalize_context(context)`、`ref_state(frames)`（形状探测用参考态）、
  `valid_imagination_start(frames, horizon)`、`to(device)`。

**采集钩子**（构造时绑定 env，NumPy 进出）：`current_transition() -> (s, frames, context)`、
`endpoint_state() -> s1`、`split_observation(full) -> (obs, goal)`、
`termination_counts(metrics) -> float`（head 用自己 env 的指标键分类任务失败）、
`sampling_state()/load_sampling_state()/sampling_metrics()`。

**两个刻意的不透明约定**：

1. **`f` 对 provider 完全不透明**。想象中 provider 只做 `f + 1`，语义由 head 定义：
   g1 = 语料帧号（受 `clip_end` 约束的起点有效性），go2 = 步态相位步数（`phase = fmod(f·dt·freq, 1)`）。
2. **观测拆分归 head**。go2 的 command 块在观测数组中部（后随 feet_phase），provider 不得定死
   `[proprio | goal]` 前后缀——拆分是任务知识（`split_observation`）。

**env 侧最小面**：`SupportsApg = num_envs / cfg / init_state / step`（`ArrayEnv` 基类表面，
Manager 与 Direct 工作流通用）；任务专属数据全部经 head 读取，head 只使用其 env 的公共表面
（如 go2 head 经 `QuadrupedWalkTask` 的 `velocity_commands`/`gait_phase`/随机化访问器读取，
不穿透私有状态）。

## 任务侧归置

- 解析头随任务族放置：`motrix_envs/locomotion/supermimic/apg_head.py`、
  `motrix_envs/locomotion/quadruped/apg_head.py`；per-clip 确定性评估（语料任务的评测语义）在
  任务目录的 `evaluate.py`，provider 只保留通用 `load_policy`。
- 共享的可微 FK（`ModelKinematics`，读 MotrixSim 模型构建 torch 运动链，仓库唯一实现）归
  `motrix_envs/locomotion/kinematics.py`，多任务 head 复用。
- `motrix_envs` 的 `apg` optional extra 依赖 `torch`（head 的旋转/状态布局数学来自
  `motrix_env_core.torch.geometry`）：仅当 task yaml 把 APG run 指到任务 head 时才需要，
  纯仿真用途不引入 torch。

## 双任务实证

| | g1-supermimic | go2-walk-flat |
|---|---|---|
| 任务族 | 动作语料跟踪（ManagerEnv） | 速度指令 locomotion（DirectEnv） |
| goal 来源 | 语料未来帧特征（宽随 env 的 `num_goal_frames`） | context 中的准静态速度指令（宽 = 指令维度） |
| `f` 语义 | 语料帧号 | 步态相位步数 |
| context | 摩擦 + 关节偏移 + CoM（1+N+3） | 指令 + 摩擦/质量/CoM/延迟 |
| world 态读取 | motion command 发布的 state 缓冲 | head 自编译 Link query（世界位姿/四元数/速度） |
| 奖励 | 跟踪核（FK 姿态/速度误差） | 镜像 walk reward 各项（解析步态相位 + 足端高度接触代理） |

go2 head 的想象期近似（任务侧决策，docstring 记录）：步态相位解析重建（不建模站立冻结与地形采样，
面向平地任务）；足端接触用体相对高度代理而非碰撞查询。

## 持续不变量

- 新增任务只改 `motrix_envs` + task yaml；`motrix_rl_builtin/apg` 与 `motrix_rl` 不动。
- provider 内 grep 任务专名（g1/go2/lafan/motion/supermimic 等）零命中。
- provider 单测内置第三方假 head，作为「任务侧零改动可实现协议」的常驻证明。
- 测试断言从 env 公共表面/公式派生（如 `obs_dim + goal_dim` = 策略观测宽度），不钉任务调参值。
- sync/async 采集与 play 三路径观测同源：消费 env 的 `obs.policy`，解析 head 只用于想象。
- 双 GPU WM worker（`trainer.wm_device`）默认关闭且无自动化覆盖；启用前须在双卡主机人工验证。

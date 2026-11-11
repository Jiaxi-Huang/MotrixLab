# supermimic 上游与 MotrixLab 移植对比

## 摘要

本文对比上游 [supermimic](https://github.com/HybridRobotics/supermimic)（本机 `~/code/supermimic`，基于 mjlab 的 G1 全身运动跟踪 world-model + policy 研究实现）与本仓库的对应落地 `g1-wbt-supermimic` 任务 + `motrix.apg` provider，覆盖任务定义（观测、动作、奖励、终止/截断、初始状态采样、运动数据、域随机化）与 RL 算法（策略损失、critic、世界模型、actor 网络、数据面与训练循环）。结论：MotrixLab 侧是上游 "世界模型 + 想象 BPTT" 配方的 provider 化移植，训练循环超参、reset 噪声、动作缩放公式、return 课程混合比等逐字一致；主要结构性差异在于目标观测（上游有未来帧窗口，本仓库只有当前帧）、域随机化上下文条件化（上游有、本仓库无）、多任务编码器 actor（上游有、本仓库单任务直 MLP）、截断/终止语义一等化（本仓库更细），以及想象侧任务数学的实现方式（上游两处镜像实现，本仓库双路径编译单源）。本文属研究对比记录，不定义接口。

对照的主要源码入口：

- 上游（仓库外，路径为行文参照）：`supermimic/representations.py`、`supermimic/tasks/{task_spec,env_cfgs}.py`、`supermimic/utils/tracking_reward.py`、`supermimic/envs/world_model_env.py`、`supermimic/learners/{policy_learner,wm_learner}.py`、`supermimic/runners/{supermimic_runner,ppo_runner,reset_sampling}.py`；其基础环境是 mjlab 的 `unitree_g1_flat_tracking_env_cfg`。
- 本仓库：[g1/supermimic.py](../../motrix_envs/src/motrix_envs/locomotion/wbt/g1/supermimic.py)、[wbt/mdp/](../../motrix_envs/src/motrix_envs/locomotion/wbt/mdp/)（term 实现）、[apg/learner.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/learner.py)、[apg/networks.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/networks.py)、[apg/config.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/config.py)、[motrix.apg.yaml](../../configs/task/g1-wbt-supermimic/motrix.apg.yaml)、[manager/modeling.py](../../motrix_env_core/src/motrix_env_core/manager/modeling.py)；设计文档见 [wm-bptt-trainer](../design/wm-bptt-trainer.md) 与 [apg-provider](../design/apg-provider.md)。

## 1. 两个代码库的关系

| | supermimic（上游） | MotrixLab（本仓库） |
|---|---|---|
| 定位 | HybridRobotics 研究代码：SuperTrack 思路的 world-model + policy 双训练，构建在 mjlab（MuJoCo 后端、IsaacLab 风格 manager 体系）之上 | MotrixSim 之上的通用 RL 框架（provider 插件 + 部署链）；`g1-wbt-supermimic` 是 WBT 任务家族的配置变体（53 行 delta），`motrix.apg` 是内置 provider |
| 任务 | `SuperMimic-GenCoord-G1` / `SuperMimic-MaxCoord-G1`，G1 29DoF 全身运动跟踪 | `g1-wbt-supermimic`（当前语料为 dance 目录占位） |
| 算法 | 世界模型 + 想象 BPTT（主路线 `SuperMimicRunner`）；MBPO 式想象 PPO（副路线 `PpoRunner`，rsl_rl） | APG = 同一 "WM+BPTT" 配方的 provider 化移植（单路线） |

MotrixLab 侧确为上游配方移植的出处证据：

- [apg/config.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/config.py) docstring 明说配方镜像 Uni-SuperMimic 的 `SuperMimicConfig`；[motrix.apg.yaml](../../configs/task/g1-wbt-supermimic/motrix.apg.yaml) 注释逐条标注 upstream 参数名。
- [motrix_env_core/math/kinematics.py](../../motrix_env_core/src/motrix_env_core/math/kinematics.py) 的可微 FK 注明 "Adapted from SuperMimic … Copyright (c) 2026 ETH Zurich"。
- [test_wbt_motion_command.py](../../motrix_envs/tests/test_wbt_motion_command.py) 的 return 采样分布测试注明 "the recovered SuperMimic mixture verbatim"。

逐字一致（同源移植证据）的部分：

- 14 个 tracked bodies 列表完全相同：pelvis、双腿 hip_roll / knee / ankle_roll、torso、双臂 shoulder_roll / elbow / wrist_yaw。
- 动作缩放公式 `scale = 0.25 · effort_limit / kp`（mjlab `g1_constants.py` 与本仓库 `motrix_env_core/mdp/action.py` 一致）。
- reset 噪声数值完全一致（见 §2.6）。
- 本体感知观测噪声半宽一致（0.05 / 0.2 / 0.01 / 0.5 / 0）。
- return 课程混合比 0.5 / 0.4 / 0.1 与 softmax(−25·V/g_max)。
- 训练循环超参：imagination horizon 32、WM 监督窗 8、WM batch 2048×10、actor batch 4096、warmup 500 iter 仅训 WM、utd 0.4（每 iter 2.5 步采集）、cosine 衰减至 3%、bf16。

以下各节重点讲差异。

## 2. 任务定义对比

### 2.1 环境与基础设施

| | 上游 | MotrixLab |
|---|---|---|
| 仿真后端 | mjlab（MuJoCo/MJX），manager-based | MotrixSim，自有 env_core manager 体系 |
| 语料格式 | mjlab `MotionLoader`（concatenated clips + `motion_ends` 索引） | 自定义 NPZ v1（源自 BeyondMimic 语料；fps=50 强制校验、一文件一 clip、全局帧轴拼接 + `frame_clip_end`），见 [motion-npz-schema](../design/motrixlab-motion-npz-schema.md) |
| 机器人 | G1 29DoF | UnitreeG129Dof（29 关节）✅ |
| 控制节拍 | ctrl 0.02s（50Hz，运动帧率一致） | sim dt 0.005 / ctrl_dt 0.02（decimation 4）✅ |
| episode 时长 | mjlab 基础 cfg + clip 尾部 clamp 续跑 | `max_episode_seconds=10` 上限截断 + clip 尾显式 truncation |

### 2.2 观测

上游 actor 观测 = 传感器本体感知 + 目标窗口（`supermimic/tasks/env_cfgs.py`、`representations.py`）：

- 传感器组 `[projected_gravity(3), base_ang_vel(3), joint_pos_rel(N), joint_vel(N), last_action(A)]`，可选 history 帧环（trainer 侧维护）。
- goal 组（关键）：未来 **10 帧关节目标窗**（stride 5 ≈ 0.1s 间隔，约 **0.9s lookahead**，每帧 `[qj(N), vj(N)]`，索引 clamp 在当前 clip 末帧）+ 锚点相对位姿 `rel_ori rot6d(6×10)`（actor 用）、`rel_pos(3×10)`（仅 WM/critic 用）。
- critic（特权）：`[rep_state, wm_goal, sim_ctx]`——模型态全量 + 域随机化上下文。

本仓库 policy 组（154 维，`wbt/mdp/observations.py`）：

`[motion_joint(qj+vj 当前帧, 58), motion_ref_ori_b(6D, 6), base_ang_vel(3), dof_pos_rel(29), dof_vel(29), actions(29)]`；10 帧历史栈在 trainer 侧组装（`actor.history=10`）。value 组（286 维）另加参考体位置差(3)、14 body 位置(42)、body 姿态 6D(84)、base_lin_vel(3)。

| 维度 | 上游 | MotrixLab |
|---|---|---|
| 目标前瞻 | **未来 10 帧窗口（~0.9s）+ 相对锚点位姿** | **仅当前帧** qj/vj + 参考体相对姿态 6D；无 future window、无 rel_pos |
| 姿态感知 | projected_gravity（重力向量） | 无重力向量（由参考相对姿态 6D 替代其角色） |
| 噪声 | 半宽 [0.05, 0.2, 0.01, 0.5, 0] | 同款半宽 ✅ |
| history | trainer 侧帧环（`history_length` 可配） | trainer 侧帧栈（history=10，对应 upstream `hist: 10`） |
| 多任务 | actor 为 motion + velocity 双任务多编码器（slot_mix 批内分槽） | 单任务单 actor（goal 经 obs 组单源） |

"无未来窗口"是最 structural 的任务差异：上游策略能看到即将到来的动作序列（长时程编排能力），本仓库策略只看当前瞬时目标。另注意 value 组的 84 维 body 姿态存在**故意保留的 bug 兼容行为**（`observations.py` 中 body 0 的 6D 矩阵元素被后续 body 当参考四元数使用，注释明令不得静默修复）。

### 2.3 动作

两边完全同构：29 维、位置控制、`目标 = raw · (0.25 · effort_limit / kp) + 默认姿态`（逐关节 scale）。本仓库额外显式定义对称动作 Box（归一化动作铺满关节控制范围），并有动作延迟框架能力（默认关闭）。

### 2.4 奖励

两边同族：高斯核 `w·exp(−err²/σ²)` 跟踪项 + 惩罚项，每步 ×ctrl_dt(0.02)，σ 全套一致（0.3 / 0.4 / 1.0 / 3.14），均无显式终止惩罚（early termination 隐式惩罚）。权重分配不同：

| 项 | 上游（mjlab 基础 + supermimic 修改） | MotrixLab WBT |
|---|---|---|
| 根/参考体位置 | 0.5，σ=0.3，**anchor=pelvis**（supermimic 从 mjlab 默认 torso_link 改为 pelvis） | **1.0**，σ=0.3，**参考体=torso_link** |
| 根/参考体姿态 | 0.5，σ=0.4 | 0.5，σ=0.4 |
| 身体相对位置 | 1.0，σ=0.3 | **2.0**，σ=0.3 |
| 身体相对姿态 | 1.0，σ=0.4 | 1.0，σ=0.4 |
| 身体线速度 | 1.0，σ=1.0 | 1.0，σ=1.0 |
| 身体角速度 | 1.0，σ=3.14 | 1.0，σ=3.14 |
| action_rate | **−0.1** | **−0.5**（G1 覆盖；WBT 家族默认 −1.0） |
| 关节限位 | **−2.0**（supermimic 从 −10 改） | **−10**（soft 0.9 · 半范围，cap 5.0） |
| 自碰/接触 | self_collisions 奖励被 supermimic 删除 | 无（`undesired_contacts` 亦删——接触力 query 类型级不可建模，见 [apg-provider](../design/apg-provider.md) P-F） |
| 正权重和 | 5.5 → 每步上限 0.11，v_max=11（γ=0.99） | 6.5 → 每步上限 0.13，g_max=13 |

奖励执行架构的深层差异：上游在 WM 环境里**重实现**了一份 tracking reward（`supermimic/utils/tracking_reward.py`，含 re-anchor 到预测 root 的 xy+yaw）——同一数学、两处代码、靠镜像对齐；本仓库通过**双路径编译**让真实 env 与想象侧消费**同一份 term program**（"任务数学单源"，[manager/modeling.py](../../motrix_env_core/src/motrix_env_core/manager/modeling.py)），想象侧 tracked body 状态由可微 FK 从物理块精确派生后再算奖励，数值一致性由 fixture 锁定。

### 2.5 终止与截断

| | 上游 | MotrixLab |
|---|---|---|
| 参考体 z 误差 | pelvis，阈值 **0.25** | torso_link，阈值 **0.5**（宽松 2×） |
| 参考体姿态 | 0.8 | 0.8 |
| 末端 body z | ee_body_pos 被 supermimic **删除**（mjlab 原为踝/腕 0.25） | `bad_body_z` **保留**（双踝+双腕 0.25，body 集合与 mjlab 相同） |
| 关节级保护 | 无 | `bad_dof_pos`（越界 0.5）、`bad_dof_vel`（100 rad/s） |
| clip 结束 | motion_times **clamp 在末帧继续跑** | `motion_clip_end` 显式 **truncation**（time-limit 语义） |
| 截断/终止区分 | **无**；WM 里单一软终止概率 `sigmoid((max(err/thr)−1)/0.05)` | **一等区分** terminated/truncated（截断对 lambda-return bootstrap，终止不 bootstrap） |

想象侧的终止处理都源自真实阈值：上游转成软概率（进 GAE 折扣与 survival masking）；本仓库直接用同一 term program 在想象态上算 terminated/truncated，再用 survival gate `(∏ ~(term|trunc))` 加权回报。

### 2.6 初始状态采样与 reset

- 采样模式：上游 mjlab 提供 uniform / adaptive / start，supermimic 新增 **return 课程**（ReturnSampler：0.5 uniform + 0.4 softmax(−25·V/v_max) 低回报聚焦 + 0.1 keep；γ=0.99，memory 256，horizon 20）。本仓库实现 uniform / adaptive + **逐字复刻的 return 课程**（g_max = dt·Σw⁺/(1−γ) = 13 动态计算，测试锁定公式）。
- **当前实际状态**：`g1-wbt-supermimic` 配置 `sampling_mode="uniform"`，**return 课程代码在但训练配置未启用**（全仓库无 override；任务文件注释说"训练配置切 return"但 yaml 没切）。
- reset 噪声数值逐字一致：root pos ±(0.05, 0.05, 0.01)、Euler ±(0.1, 0.1, 0.2)、线速度 ±(0.5, 0.5, 0.2)、角速度 ±(0.52, 0.52, 0.78)、关节位置 ±0.1（mjlab `tracking_env_cfg.py` vs 本仓库 `wbt/mdp/reset.py`）。
- 想象起点：两边都从 replay 采样初始状态（RSI 式）。上游带 H−1 历史帧（`sample_causal` + pad_mask，padding 槽共享同一噪声实现）；本仓库 `SequenceReplay.sample_starts` + history 向 episode 头钳位补帧。

### 2.7 运动参考数据

上游 `MotionLoader` 与本仓库 NPZ v1 都做纯当前帧 gather、不插值。差异在上游查询支持未来窗口索引（clamp 到 clip 末帧）；本仓库以 `frame_clip_end` 显式记录每帧所属 clip 的末帧索引。本仓库当前语料是 dance 目录占位，supermimic 目标语料尚未入 assets。

### 2.8 域随机化与模型上下文

- 上游保留三类 DR：`foot_friction / base_com / encoder_bias`，并把 DR 参数打包成 **sim_ctx 作为 Dynamics 网络与 critic 的额外输入**（上下文条件世界模型，robotic_world_model 风格）——WM 学的是"参数化动力学"。
- 本仓库 WBT **完全没有 events/DR**，dynamics 无上下文输入，外生通道目前只有 `steps`（INCREMENT 规则）——WM 学的是标称动力学。若后续引入 DR，需扩展外生通道注记体系（见 [manager/modeling](../design/manager/modeling.md)）。

## 3. RL 算法对比

### 3.1 算法族谱

- 上游两条路线：① `SuperMimicRunner`（主）：真实采集 → replay → WM 监督训练 → **BPTT 策略学习**（想象 32 步直接对回报反传；actor-critic 变体加 critic bootstrap + GAE critic 拟合）；② `PpoRunner`（副）：同样采数据训 WM，然后在想象 rollout 上跑 rsl_rl PPO（MBPO 风格）。
- 本仓库 APG：单路线，等价于上游 ① 的 actor-critic 变体移植。**两边 BPTT 路线都没有 PPO 组件**（无 ratio / clip / KL / entropy）；探索都靠采集期高斯噪声（上游 `cfg.noise=0.1`；本仓库 `exploration_noise_std=0.1×action_limit` + clip）。

### 3.2 策略损失（BPTT）

两边同构：`G = Σ γ^t·(∏gate)·r_t + γ^H·(∏gate)·V(s_H)`，gate 为生存门控；梯度契约相同（unroll 前冻结 dynamics/critic 参数、梯度经输入回传到 actor，事后恢复）。

| | 上游 | MotrixLab APG |
|---|---|---|
| γ | 默认 **1.0**（未折扣 + 纯生存屏蔽），AC 变体 0.99 | 0.99 |
| 梯度裁剪 | per-parameter value clip **0.1** | max_norm 1.0 + value clip **0.1**（连上游硬编码常量一起移植，`apg/learner.py`） |
| actor 优化器 | Adam(0.9, **0.95**) | 同 ✅ |
| lr 调度 | cosine → 基础 lr 的 3% | 同 ✅ |

### 3.3 Critic

| | 上游 | MotrixLab |
|---|---|---|
| 回归目标 | **GAE**（δ 的指数加权和）on 想象 rollout | **TD(λ) 树形 lambda-return**（`target_t = r + γ·gate·[(1−λ)V + λ·target_{t+1}]`）——同为 λ-return 家族，有限 H rollout 上组合方式不同 |
| 输出裁剪 | `clamp(max=v_max)`，v_max = max_step_reward/(1−γ)（bootstrap 与目标都裁） | **无 v_max 裁剪** |
| 训练 | 5 epochs × 4 minibatches MSE | 同 ✅（λ=0.95 同） |
| 输入归一化 | critic 输入无独立归一化 | 独立 EmpiricalNormalization，**从想象观测更新**（对应上游 critic norm 角色） |

### 3.4 世界模型

| | 上游 Dynamics | MotrixLab FloatingBaseDynamics |
|---|---|---|
| 网络 | MLP [2048]×3，LayerNorm+GELU，EmpiricalNormalization(state, action) | MLP [2048]×3，SiLU+LayerNorm，**零初始化输出层**，`_RunningNorm`(in/out)（eps 1e-2/1e-4，2e8 样本冻结） |
| 预测量 | **加速度级 delta**（gencoord `[Δv_b, Δw_b, Δvj]`；maxcoord 每体 `[Δv, Δw]`），解析**半隐式积分**（substep_coeff=(d+1)/2d 匹配仿真子步） | **frame-local 增量**（yaw-frame 位置增量、so(3) 旋转向量、body-frame 速度增量、关节增量），exp/map 精确分支 |
| 表示 | 两种：gencoord（关节空间）/ maxcoord（每刚体 + analytic IK 反解关节） | 主方案 floating_base（≈gencoord 强化版）+ residual 备选；**maxcoord/IK 路线未移植** |
| tracked bodies | gencoord 经 FK 派生（算奖励用） | **可微 FK 精确派生进 sim_state**（"WM 只学动力学、运动学不学"，奖励/终止直接消费） |
| 监督损失 | K=8 步 unroll，逐通道组 **Bryson 权重**（1/nominal²）MSE | K=8 步 unroll，**目标局部系度量** + error_floor 逐块下限（0.003/0.005/0.02/0.05/…）——同为"名义误差归一"思想，实现不同 |

### 3.5 Actor 网络

- 上游 `MultiEncoderAE`：每任务 goal encoder → **tanh latent 瓶颈(32)** + 共享 decoder [2048,1024,512,256]；latent BC 辅助损失（0.1）拉齐非锚任务编码器；slot_mix 批内分槽同时训练 motion + velocity 任务。
- 本仓库 `MlpActor`：纯 MLP [2048,1024,512,256]，identity 输出，输入为展平的 10 帧 obs 栈。**无 latent 瓶颈、无多任务**（[apg-provider](../design/apg-provider.md) 明确不回收 GoalConditionedActor，goal 经 obs 组单源）。

### 3.6 数据面与训练循环

循环配方两边精确对齐：短 on-policy ring（1000 步/lane）→ 分数步采集（utd 0.4 → 每 iter 2.5 步、2/3 交替）→ WM 10 batch × 2048 → actor 4096 × 1 次更新 → warmup 500 iter 仅训 WM → cosine → 3% → bf16。

差异在工程质量：

- **episode 边界**：本仓库有 **FinalState 终态捕获**（done 行 auto-reset 前的真终态入 replay，含外生终值，见 [array-env-final-observation](../design/array-env-final-observation.md)）+ `SequenceReplay` 的 valid/intact 掩码 + terminated/truncated 分列 schema（见 [rl-sequence-data-plane](../design/rl-sequence-data-plane.md)）；上游靠 pad_mask / episode 链。
- **可恢复性**：本仓库学习状态（模块/归一器/优化器/RNG/计数器）精确恢复（已知偏差：sampler 课程统计不序列化），replay 重新 warm-up；上游为研究代码无此契约。
- **部署**：本仓库 ONNX 导出把 obs 归一化 + action_limit 缩放折进图内（parity 校验），直达 MuJoCo / MotrixSim / Unitree 硬件链（见 [onnx-export](../design/onnx-export.md)）；上游 exporter 处理 term-major → temporal 转换。
- **性能技巧取向**：上游 CUDA graph 捕获更新 + torch.compile；本仓库 FP32 数学 + MLP bf16（amp）、collector pinned-memory 异步拷贝；async 拓扑已预留（wm-bptt-trainer Phase 4，未实现）。

## 4. 差异汇总表

| 维度 | supermimic 上游 | MotrixLab | 性质 |
|---|---|---|---|
| 目标观测 | 未来 10 帧窗（0.9s lookahead）+ 锚点相对位姿 | 仅当前帧 + 参考体相对姿态 6D | **结构性** |
| 姿态感知 | projected_gravity | 无 | 结构性 |
| 参考体/锚点 | **pelvis** | **torso_link** | 数值 |
| 奖励权重 | 根位置 0.5 / 身体位置 1.0 / action_rate −0.1 / 限位 −2 | 1.0 / 2.0 / −0.5 / −10(cap 5) | 数值 |
| 终止 | pelvis z 0.25 + 姿态 0.8；clip 尾 clamp 续跑 | torso z 0.5 + 姿态 0.8 + 踝/腕 0.25 + 关节级检查；clip 尾显式 truncation | 语义 |
| 截断/终止区分 | 无（软终止概率） | 一等区分（影响 bootstrap） | 结构性 |
| 域随机化 | foot_friction / base_com / encoder_bias → **sim_ctx 条件化 WM** | 无 DR、无上下文输入 | **结构性** |
| 多任务 | motion + velocity 双编码器 + latent BC | 单任务 | 结构性 |
| Actor | latent(32) 瓶颈 + 共享 decoder | 直 MLP | 结构性 |
| Critic 目标 | GAE + v_max 裁剪 | TD(λ)、无裁剪 | 算法细节 |
| WM 预测量 | 加速度 delta + 半隐式积分 | frame-local 增量 + FK 派生 | 实现路线 |
| WM 表示 | gencoord / maxcoord+IK 双路线 | floating_base / residual | 范围 |
| 想象侧任务数学 | WM env 里重实现（两处代码） | 双路径编译单源（fixture 锁一致） | **架构改进** |
| return 课程 | 默认启用 | 已复刻**未启用**（uniform） | 配置状态 |
| 语料 | — | dance 占位 | 资产状态 |

## 5. 值得注意的差异与潜在影响

1. **未来窗口缺失**是任务侧最大差异：上游策略可预读 0.9s 的动作编排，对长时程/快节奏语料表达力更强；本仓库当前帧目标对连续性好的 dance 语料可能够用，但移植上游完整语料后建议评估是否补 future window。
2. **sim_ctx 条件化缺失**：上游 WM 对 DR 参数条件化，本仓库 WM 学标称动力学——若后续加 DR，外生通道体系（目前仅 `steps`）需扩展，否则 WM 会在非平稳动力学上欠拟合。
3. **return 课程未启用**：代码路径与上游逐字一致但 yaml 没开，当前训练用 uniform 起点采样——与上游配方不等价，需确认是有意为之还是遗漏。
4. pelvis vs torso_link 锚点、终止阈值 0.25 vs 0.5，直接影响 episode 长度分布与 v_max/g_max 标定（11 vs 13），做对比实验时需注意。
5. critic 无 v_max 裁剪 + TD(λ) vs GAE 的组合差异，理论上影响价值回归的稳定性（上游用裁剪抑制价值高估）。
6. 本仓库的双路径编译（任务数学单源）相对上游"两处实现对齐"是明确的架构改进；maxcoord 表示、多任务编码器、velocity 指令任务属于上游有而本仓库未移植的能力。

## 6. 对齐落地记录（2026-11-11）

按本文差异清单对 `g1-wbt-supermimic` 变体与 `motrix.apg` learner 做了第一轮对齐：

**已对齐**（任务侧全部落在 supermimic 单文件变体，家族基线零改动）：

| 项 | 落点 |
|---|---|
| 未来 10 帧目标窗（stride 5，~0.9s lookahead，`[qj,vj]×10`） | 新 obs term `MotionFutureJointObsCfg`，policy/value 组 |
| 锚点相对姿态窗 `rel_ori 6D×10` | 新 obs term `MotionFutureReferenceOrientationObsCfg`，policy/value 组 |
| 锚点相对位置窗 `rel_pos 3×10`（特权） | 新 obs term `MotionFutureReferencePositionObsCfg`，value 组 |
| projected_gravity(3)，噪声 0.05 | 新 obs term `ProjectedGravityObsCfg`，policy 组 |
| 锚点 pelvis（自 torso_link） | `reference_body_name="pelvis"` |
| 奖励权重 0.5 / 1.0 / −0.1 / −2.0 | 变体重建 rewards dict |
| 终止：ref z **0.25**（pelvis）+ ref ori 0.8；ee-body-z 删除 | 变体重建 terminations dict |
| return 课程启用（上游默认） | 变体 `sampling_mode="return"`（混合比 0.5/0.4/0.1 为家族默认） |
| critic v_max 裁剪（bootstrap 与目标链） | learner `_value_of` clamp；v_max = ctrl_dt·Σw⁺/(1−γ) 自奖励权重动态计算 |

观测维度变为 policy 733 / value 889；全部窗口/重力 term 经双路径派生（`modeling_view()` dry-run 通过——窗 gather 是 steps 外生通道 + clip 共享常量的纯索引）。

**保留不对齐**（有明确理由）：terminated/truncated 一等区分与 clip 尾 truncation（优于上游 clamp 续跑，D5 冻结）；TD(λ)（同 λ-return 族）；单任务直 MLP actor（GoalConditionedActor 不回收决策）；γ=0.99（上游 AC 变体）；家族关节级保护 term 保留（bad_dof_pos/vel，仿真防爆护）。**待条件成熟再议**：sim_ctx 域随机化条件化（当前无 DR 基建）、多任务编码器、maxcoord 表示、目标语料入库。

### 第二轮（外部评审后，同日）

外部逐项核对确认第一轮对齐成立，另指出三处待决；已全部处置：

1. **goal 窗不再入历史栈**（结构性，评审主发现）：上游历史栈只作用本体感知。落地为
   `apg.goal_terms`——policy 组仅含本体感知（projected_gravity / ang_vel /
   dof_pos / dof_vel / actions，93 维 ×10 栈），goal 窗（未来关节目标 +
   锚点相对姿态，640 维）作为 value 组内具名 term 列切片，平铺单份拼进
   actor 输入（列范围自 `env.observation_groups` 布局按名推导，任务零硬编码）；
   goal 独立归一器，play/ONNX 导出同步拆分输入契约（部署侧需按
   `[proprio 栈 | goal]` 供数）。actor 输入回到上游量级（~1570 vs 7330）。
2. **未来关节窗改逐帧块布局**：`[qj(N), vj(N)]×帧` 拼接（此前为逐关节标量
   交错，信息等价但与上游非逐字）。
3. **v_max 裁剪范围对齐上游**：只裁 bootstrap 与回归目标，TD(λ) 递归内
   per-step value 不裁（此前裁得更强）。

实施中另自查修复一处评审未覆盖的 unroll 时序 bug：想象步此前在
pre-dynamics 态上 evaluate/observe 且起点 obs 重复入栈，已改为
dynamics 先行、reward/obs 落在 s_{t+1}（与真实 env 的 post-physics
evaluate 时序一致）。

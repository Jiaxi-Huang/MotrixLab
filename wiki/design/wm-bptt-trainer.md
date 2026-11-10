# APG（WM/BPTT）Trainer 设计与路线图

## 摘要

在现有 provider 插件机制与 schema 驱动传输层（#160/#162）之上，分阶段支持 APG
（world model + 通过冻结 WM 的策略 BPTT）trainer。方案分三层：

- **环境侧课程基建**（`motrix_envs/motion`）：motion 起点采样器统一为
  `sampling_mode`（uniform / adaptive / return）驱动的 torch 原生 sampler 族，
  多 clip 感知（per-clip bins），课程状态从共享 `WbtMotionCommand` 迁入 sampler 对象；
- **环境侧建模能力（派生式，无新增 modeling 包）**：任务数学"一次编写、两处
  消费"——manager 执行器把同一份 term program **双路径编译**（真实路径绑定常驻
  `ManagerContext`，想象路径绑定 modeling state 支撑的 shadow context），
  modeling view（字段集 / 外生规则 / 想象调度）全部从既有声明 read 集与 cfg
  注记**派生**；配 FinalState 终态捕获与一等 truncation 语义；
- **APG trainer**（`motrix_rl_builtin.apg`）：任务无关的通用 trainer（任何可
  派生 modeling view 的任务即可训练），sync 先行，async 复用
  `motrix_rl.transport`。

任务特化（supermimic / WBT 细节）留在 `motrix_envs` 任务层。roadmap 按五个 phase
推进——课程 / 建模 / 序列数据三线并行，汇合于通用 trainer——每个 phase 由相互
独立、可分别合入的 commit 组成；packed 多 clip 传输优化作为非核心的后续项。

## 背景与动机

### 框架现状

`motrix_rl` 控制平面对新算法已经足够开放：`TrainerBase` 契约只有
`train()/play()`（`contracts.py:21-30`），trainer 完全拥有自己的循环；provider 经
entry-point `motrix_rl.frameworks` 发现，`config_type` 自动安装 Hydra schema；
`TransitionSchema` / `HostTransitionRing` / `InferenceBundleSpec` / weight channel /
`AsyncTrainerTopology` 全部由 provider 声明、框架承载。env 层已有特权观测通道
（`EnvObs.value`）与稀疏终态观测（`FinalObs`）。

真正的缺口属于"序列数据训练器"这一类：无循环状态推理、transition 只有单步语义、
n-step 采样烧死在 fastsac 私有 buffer、async collector/learner 编排无共享设施。

### 环境侧现状（采样与课程）

多 clip 基础设施（#156）已把多个 clip 拼为带 `frame_clip_end` 边界的全局帧轴。
但 adaptive 起点采样存在三处结构性问题：

1. **课程状态驻留共享 term**：`adaptive_bin_failed_count` / `sampling_cdf` 等
   十余个张量与超参散落在 `WbtMotionCommand` 上（`wbt/mdp/command.py:104-126`），
   新增课程（如 return-based）只能继续往共享 term 上加状态；
2. **两份重复数学**：`motion/sampler.py` 的 numpy `AdaptiveTimestepsSampler` 与
   command reset/advance hook 内的 torch 图内实现是同一算法的两份代码；
3. **bins 跨 clip 混叠**：`num_bins = total_frames // env_fps + 1` 按全局帧轴分箱
   （`command.py:321`），多 clip 时一个 bin 可横跨两个 clip，失败归因与采样语义失真。

### 前一尝试的教训（作为设计约束）

MotrixLab_mbrl_manager 工作树验证了"学习式 WM + 想象 BPTT"管线可以端到端运行，
算法核心（learner/replay/networks）任务中立、可回收；架构问题集中于四处，构成
本设计的负面约束：

1. 课程状态（return sampling 的 11 个张量）驻留共享 `WbtMotionCommand`；
2. adapter 杂物间：YAML 携带 python path 绑定、评估协议靠 deepcopy + 突变 cfg、
   采样状态 checkpoint 挂 adapter；
3. 私有平行传输栈重造了主干刚落地的 `motrix_rl.transport`；
4. 框架缺口用任务侧 override 糊：`_update_truncate` 覆写、绕过声明式 read 直读
   `env.sim_data`、私有归一器不满足 `NORM_KEYS` 契约。

由此得出框架改动的**准入标准**：每个框架层改动必须以"任何实现对应契约的任务 /
trainer 都需要"来说明；任务名、超参、课程机制不得跨层；已有通用机制必须复用
而非平行重建。本设计不逐点沿用该尝试的形态，只回收其任务中立且契约干净的部分。

## 目标

- APG trainer 以标准 provider 插件接入（`motrix.apg`），`TrainerBase` 契约零改动；
- trainer 任务无关：通过 modeling 契约支持多任务（g1 WBT 与 go2 四足为前两个实例）；
- 起点采样课程成为 sampler.py 内可插拔、多 clip 感知的一等机制，uniform /
  adaptive / return 同协议；
- 任务数学单源（real env 与 imagination 消费同一实现）成为 env 侧一等契约，
  机制为派生式双路径编译（任务侧零 python，新增 APG 变体 = 单文件 cfg）；
- **任务家族无缝并列**：g1-supermimic 与 g1-wbt-dance / g1-largebox /
  g1-wbt-backflip 同层并列（共享 family common cfg 与 motion 基建，仅叠加
  APG 专属声明），go2 同理；
- 框架层需求以通用机制解决（双路径编译、FinalState、truncation），不做任务特制；
- 每个 phase 由独立可合入的 commit 构成，phase 间允许并行开发。

## 非目标

- 不做通用"想象引擎"运行时或多算法 WM 抽象——只实现 APG 配方（BPTT through
  WM + lambda-return critic）需要的能力；
- 不引入有状态 actor 推理（APG actor 是无状态 MLP over modeling state；
  `InferenceBundleSpec` 的 section 设计保留了加 state section 的位置，有真实需求再动）；
- 不做全面的通用 async trainer 骨架提升——只把 fastsac 与 apg 共同需要的部分留在
  `motrix_rl`，编排归各 provider；
- packed 多 clip 传输优化不在主线内（Phase 5 后续项）；
- 不为假想的未来任务预先扩展契约（遵循仓库编码规范：接口通用、实现范围由当前
  需求驱动）。

## 架构

### 分层与职责

```text
motrix_envs (任务层)
  motion/sampler.py：MotionStartSampler 族（uniform/adaptive/return，多 clip）
  g1 WBT / go2 任务包：家族共享 term 数学、motion 基建、单文件 APG 变体 cfg
        │  消费框架契约（声明 read 集 + 外生注记，APG 能力派生）
motrix_env_core (框架)
  manager 执行器内化的双路径编译与派生 ModelingView（无独立 modeling 包）
  FinalState 终态捕获（FinalObs 契约泛化，见"转换捕获与传输定案"）
  Manager truncation term group、外生通道 cfg 注记、ctx.reward mirror、
  math/rotation、target_offset、device-mobile FK
motrix_env_motrixsim
  read-program 声明驱动的 post-step FK freshness
motrix_rl (控制平面)
  WM TransitionSchema 模板、episode 边界正确序列回放（共享模块）、
  NORM_KEYS 归一器契约、_METRICS_FAMILIES 注册
motrix_rl_builtin (provider)
  MotrixFramework 聚合 (FastSacProvider, ApgProvider)
  apg/：config、networks（MlpDynamics / FloatingBaseDynamics）、learner、policy、
  sync trainer → async trainer
```

### 关键设计决策

**D1 — Trainer 契约不扩展。** `TrainerBase.train()/play()` 已被 fastsac sync/async
两路验证足够。APG trainer 通过 `TrainerContext` 消费 run 目录、checkpoint、metrics，
与 fastsac 同构；多任务能力来自"trainer 只依赖派生 modeling view"，不来自契约膨胀。

**D2 — 起点采样统一为 sampler.py 的 torch 原生 sampler 族。**
`motion/sampler.py` 定义 `MotionStartSampler` 协议：状态为 torch 张量（可作
kernel_data 嵌套成员，hook 内方法调用由 Dynamo trace），三个实现：

- `UniformTimestepsSampler`：无状态，全局帧轴均匀采样（等价现
  `sampling_cdf.numel() == 0` 路径）；
- `AdaptiveTimestepsSampler`：失败直方图 EMA + 指数核平滑（数学同现实现），
  **泛化为 per-clip bins**——`(num_clips, max_bins)` 统计、按 `frame_clip_end`
  归属 clip，消除跨 clip 混叠；
- `ReturnTimestepsSampler`：per-(clip, bin) 起点回报统计 + per-lane 折扣回报累积
  （advance 内读 `ctx.reward`），episode 结束折入统计，采样偏向低回报起点
  （数学从前一尝试 `return_sampling.py` 回收），多 clip 原生。

协议 hook 与现有 kernel 对齐：`fold_episode_stats(...)`（reset kernel 开头，compact
行）、`rebuild(...)`（重建采样 CDF）、`sample(draws, ...)`（reset 行抽起点，
**RNG draw 数为每模式固定常量**，保持现有 `count = 3 if cdf else 2` 式约定）；
return 模式增加 `accumulate_step_rewards(ctx)`（advance 内）。配置侧
`adaptive_sampling_enabled: bool` 替换为 `sampling_mode: str ∈ {uniform, adaptive,
return}`，adaptive 超参（uniform_ratio/alpha/kernel_*）归 adaptive 实现所有。
`WbtMotionCommand` 瘦身为持有单个 `sampler` 嵌套成员并委托；numpy
`AdaptiveTimestepsSampler` 删除（两份重复数学收敛为 sampler.py 一份）。

**D3 — 任务↔trainer 绑定走派生能力探测，不走字符串路径。** APG 可训练性是 env
cfg 的派生属性：`ManagerEnv` 基类提供 `modeling_view()` 派生访问器（构造期
dry-run，无需子类 override）。sync trainer 直接问 env 实例；async learner 沿用
fastsac 的 1-env probe 先例。YAML 中不出现 python path；派生失败的任务配 APG
trainer 在配置期显式报错。

**D4 — 转换捕获：终态捕获 + provider 侧装配，不引入 recorder。** 逐字段分析
（见[转换捕获与传输定案](#转换捕获与传输定案)）表明，唯一必须发生在 env 内的
捕获是 done lanes 的 pre-reset 终态——auto-reset 会销毁后继；action 由 collector
自持，活 lanes 的后继可从 post-step 声明 read 数据直接读取。定案：泛化既有
`FinalObs` 契约为 `FinalState`（覆盖全部 done lanes、声明 read 字段与常驻
command/action term state 行），转换装配在 provider 侧完成，sync / async 共用
同一条装配路径；进程间传输沿用 `motrix_rl.transport`（WM `TransitionSchema`），
episode 边界即 `dones`，序列窗边界在 learner 侧 mask。

**D5 — Manager truncation 一等化。** Manager cfg 新增 `truncation` term group
（dispatch 返回 `(N,)` bool，OR 进 `truncated`），与 `ArrayEnvState` 既有的
terminated/truncated 两维对齐。clip 结束是时间性截断而非失败，用于正确区分
lambda-return 的 bootstrap 语义。删除任务侧 `_update_truncate` 覆写。

**D6 — 序列回放提升为 `motrix_rl` 共享模块。** per-lane device-resident ring、按
episode_id/step_id 切窗、边界与 ring-wrap 感知的采样器（从前一尝试回收泛化）。
fastsac 的 n-step `sample()` 不强制迁移。

**D7 — 归一器对齐 `NORM_KEYS`。** APG 的 running obs normalizer 满足
`inference_bundle.py` 的 normalizer section 契约（mean/std/var/count 状态字典），
async 权重同步走既有 `WeightChannelShared`。

**D8 — checkpoint 语义不扩种。** actor 导出为 `POLICY`（无状态 actor 直接复用现有
ONNX exporter 与 parity 校验）；WM/critic/optimizer/normalizer/序列回放采样器/
sampler 课程状态全部归入 `TRAINING_STATE`。课程状态随 sampler 对象序列化，
trainer 不感知其内部结构。

**D9 — 评估协议是 trainer 配置，不是框架概念。** APG cfg 内声明 evaluation
profile（deterministic 覆盖项 + seed + 全库批量评估参数），由 trainer 应用为
eval env cfg 变体；不引入框架级 curriculum / evaluation 机制。

**D10 — 任务数学单源与可微 FK 上游化。** 单源通过派生式双路径编译实现
（同一 dispatch、双 ctx 绑定），契约细节见
[任务侧契约](#任务侧契约派生式-modeling-view)。可微 FK 的
device-mobile 能力（前一尝试以 `TrackedChain` 桥接 device-pinned 的 core
KinematicTree，#163）上游化进 core，任务层不再维护桥接。

### 任务侧契约（派生式 modeling view）

APG 可训练性是 env cfg 的**可派生属性**，不是任务要实现的第二套定义：env_core
不新增 modeling 包，任务作者不为 APG 写 python——新增一个 APG 变体任务 = 新增
一个与 largebox 同构的单文件 cfg。

#### 设计原则

1. **派生而非定义。** modeling 的一切——字段集、双路径 program、action 映射、
   起点过滤——从既有声明派生：read 集（`SimQueriesCfg`）、term 集、cfg 注记、
   action process。
2. **数学单源 = 同一 term 双路径编译，不人工拆分函数。** term 的 dispatch 函数
   是唯一数学源；真实路径与想象路径的差异只在 **ctx 绑定**（常驻
   `ManagerContext` vs modeling state 支撑的 shadow context）。拒绝失败案例的
   "单一 reward term 委托 `task.reward_of`"路线，也不采用逐 term 人工拆分
   gather / 纯数学的方案。
3. **语义类型来自 read query 类。** `SimQueriesCfg` 的 query 是语义类型化的
   具体类（`JointPositionQuery` / `LinkQuaternionQuery` /
   `LinkAngularVelocityQuery` / `RaycastCameraImageQuery`…，`sim/read.py`）——
   modeling 字段集与其语义（关节 / 旋转 / 速度 / 接触 / 图像）直接派生，
   不可想象类型被类型系统自然识别。新增 `SimDataQuery` 子类时声明可否想象
   是框架级义务。
4. **动力学结构归 trainer。** floating-base / so(3) 分解是 WM 的网络结构选择，
   属 apg provider：按字段语义类型自动组装，cfg 级覆盖兜底。env_core 不出现
   DynamicsLayout。
5. **权重与参数单源于 env cfg。** 想象路径聚合 reward 时读取 env cfg reward
   group 的权重表（派生时提取，不复制数值）。

#### 双路径编译（内化于 manager 执行器）

resolver 产出的 `ResolvedManagerProgram` 是纯数据（typed call records + 参数
描述符），ctx 由 Torch 执行器在构建 kernel 时绑定——**同一份 call records 可以
绑定到第二个 shadow ctx 再编译一次**。机制：

- **v1 不引入显式标记**（`@modeling` / `@differential` 均不做）：执行器直接从
  既有 program 尝试派生 modeling program，dry-run 成功与否即 capability——
  任务侧零足迹；显式意图标记留作后续可选项（出现真实混合任务时再加）；
- shadow `ManagerContext` 的 `sim` / `commands` / `actions` map 是 modeling
  state 张量束的寻址视图（字段名 ≡ 声明 read 集，query 类型提供语义）；
- command `update` hook 一并参与双路径（WBT 的 yaw-align 读 root pos/quat，
  均在 read 集内）；command term 的 **mutable kernel_data 走 shadow 拷贝、
  常量成员（clip 数据、FK chain）共享只读**——可变/常量的区分规则在 P2 评审
  冻结；
- 派生粒度 **group 级 all-or-nothing**：reward 组 / 各 observation 组 /
  termination 组要么整体可派生，要么整体不可想象（混合会使加权和残缺、观测
  不全）；
- shadow `rand` 走确定性模式（obs 噪声关闭语义）；modeling program 先 eager
  保证正确性，`torch.compile` 为可选优化（独立编译域，guard 落在 shadow 宿主
  对象 identity 上，与真实 kernel 编译缓存互不干扰）。

#### 外生通道：cfg 注记，不新模块

封闭规则集挂在**既有 term cfg** 的声明式字段上，落在 family 共享 cfg 一次，
不落每个任务：

| 规则 | 语义 | 注记落点 | WBT 例 |
|---|---|---|---|
| `increment` | 每步递增（想象侧同样推进） | `CommandCfg` | motion phase（帧索引） |
| `copy_action` | 由 policy action 决定 | `CommandCfg` | velocity command 类 |
| `constant` | episode 内常量、reset 时重采样 | reset / randomization term cfg | friction / offset / COM |

外生规则与真实 env 的 command term `advance` 语义**等价**（fixture 测试数值
对齐）；WBT 的 clip 结束在想象侧映射为 truncation（D5）而非重新采样——
resample 是采集侧课程关注点，不属于想象语义。

#### trainer 消费面（派生 view）

`ManagerEnv` 基类提供派生访问器（如 `modeling_view()`，dry-run 时机在此；
无需 env 子类 override）：

```python
class ModelingView:                     # manager 包内的派生数据类，非任务实现
    fields: Mapping[str, FieldDesc]     # 字段名 → 语义类型（read query 类）+ shape
    exogenous: Mapping[str, ExogenousRule]
    program: ModelingProgram            # reward / obs / termination / command-update 双路径调度
    action_process: Callable            # 复用 env.action_terms[].process（同一函数对象）
```

trainer（apg provider）消费 view：networks 按 `FieldDesc` 组装动力学结构；
learner 用 program 在想象态上算 reward / obs / survive；`valid_start` 默认由
learner 按 episode bookkeeping 推导（过滤跨 episode 起点），任务级覆盖留待
真实需求出现。

#### 与现有任务的关系

任务分三种形态，迁移只有单向的"升格"，没有破坏性变更：

1. **未升格任务（默认）**：零改动。model-free 训练（fastsac/PPO）不受任何
   影响；配置 APG trainer 时派生 dry-run 失败，配置期显式报错（D3）。
2. **升格任务**：**只加 cfg 注记**（外生规则字段），term 函数体与结构不动——
   现有 WBT 家族 term 已是"读 ctx + 纯数学"形态，其访问字段在 read 集内即可
   派生。env cfg 的 term 结构、注册 id、model-free 可训练性全部保持；升格后
   **同一份 env cfg 同时可跑 model-free 与 APG**（双路径消费同一 dispatch 与
   同一权重表）。
3. **APG 变体任务**（如 g1-supermimic、go2 四足）：与 g1-wbt-dance /
   g1-largebox / g1-wbt-backflip 完全同构的**家族单文件变体**——继承 family
   common cfg（`wbt/g1/common.py` 模式），共享全部 term 数学、motion 基建与
   sampler；文件内容只有 cfg 声明（truncation group、reset 随机化即 `constant`
   注记、`sampling_mode` 课程、eval profile（D9）），注册独立 env id。CLI 侧
   `task=g1-supermimic/motrix.apg` 与 `task=g1-wbt-dance/motrix.fastsac`
   同层并列。

Direct 工作流任务（如 go2 `walk_np`）的关系：共享 family 数学（Direct 的
`compute` 调共享函数），但 APG 采集侧需要满足建模契约（声明 read 集与注记），
因此 Direct 任务家族配一个 Manager 形态的并列变体 env（仓库本就支持
Direct/Manager 双工作流；这是同一配方的两种前端，不是第二任务体系）。

#### 任务包结构（单文件）

```text
motrix_envs/locomotion/wbt/g1/
  common.py                              # 家族基线 cfg（既有）
  dance.py / largebox.py / backflip.py   # 既有单文件变体
  supermimic.py                          # 唯一新增：family cfg 子类 + APG cfg 声明
```

term 数学、motion loader、sampler、FK chain、外生注记全部复用家族共享件，
不在变体里复制任何 python。

#### 一致性校验（派生访问器构造期）

- **dry-run**：哑 modeling state 上执行派生 program——非 read 集字段访问即时
  显式报错（含 term 名与字段名），group 完整性、obs 宽度、reward 权重聚合
  （从 cfg reward group 提取）一并校验；
- action 映射与 action term process **同一函数对象**（identity 校验，非
  重实现）；
- 外生规则与 command `advance` 的数值等价由 fixture 测试覆盖（WBT phase 递增
  H 步对齐真实 env）。

校验失败在 env 构造期抛出，不允许"两侧数学漂移"的任务进入训练。

#### 与训练 / 评估设施的衔接

- **多任务** = 多个可派生 modeling view 的任务共享同一 APG trainer（g1 WBT
  与 go2 为前两个实例），trainer 无任务分支；
- 课程状态（sampler）与外生 constant 通道随 `TRAINING_STATE` 序列化（D8），
  trainer 不感知内部结构；
- eval profile（D9）、`mxlab play`、ONNX 导出对任务侧零额外要求（actor 无
  状态，消费派生 obs）。

### 转换捕获与传输定案

#### 需求分析：WM transition 的每个字段从哪来

| 字段 | 来源 | 是否需要 env 内捕获 |
|---|---|---|
| modeling state `s_t`（含外生通道） | post-step 声明 read 数据（`env.sim_data`）+ 常驻 command/action term 的 host state；partial reset 只重写 done 行，**活行的 post-physics 值保留** | 否 |
| action `a_t` | collector 自持（policy 原始 action；动力学控制量由 learner 经 `task.action_of` 推导，不存储冗余） | 否 |
| reward / terminated / truncated | `ArrayEnvState` 既有字段 | 否 |
| 终态后继 `s_T`（done lanes） | 捕获点位于 evaluate 之后、reset 写入之前（`_capture_final_obs` 现位），reset 后即被销毁 | **是——唯一必须项** |

#### 方案对比（含 holosoma 参考）

- **A. EnvExtension + TransitionRecorder**（前一尝试）：step 内 hooks 每步发布
  完整 `TransitionSnapshot`。**否决**：重复 collector 已知信息（action）与
  post-step 仍可读的活行状态；为单一消费者引入新的生命周期扩展机制；且该尝试
  sync 走 recorder、async 走私有 block 传输，实际是两套装配路径。
- **B. holosoma 式 Phase hook registry**（`~/code/holosoma`：FRAME_BEGIN /
  PRE_STEP / POST_STEP / FRAME_END / EPISODE_START / EPISODE_END / CLOSE 类型化
  观察者总线，enable/disable/every 句柄，插件按 phase 订阅）：观察者总线更通用，
  但当前唯一消费者是终态捕获，按"不为未提出的消费者引入通用机制"原则**暂不
  引入**；其 final observation 以稀疏行经 step 返回值外传的模式与下案同型，
  设计精神采纳。
- **C（定案）. `FinalState`**：把 MotrixLab 既有 `FinalObs` 契约（truncated-only、
  obs-only、backend 可选实现）泛化为 FinalState——捕获点不变（evaluate 后、
  reset 前，单次 gather），覆盖 terminated + truncated 全体 done 行，内容为
  声明 read 字段行 + 常驻 command/action term state 行（外生通道终值：advance
  已执行、reset 未执行的时点上恰好是终值）；truncated 行的 obs 保留给既有
  model-free 消费者。契约变更同步更新
  [ArrayEnv 终态观测结果契约](./array-env-final-observation.md)。

#### 定案链路

```text
env.step ──> state (obs/reward/dones/…) ──┐
        └──> FinalState (done 行)  ────────┤
env.sim_data (活行, post-step) ───────────┼─> provider 转换装配器 ──> sync: SequenceReplay
collector 自持 action ────────────────────┘   (view 只走声明 read 集)  async: HostTransitionRing(WM schema)
                                                                        └─> learner 序列回放（边界 mask）
```

装配器消费派生 `ModelingView`（声明 read 集 + term host state 的通用视图，
禁止绕过声明直读 backend 数据——前一尝试的教训）；sync 与 async 共用同一个
装配器，`TransitionSchema`/`SlotPayloadStore` 按 D4 声明。

#### 与原有 per-term 任务的共存

- FinalState 是框架级机制，字段来自任务**本就为 term 而声明**的 read 集与常驻
  term state：per-term 任务零改动、零感知；未请求终态捕获时零成本（现 FinalObs
  语义原样保留）。
- `step()` 返回契约不变：model-free 消费 obs/reward/dones 完全不受影响；APG
  额外消费 `sim_data` / term state / FinalState，两者共存于同一 env 实例。
- 若未来出现逐帧多消费者（相机 / 传感类插件），再评估引入 holosoma 式 phase
  hooks（记为非目标，此处存档判断依据）。

### 数据流

Sync（Phase 3）：

```text
Trainer.train()
  env = registry.resolve(env_name).make(...)
  task = env.modeling_view()          # D3 派生能力探测（构造期 dry-run）
  loop:
    actor(modeling_state, no_grad) → env.step
    装配器(reader: sim_data 活行 + FinalState done 行 + 自持 action + state)
      → SequenceReplay.extend         # D6, episode 边界正确
    learner.update_world_model()       # 多步窗口 MSE
    learner.policy_update()            # 想象 BPTT (horizon H) + lambda-return critic
  checkpoint: POLICY=actor / TRAINING_STATE=全量（含 sampler 课程状态）
```

Async（Phase 4）：

```text
parent: topology = resolve_async_trainer_topology(...)
        per-collector: HostTransitionRing(WM schema) + WeightChannelShared(bundle spec)
collector proc: env + modeling_view + inference-only actor + 转换装配器（与 sync 共用）
        (weights ← InferenceBundleSpec: actor params + obs normalizer, NORM_KEYS)
learner proc: drain rings → 序列回放 → WM/BPTT 更新 → publish_weights
        (UTD 治理与全局步基准沿用 fastsac 的 Control 机制)
```

## Roadmap

### 主线脉络

一句话：**课程（P1）、建模（P2）、序列数据（P3.1）三条工作线并行推进，汇合于
通用 trainer（P3.2），经第二任务证明任务无关（P3.3），再以 async 放大规模
（P4）、packed 收敛传输成本（P5）**。每个 commit 都以"任何实现契约的任务 /
trainer 都需要"准入，APG 特化只出现在 provider 与任务 cfg 层。

```text
课程线   P1.1 ─> P1.2 ─> P1.3 ────────────────┐   起点课程可插拔（D2）
                                               │
建模线   P2.1  P2.2  P2.3★  P2.4  P2.5 ───────┤   任务数学可想象（D3/D4/D5/D10）
                                               │   （五 commit 彼此独立）
数据线   P3.1 ─────────────────────────────────┤   真实经验可回放（D6/D7）
                                               │   （不依赖 P1/P2）
                                               ▼
                 P3.2 APG provider + sync trainer + g1-supermimic（三线汇合）
                               │
                               ▼
                 P3.3 go2（通用性证明：零 trainer 改动）
                               │
                               ▼
                 P4.1 async ──> P4.2（条件）共用提炼

P5 packed 多 clip：独立于主线，单独设计单独评审
```

- **长杆是 P2.3（双路径编译）**：机制最新（shadow ctx 绑定、kernel_data shadow
  拷贝/共享、dry-run 报错质量）、风险最集中（见风险区）——最早启动、先 eager
  后 compile，shadow 规则在 P2 评审冻结；
- **并行开局**：P1.2 / P2.1 / P3.1 互不依赖、各自带独立验收（行为等价回归 /
  不声明则行为不变 / 纯新增模块），适合作为第一批 PR；
- **P2 内部**：五个 commit 无相互代码依赖，可交错合入（phase 级 fixture 验收需
  2.3 就位；WBT 家族想象另需 2.4 的 FK）；
- 每个 commit 独立可合入、独立可回归；测试随各 commit 的最终契约编写（不先于
  提交方案，遵循全局规则）。

### 验收主线（证据链）

每个验收点各证一层，层层递进：

| 验收点 | 证明什么 |
|---|---|
| P1.2 现有 WBT 任务回归一致（含 RNG 消耗顺序不变） | 课程重构不改现有任务行为 |
| P2 fixture：同一 term 双路径数值一致 | 任务数学单源成立（想象 = 真实） |
| P3.2 g1 端到端收敛 + play/ONNX parity + 升格 env model-free 回归一致 | trainer 可用，且不破坏现有家族 |
| P3.3 go2 端到端跑通 | trainer 任务无关（第二任务零 trainer 改动） |
| P4.1 async 与 sync 等步数损失曲线对齐 | 规模化不损样本效率 |

### Phase 1 — WBT 采样器统一（环境侧课程基建）

以现有 WBT 任务（g1 单/多 clip）为验证载体，不引入任何 trainer 概念。

commit 拆分：

1. **`ctx.reward` mirror**（`motrix_env_core`）：`ctx.reward` 与 `state.reward`
   同存储的只读镜像，reset/advance hook 可读最近一步 reward。通用能力，
   是 return 累积的唯一 env_core 前置。
2. **sampler.py 重构**（`motrix_envs`，行为等价）：`MotionStartSampler` 协议 +
   `Uniform/Adaptive` torch 实现（多 clip per-clip bins）；`WbtMotionCommand`
   十余个散落属性收敛为单个 `sampler` 嵌套成员；`adaptive_sampling_enabled`
   → `sampling_mode`；numpy 旧实现删除。验收：uniform/adaptive 模式下现有
   WBT 任务回归一致（含 RNG 消耗顺序不变）。
3. **`ReturnTimestepsSampler`**（`motrix_envs`，依赖 commit 2）：return 统计、
   per-lane 折扣回报累积与采样数学落地；WBT 任务 `sampling_mode=return` 可跑，
   课程统计经 metric 发布。验收：单 clip 对齐前一尝试的采样分布行为，
   多 clip 无跨 clip 归因。

### Phase 2 — env 侧建模契约（modeling / 捕获 / truncation）

APG 的环境前置，全部按"任何需要想象侧任务数学的任务"准入，与 Phase 1 并行。

commit 拆分：

1. **Manager `truncation` group**（`motrix_env_core`）：一等 truncation term
   （D5），现有 env 不声明则行为不变。
2. **FinalState 终态捕获**（`motrix_env_core`）：FinalObs 契约泛化——单捕获点
   （evaluate 后、reset 前）覆盖全部 done 行，内容为声明 read 字段行 + 常驻
   command/action term state 行；同步更新
   [ArrayEnv 终态观测结果契约](./array-env-final-observation.md)。
3. **resolver / executor 双路径编译**（`motrix_env_core`，manager 包内）：
   modeling program 派生（同一 call records 绑定 shadow ctx，command update
   hook 一并参与）、command kernel_data 的 shadow 拷贝 / 常量共享规则、
   `ModelingView` 派生类型与 `modeling_view()` 访问器、group 级 all-or-nothing
   与 dry-run enforcement、外生通道 cfg 注记字段。
4. **FK 上游化 + query 语义核验**（`motrix_env_core` + `motrix_env_motrixsim`）：
   device-mobile KinematicTree（D10）、read-program 声明驱动的 FK freshness、
   核验 read query 类型 → trainer 动力学结构自动组装的覆盖面（新 query 类型
   声明可否想象的框架义务）。
5. **杂项契约**（`motrix_env_core`）：`target_offset` reset 随机化、action range
   显式放宽（无 ctrl_range → 配置化无界/对称维度 + warning）、resolver term
   参数可携带 kernel_data 载体（FK chain，消除任务侧闭包 workaround）、
   `math/rotation` 增补。验收：现有 env 全量回归。

验收（phase 级）：最小 fixture 任务上**同一 term 双路径数值一致**——真实 kernel
与 modeling program 在相同输入下 reward / obs / termination 输出一致（含权重
聚合、group 校验与 dry-run 全部生效）。

### Phase 3 — APG trainer + 任务侧定义（通用多任务）

三线在此汇合：通用 trainer 落地，并以 g1 → go2 两个任务实例验证任务无关
（D1/D3/D6/D7/D8/D9）。

commit 拆分：

1. **序列数据平面**（`motrix_rl`）：WM TransitionSchema 模板、episode 边界正确
   序列回放共享模块（D6）、`NORM_KEYS` 归一器契约（D7）、`_METRICS_FAMILIES`
   注册 `apg`。无 provider，独立可合入。
2. **APG provider + sync trainer + g1 任务**（`motrix_rl_builtin` + `motrix_envs`）：
   `MotrixFramework` 聚合双 provider；`apg/` 的 config/networks/learner/policy
   （算法核心自前一尝试回收，传输栈不回收；dynamics 结构按 `FieldDesc`
   自动组装）；`env.modeling_view()` 绑定（D3）；WBT 家族升格 = family cfg
   加外生注记（term 函数体不动）+ g1-supermimic **单文件**家族变体；
   checkpoint/play/ONNX 导出（D8）。验收：g1 端到端有限损失收敛（uniform
   起步，切 `sampling_mode=return` 课程），`mxlab play` 与导出 parity 通过；
   升格后的基线 WBT env 在 model-free 训练下回归一致。
3. **go2 四足任务**（`motrix_envs`，依赖 commit 2）：第二任务实例验证 trainer
   通用性——共享 `quadruped` 家族数学，Direct `walk_np` 与 Manager 变体并列
   共存，同一 trainer 仅更换任务配置（modeling view 各自派生）。验收：go2
   端到端跑通。

### Phase 4 — async 采集

复用 #160/#162 的 transport 拓扑把 sync 管线进程化；编排、UTD 治理与权重发布
复用 fastsac 模式，不新建通用 async 骨架。

commit 拆分：

1. **APG async**（`motrix_rl_builtin`）：collector/learner 进程化走
   `motrix_rl.transport` + `AsyncTrainerTopology`（WM schema +
   InferenceBundleSpec 声明）；UTD 治理、权重发布节奏、全局步基准复用 fastsac
   模式。验收：async 与 sync 等步数损失曲线对齐；背压与 seqlock 行为正确。
2. **（条件）共用提炼**：仅当 fastsac 与 apg 出现明确重复的编排代码时，把重复
   部分提炼进 `motrix_rl`；不预先建骨架。

### Phase 5（后续优化，非核心）— packed 多 clip

多 clip 参考数据以 device-resident packed buffer 一次性驻留（或分块 pin-memory
批量上传），消除逐步 CPU→GPU 传输开销；相关 converter 随之泛化。前一尝试的
dangling commit（88d0eb0）中有 packed 栈先例可参考思路。独立于 trainer 演进，
不阻塞主线，单独设计单独评审。

## 测试策略

前一尝试自带的 `test_mbrl_*` / `test_supermimic_*`（~2500 行）已删；测试不直接
搬回，而是随各 commit 按最终契约编写：契约级（sampler 协议与 RNG 顺序、
truncation、modeling 协议、schema、序列回放、NORM_KEYS）随 Phase 1/2/3 的
对应 commit；算法级（learner 收敛 smoke、想象侧与真实侧数值一致性）随
Phase 3。

## 风险与开放问题

- **RNG 顺序兼容**：sampler 重构必须保持各模式固定 draw 数，否则现有任务的
  采样流变化会使历史 run 不可比——commit 2 以行为等价为验收线；
- **派生式双路径的边界情形**：modeling program 是独立编译域（shadow ctx 宿主
  对象、独立 guard），先 eager 后 compile，避免与真实 kernel 的编译缓存互相
  干扰；command kernel_data 的 shadow 拷贝 / 常量共享区分规则是机制复杂点，
  P2 评审冻结；隐式派生（无显式标记）的错误定位完全依赖 dry-run 报错质量，
  报错必须落到具体 term 与字段名；reward term 内使用 `ctx.rand` 的罕见情形
  需在协议冻结时定义想象侧语义（确定性或注入）；派生不检测数值等价——
  等价性由 dry-run + fixture 数值一致性测试兜底；显式意图标记
  （`@modeling` / `@differential`）在真实混合任务出现前不引入；
- **per-clip bins 的空 bin**：clip 长度差异大时短 clip 的 bin 粒度由
  `max_bins` 归一，需在实现时定分箱规则（等时长优先，不足合并尾 bin）；
- **action range 放宽**改变依赖旧 raise 的任务行为，Phase 2 需全 env 回归；
- **FinalState 捕获成本与语义边界**：done 行 gather 与现 FinalObs 同构、字段更宽，
  稳态量级可忽略，但大规模同步 reset 步（初始化/课程切换）值得基准；捕获覆盖
  terminated 行是语义增量，model-free 消费者不受影响，但终态观测契约文档必须
  同一 commit 更新，防止两个契约版本并存；
- **FK freshness** 的自动 post-step 刷新对声明链查询的 env 引入全局每步成本，
  Phase 2 需基准量化（必要时改为 read-plan 显式声明）；
- **async worker 的 motion 引用驻留**（体量级参考数据每 worker 一份）是规模上限，
  Phase 4 用 CollectorCfg.device 先行缓解，根治归 Phase 5 的 packed 驻留；
- APG 命名（provider id `motrix.apg`、metrics family）在实施前冻结。

## 关联

- [RL 多算法架构设计](./rl-multi-algorithm-architecture.md) — provider 契约与发现机制
- [FastSAC 异构训练器设计](./fastsac-async-heterogeneous-trainer.md) — transport/weight channel 语义
- [ArrayEnv 终态观测结果契约](./array-env-final-observation.md) — FinalState 泛化的基线契约
- [多 Motion Clip 训练基础设施设计](./motion-multi-clip-infra.md) — 全局帧轴与 `frame_clip_end`
- [Manager Runtime 设计](./manager/runtime.md)、[Torch 执行器设计](./manager/torch-executor.md) — term 协议与执行边界
- [Framework / Task 配置分离设计](./framework-task-split.md) — task recipe 组合方式

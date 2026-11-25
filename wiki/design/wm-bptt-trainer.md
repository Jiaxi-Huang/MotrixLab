# APG（WM/BPTT）Trainer 设计与路线图

## 摘要

在已落地的课程、Manager 建模与序列数据基建之上，近期目标是对齐本地
SuperMimic 的 **G1 GenCoord motion tracking**：观测、环境、奖励、actor / critic /
world model 网络结构，以及训练、推理和导出的输入输出。先建立可验证的同步 APG
（world model + 通过冻结 WM 的策略 BPTT）闭环，再优先参考 XMimic 优化 trainer
编排、单 GPU 执行与多 GPU 并行。

任务语义以 SuperMimic 为准，训练工程优先参考 XMimic；保持 MotrixLab 的 Torch /
MotrixSim、provider、modeling 和 transport 边界。任务数学落在 `motrix_envs`，
真实与想象路径单源；APG 网络与 learner 落在 `motrix_rl_builtin.apg`。

## 观测布局与 main 契约

APG 从 Manager 的命名 observation term layout 与 `ObservationSpec` 解析输入，
不通过机器人关节数推导观测尺寸。policy 必须且仅包含 `proprio`、`goal` 两项，
顺序可交换；value 使用完整的特权观测组。所有组保持单帧，非单帧组在初始化时拒绝。
布局由 APG 插件内的不可变对象持有，collector、learner、normalization 与 play
使用相同切片；异步子进程核对环境布局后使用主进程传入的布局。

历史只包含 proprio，goal 使用当前帧。真实采集与 imagination 分别使用 main 的
`repeat_observation` / `append_observation`，后者保留跨步梯度。
环境 observation history 与策略输入 history 不叠加。
replay 的 `obs` 保存命名 proprio；导出仍接受 oldest-first、frame-major 的
proprio history 加当前 goal，不暴露环境 term 的物理排列。

checkpoint 继续保存 proprio/goal/action 尺寸与 Policy 的 low/high buffers。
resume/play 在加载边界核对尺寸；旧 checkpoint 与 ONNX 输入格式保持兼容，
无需保存环境切片。动作、sim 与 command 的 FinalState、独立 modeling RNG、
SequenceReplay 和函数式 imagination workspace 仍是 APG 必要扩展；
真实与想象路径共享 term 调用契约，不共享原地执行生命周期。

默认 Muon 与 compiled imagination 在 Torch 2.11 下验证；workspace 锁定 Torch 2.7
尚不支持默认 APG 路径。依赖升级及各 provider 回归是独立的合入前待办，
本次结构适配不改变依赖锁、默认优化器或 compile 配置。

## 参考基准与范围

本轮固定本地 checkout，后续对齐规格与验收记录须携带参考版本：

| 参考 | 版本 | 负责的基准 | 主要源码入口（相对各仓库根目录） |
|---|---|---|---|
| `~/code/supermimic` | `abc124c` | GenCoord motion 的环境、观测、奖励、网络和输入输出 | `src/supermimic/representations.py`、`tasks/env_cfgs.py`、`envs/`、`utils/tracking_reward.py`、`models.py`、`runners/exporter.py` |
| `~/code/xmimic` | `54860e1` | trainer 编排与加速策略 | `xmimic/algorithm.py`、`fast.py`、`fast_nets.py`、`single_gpu.py`、`multi_gpu.py` |

XMimic 的 `xmimic/mdp.py` 使用 `(p, R, v, w, q, qd)`：根位姿、根线/角速度、
关节位置/速度，经 FK 计算身体状态，属于广义坐标路线。但它基于 BlinkMimic 的
观测、奖励与网络配方，不能作为当前 SuperMimic 的语义替代品。SuperMimic 继承的
mjlab 机器人与环境配置也属于核对范围，须在对齐规格中记录实际依赖版本。

只对齐 motion 分支的 goal encoder、latent、decoder 与历史观测，排除 velocity
encoder、多任务 slot 混合及跨任务辅助损失；不把多任务训练效果作为本轮目标。
MaxCoord、go2 第二任务验证均留待后续需求，不阻塞本轮闭环。

## 基建现状与架构约束

- 课程采样器已统一为 torch 原生 uniform / adaptive / return，支持多 clip。
- Manager truncation、done 行 FinalState 捕获、派生式双路径建模和 query
  modelability 已落地；可微 FK 已上游化，SuperMimic 任务级 parity 尚待验收。
- WM schema、episode 边界正确的 SequenceReplay、共享归一器与 `apg` metrics
  family 已落地；同步 APG provider 已接入，参考实现的完整语义对齐与训练质量仍待验收。
- 保留 `TrainerBase.train()/play()`、标准 provider discovery 与既有 transport；
  不引入私有平行传输栈、YAML Python 路径绑定或任务侧生命周期 override。
- 框架改动只由实际对齐缺口驱动；任务名、奖励公式与课程超参不得跨入框架层。
  旧尝试中的 learner / networks 只能在符合本轮对齐规格后复用。

## 目标

- 通过 `motrix.apg` 接入 GenCoord motion 训练，trainer 不按具体任务名分支。
- 对齐 SuperMimic 的状态表示、历史/目标窗口、噪声、动作映射、环境生命周期、
  奖励与网络；所有差异须显式记录，不能以“能训练”代替对齐。
- 真实与想象路径消费同一份任务数学；允许新增必要 term、状态表示与配置，
  不要求 SuperMimic 只是既有 WBT 的单文件配置变体。
- 保持现有 WBT 注册、默认配置和 model-free 行为不变，复用 motion、sampler
  与 FK 基建，不强行共用语义不同的 term。
- 同步参考路径验收后再逐项加速，报告数值偏差、样本效率与吞吐收益。

## 非目标

- 不迁移到 JAX / MJX，不复制 XMimic 的完整算法配方或全部多 GPU 拓扑。
- 不引入循环 actor 隐状态；历史观测作为显式输入，采集、回放、play 与导出均
  须维护一致的帧顺序和 reset 语义，不能因此省略历史管理。
- 不做通用想象引擎、多算法 WM 抽象或预建 async trainer 骨架。
- 不承诺跨仿真器轨迹逐位一致、参考 checkpoint 文件格式兼容或 XMimic 的速度数字。
- packed motion、独立评估 worker 等优化由实测瓶颈驱动，不作为初始闭环前置。

## 架构

### 分层与职责

```text
motrix_envs (任务层)
  motion/sampler.py：MotionStartSampler 族（uniform/adaptive/return，多 clip）
  g1 SuperMimic：GenCoord motion 的 term、状态表示与 cfg；复用 WBT/motion 基建
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
  apg/：config、networks（motion actor / critic / GenCoord dynamics）、learner、policy、
  sync 参考 trainer → 单 GPU 加速 → async / 多 GPU
```

### 关键设计决策

**D1 — Trainer 契约不扩展。** `TrainerBase.train()/play()` 已被 fastsac sync/async
两路验证足够。APG trainer 通过 `TrainerContext` 消费 run 目录、checkpoint、metrics，
与 fastsac 同构；多任务能力来自"trainer 只依赖派生 modeling view"，不来自契约膨胀。

**D2 — 复用 torch 原生 motion sampler 族。** `MotionStartSampler` 提供
uniform / adaptive / return，多 clip 统计归 sampler 所有，`WbtMotionCommand`
持有并委托；return 模式通过 `ctx.reward` 累积 episode 回报。既有采样行为与 RNG
契约保持，SuperMimic 的采样配方在 P3.2 对照后选择，不能以已有 return sampler
替代参考课程的语义核验。

**D3 — 任务↔trainer 绑定走派生能力探测，不走字符串路径。** APG 可训练性是 env
cfg 的派生能力加 provider 输入契约：`ManagerEnv` 基类提供 `modeling_view()`
访问器（派生时 dry-run，无需子类 override）。sync trainer 直接问 env 实例；async learner 沿用
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
连续性与 done 边界切窗、边界与 ring-wrap 感知的采样器（从前一尝试回收泛化）。
实际连续性、历史重建与 schema 契约见[序列数据平面设计](./rl-sequence-data-plane.md)。

**D7 — 归一化数学与传输契约分别对齐。** actor proprio、goal 和 WM 的归一化
按 SuperMimic 核对，不能因已有共享 normalizer 就假定数值等价。对外状态满足
`NORM_KEYS` 契约；inference bundle 覆盖推理实际需要的全部归一化状态，async
同步走既有 `WeightChannelShared`。

**D8 — checkpoint 语义不扩种。** actor 导出为 `POLICY`，沿用现有导出机制，
但须补齐 motion goal、历史输入和归一化的接口适配与 parity 验收，不能假设直接
导出即可。WM/critic/optimizer/normalizer/序列回放采样器/
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

APG 的任务数学通过 env cfg 派生，不另外维护一套想象侧公式。SuperMimic 对齐
允许新增任务 term 与表示转换；建模能力可派生只是前置，还需核验 provider 所需的
状态、历史、目标与动作契约。具体派生规则以[Manager 双路径建模设计](./manager/modeling.md)为准。

#### 设计原则

1. **派生而非重复定义。** 字段集、双路径 program 和 action 映射从 read 集、
   term 集与 cfg 声明派生；序列起点有效性由 replay/learner 的连续性契约处理。
2. **数学单源 = 同一 term 双路径编译，不人工拆分函数。** term 的 dispatch 函数
   是共享数学源；共享 term 通过不同 **ctx 绑定** 执行（常驻
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
   属 apg provider：消费字段语义并按 SuperMimic GenCoord 的表示、预测与积分
   规格实现，不能仅从 query 类型推断完整网络。env_core 不出现 DynamicsLayout。
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
  常量成员（clip 数据、FK chain）共享只读**——区分规则已在
  [Manager 双路径建模设计](./manager/modeling.md)冻结；
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

具体 `ModelingView` 字段与调用方式以[Manager 建模设计](./manager/modeling.md)
和实现为准，此处不另维护一份接口定义。

trainer（apg provider）消费 view：用 `FieldDesc` 绑定 GenCoord 网络所需的语义字段；
learner 用 program 在想象态上算 reward / obs / survive；`valid_start` 默认由
learner 按 episode bookkeeping 推导（过滤跨 episode 起点），任务级覆盖留待
真实需求出现。

#### 与现有任务的关系

既有 WBT 任务继续使用原有注册与配置。`g1-wbt-supermimic` 作为 G1 WBT 族任务配置（`wbt/g1/supermimic.py`，与 backflip、dance 并列），经
`task=g1-wbt-supermimic/motrix.apg` 接入；复用语义一致的 motion loader、sampler、FK
和 term，差异化观测、奖励、reset 与状态转换放在任务层。无需为了文件数限制把
SuperMimic 的数学塞入 common cfg，也不修改既有 WBT 默认配方来迁就新任务。

仅有 cfg 差异时使用配置继承；有数学差异时新增必要 term，真实与想象侧共用。
第二任务通用性验证后续单独规划，不为 go2 预建 Manager 变体。

#### 一致性校验（派生访问器构造期）

- **dry-run**：哑 modeling state 上执行派生 program——非 read 集字段访问即时
  显式报错（含 term 名与字段名），group 完整性、obs 宽度、reward 权重聚合
  （从 cfg reward group 提取）一并校验；
- action 映射与 action term process **同一函数对象**（identity 校验，非
  重实现）；
- 外生规则与 command `advance` 的数值等价由 fixture 测试覆盖（WBT phase 递增
  H 步对齐真实 env）。

派生访问器构造时拒绝不满足结构契约的任务；数值与梯度等价由 fixture 验收，
dry-run 本身不能证明数学对齐。

#### 与训练 / 评估设施的衔接

- trainer 消费 modeling 与显式 actor 输入，不按 `g1-wbt-supermimic` 名称分支。
- 课程状态与外生 constant 通道随 `TRAINING_STATE` 序列化（D8），trainer
  不感知 sampler 内部结构。
- 采集、想象、play 与 ONNX 使用一致的 motion goal 和历史排列；reset 的历史
  填充、previous action、归一化冻结及部署输入中排除特权信息均须专项验收。

### 转换捕获与传输定案

#### 需求分析：WM transition 的每个字段从哪来

| 字段 | 来源 | 是否需要 env 内捕获 |
|---|---|---|
| modeling state `s_t`（含外生通道） | post-step 声明 read 数据（`env.sim_data`）+ 常驻 command/action term 的 host state；partial reset 只重写 done 行，**活行的 post-physics 值保留** | 否 |
| action `a_t` | collector 自持（policy 原始 action；动力学控制量由 learner 经共享 action process 推导，不存储冗余） | 否 |
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

Sync（P3.4）：

```text
Trainer.train()
  env = registry.resolve(env_name).make(...)
  task = env.modeling_view()          # D3 派生能力探测（构造期 dry-run）
  loop:
    actor(motion_goal, proprio_history, no_grad) → env.step
    装配器(reader: sim_data 活行 + FinalState done 行 + 自持 action + state)
      → SequenceReplay.extend         # D6, episode 边界正确
    learner.update_world_model()       # 多步窗口 MSE
    learner.policy_update()            # 想象 BPTT (horizon H) + lambda-return critic
  checkpoint: POLICY=actor / TRAINING_STATE=全量（含 sampler 课程状态）
```

Async（Phase 5）：

```text
parent: topology = resolve_async_trainer_topology(...)
        per-collector: HostTransitionRing(WM schema) + WeightChannelShared(bundle spec)
collector proc: env + modeling_view + inference-only actor + 转换装配器（与 sync 共用）
        (weights ← InferenceBundleSpec: actor params + proprio/goal normalizers, NORM_KEYS)
learner proc: drain rings → 序列回放 → WM/BPTT 更新 → publish_weights
        (UTD 治理与全局步基准沿用 fastsac 的 Control 机制)
```

## Roadmap

### 主线与编号

保留 P1 / P2 / P3.1 编号以承接已落地基建与设计文档；近期工作从 P3.2 开始，
按“规格 → 环境 → 网络与同步闭环 → 单 GPU 加速 → 多 GPU”推进。关联设计中
原 P3.2 的 provider/装配器交付对应本路线图 P3.4，原 P4 async 对应 P5。每个阶段实施前
先完成对应设计，并在 `wiki/plan/` 建立带 TODO 的实施计划；本轮只更新路线图。

```text
P1 课程 + P2 建模/捕获/FK + P3.1 序列数据（已落地基建）
                  │  剩余能力缺口随具体对齐需求补齐
                  ▼
P3.2 SuperMimic GenCoord motion 对齐规格与差异清单
                  ▼
P3.3 环境 / 观测 / 奖励 / 动作 / 状态表示对齐
                  ▼
P3.4 网络 + APG sync + checkpoint / play / ONNX
                  ▼
P4 单 GPU 加速（参考 XMimic，保留同步参考路径）
                  ▼
P5 async / 多 GPU（按瓶颈逐步拆分）

后续独立需求：MaxCoord、velocity / 多任务、go2 通用性验证
packed motion：按 P4 / P5 实测传输与驻留瓶颈安排
```

### 基建完成状态（P1 / P2 / P3.1）

完成表示能力已落地，不等于 SuperMimic 已对齐或 GPU 训练已验收。

- [x] **P1 课程基建**：`ctx.reward` mirror、uniform / adaptive / return sampler
  与多 clip 支持（`31cb4c8` / `5ff2d87` / `53882c5`）。
- [x] **P2.1 Manager truncation**（`1dc5ce1`）。
- [x] **P2.2 FinalState**：reset 前 done 行捕获（`22131a0`），契约见
  [终态结果设计](./array-env-final-observation.md)。
- [x] **P2.3 双路径建模**（`159f617`），shadow 规则与派生能力见
  [Manager 建模设计](./manager/modeling.md)。
- [x] **P2.4 基础能力**：query modelability（`c4bcfd6`），可微 FK、声明通道与
  freshness 已由 #163 提供；不是待重建的任务侧桥接层。
- [ ] **P2.4 任务验收**：GenCoord 模型态 FK 与 sim read 视图 parity，随 P3.3 完成。
- [x] **P2.5 kernel_data 参数**：resolver `_resolve_term_arg` 已接收载体常量。
- [ ] **P2.5 剩余缺口**：target offset 随机化未见对应实现；joint position action
  对缺失 ctrl_range 仍报错。P3.2 核对参考动作与随机化语义后决定最小补充，
  不预先实现通用无界动作变体。rotation helpers 按 GenCoord 所需操作逐项核验，
  不能整项标记完成。
- [x] **P3.1 序列数据平面**（`7d06b6f`）：WM schema、SequenceReplay、共享
  normalizer 与 `apg` metrics family，详见[序列数据平面设计](./rl-sequence-data-plane.md)。

### P3.2 — 冻结 GenCoord motion 对齐规格

- [ ] 建立 SuperMimic → MotrixLab 对照表：源码/版本、当前能力、差异、实现归属、
  验收方式；追踪继承的 mjlab 配置，不仅核对 SuperMimic 的覆写项。
- [ ] 冻结观测与 I/O：字段顺序、shape、坐标系/四元数约定、单位、proprio history、
  motion goal 未来窗口、actor 与 critic/WM 特权输入区别、噪声与归一化。
- [ ] 冻结环境语义：机器人/关节顺序、控制频率与 PD、动作 clamp/scale/offset、
  previous action、reset/随机化/采样、clip 边界、termination/truncation。
- [ ] 分别核对真实侧与想象侧 reward、survival/termination、参考帧推进与噪声；
  有意差异显式记录，共享数学不应掩盖软终止等路径差异。
- [ ] 冻结 actor motion encoder / latent / decoder、critic、WM 的层结构、激活、
  normalization、GenCoord prediction/integration，以及训练损失和输入输出。
  本轮只启用 motion 分支，不保留 velocity 分支与跨任务辅助损失。

**出口**：每项语义都有来源与验收方法；识别需要补齐的 core/provider 契约。
训练循环和优化器等可优先借鉴 XMimic，但凡改变损失、采样或学习语义的选择须
单独记录，不能混入“等价加速”。固定评估数据、seed、预算与质量指标，不直接
使用 XMimic 的回报阈值。

### P3.3 — 环境与任务数学对齐

- [ ] 在 G1 WBT 族内实现 GenCoord motion 任务，复用已落地 sampler、motion、FinalState
  与 modeling；补齐必要任务 term、表示转换和 P2 剩余实际缺口。
- [ ] 对齐 proprio/goal/history、动作、奖励、reset 与边界语义，保证真实与想象
  路径共享数学；对比参考实现逐项输出，而非只比总 reward。
- [ ] 完成模型态 FK ↔ sim read parity、模型态转换/积分和关键梯度验收；验证
  shadow 执行不污染真实状态，既有 WBT model-free 正向回归通过。

**出口**：固定输入下任务数学达到规格定义的容差；跨仿真器只比较可比语义与
统计行为，不要求长轨迹逐位一致。

### P3.4 — 网络与同步 APG 闭环

- [ ] 实现标准 `motrix.apg` provider、类型化配置、GenCoord WM、motion actor
  和 critic；网络与归一化依据 P3.2，不能直接回收旧 MLP 假设。
- [ ] 以 FinalState + 活行状态装配真实转换，复用 SequenceReplay；正确重建
  previous action/history、屏蔽跨 episode 序列，完成 WM 学习与冻结 WM 的 BPTT。
- [ ] 对齐 actor/critic/WM 前向与关键梯度，再验证有限损失、学习进展和固定
  motion 数据上的评估结果；单独报告运动片段覆盖与失败情况。
- [ ] 完成 checkpoint/resume、sampler/normalizer 状态恢复、`mxlab play` 与 ONNX
  parity，覆盖 goal/history 顺序、reset 填充、推理归一化冻结和特权输入边界。

**出口**：有可复现的 GenCoord motion 同步参考路径与评估结果；“成功启动”或
“损失有限”不足以单独证明闭环对齐。

### P4 — 单 GPU 加速

- [ ] 先测分阶段耗时、主机同步、编译成本、稳态吞吐与显存，确定瓶颈。
- [ ] 参考 XMimic 的编译训练循环、循环外预计算、数据驻留、批量更新与混合精度，
  选择适合 Torch 的实现；保留同步参考路径，不照搬 JAX/XLA 或 Hopper 专属参数。
- [ ] 每项优化比较输出、loss 与梯度；混合精度使用明确容差，并核验等样本训练
  质量。涉及优化器或更新比例变化时作为训练策略实验单列。
- [ ] 若 motion 传输成为瓶颈，再设计 packed 驻留；没有证据不扩建传输抽象。

**出口**：同时报告编译时间、稳态吞吐、等样本质量与达到质量目标的墙钟时间，
记录硬件、精度、seed 与运行预算；不能只用 it/s 声称训练更快。

### P5 — async 与多 GPU

- [ ] 先采集/学习拆分：沿用 `motrix_rl.transport`、WM schema、inference bundle
  与权重通道，复用 sync 转换装配器，明确 UTD、全局步数和权重版本滞后。
- [ ] 根据瓶颈再评估 WM learner / policy learner 拆分、critic 放置及评估 worker；
  不承诺复现 XMimic 全部布局，不把 probe/failure-search 课程作为默认加速项引入。
- [ ] 验证背压、权重发布、episode 边界与恢复；多 seed 比较等样本评估质量、
  time-to-target、显存和 GPU 时间成本，显式记录异步滞后。
- [ ] 仅出现真实重复编排时提炼 fastsac/APG 共用设施；motion 重复驻留按实测优化。

**出口**：以质量与资源成本证据判断并行收益；异步不要求损失曲线逐步相同，
但不能隐去样本效率下降或改变任务配方来换取吞吐。

## 测试策略

长期测试保护通用行为契约：序列边界、FinalState、shadow 隔离、历史重置、
动作映射、梯度传播、checkpoint 与导出 I/O；优先 CPU 小 fixture，不机械照抄
任务固定超参。参考仓库输出对照、版本扫描与性能测量作为专项验收，记录输入、
源码版本、容差、命令和结果，不把外部 checkout 或 GPU 加入默认单元测试依赖。

- **数学验收**：固定状态/动作/参考帧，逐字段比较 obs、goal、奖励、终止、FK、
  网络前向和关键梯度；分别验证真实/想象路径中的噪声与软终止差异。
- **边界验收**：episode 起点、history 不足、clip 尾部、terminated/truncated、
  auto-reset 真后继、回放断链与 ring wrap。
- **闭环验收**：motion 单分支训练、固定评估协议、恢复、play/ONNX parity。
- **优化验收**：参考与优化路径的输出/梯度容差，加上多 seed 质量与耗时统计；
  dry-run、CPU 检查与真实 GPU 运行结果分开报告。

## 风险与待定项

- **参考版本漂移**：规格固定 checkout，升级参考版本须重新列差异；XMimic 的
  广义坐标路线不代表其观测、奖励和网络与 SuperMimic 当前版本一致。
- **历史与目标泄漏**：actor、critic、WM 输入不同，部署不能误带特权状态；
  历史 reset 与目标窗口边界错误可能在训练能运行时仍破坏对齐。
- **物理差异**：MotrixSim 与参考 simulator 的动力学误差、PD/控制频率和机器人
  参数必须与数学差异分开诊断，不能用放宽数值容差掩盖输入布局错误。
- **建模可派生不等于可训练**：query modelability、dry-run 与单源只能保证部分
  契约；仍需验证梯度、噪声、随机化上下文和 GenCoord 表示的完整性。
- **动作与随机化缺口**：P2.5 在 P3.2 冻结具体需求后实现最小扩展，并验证现有
  action space、model-free 与部署消费者。
- **加速收益与漂移**：混合精度、异步滞后、不同更新比例可能改变学习；任务语义
  不能随优化漂移，算法策略变化须独立验收。
- **性能成本**：FinalState 同步 reset 峰值、FK freshness、motion 每 worker 驻留
  和历史回放内存需实测，不预先认定可忽略或必须新建基础设施。

## 关联

- [RL 多算法架构设计](./rl-multi-algorithm-architecture.md) — provider 契约与发现机制
- [FastSAC 异构训练器设计](./fastsac-async-heterogeneous-trainer.md) — transport/weight channel 语义
- [ArrayEnv 终态观测结果契约](./array-env-final-observation.md) — FinalState 泛化的基线契约
- [多 Motion Clip 训练基础设施设计](./motion-multi-clip-infra.md) — 全局帧轴与 `frame_clip_end`
- [Manager Runtime 设计](./manager/runtime.md)、[Torch 执行器设计](./manager/torch-executor.md) — term 协议与执行边界
- [Framework / Task 配置分离设计](./framework-task-split.md) — task recipe 组合方式

## 同步实现规格

参考为 SuperMimic `abc124c`，其 `pyproject.toml` 指向本地 `mjlab_research`（`dd51a9f`）。
实现采用 motion-only、sensor proprio 模式，xyzw 四元数与本仓库保持一致。

- proprio 为 gravity(3)、body angular velocity(3)、relative joint position(J)、
  joint velocity(J)、previous raw action(J)；历史按帧排列、最旧在前，reset 重复首帧。
- goal 为十帧 `[q, qd]` 窗口（间隔五帧），随后为十帧 root-relative rotation 的
  6D 表示（旋转矩阵前两列按行展开）；value goal 在旋转前额外带 relative position。
  窗口在各 clip 尾帧截断，不读入下个 clip。
- GenCoord rep 为 `[z, gravity_b, v_b, w_b, q, qd]`，aux 为 `[p, quat_xyzw]`。
  WM 输出 `[delta_v_b, delta_w_b, delta_qd]`；姿态按 `(decimation+1)/(2*decimation)`
  积分系数推进。输出增量按真实转换统计归一化；监督包括 root position、
  rotation matrix/sqrt(2)、根部世界线/角速度、q、qd 的加权 MSE。
- actor goal encoder 与 history decoder 使用 Linear/ELU，latent 经 tanh，输出小幅初始化；
  WM 使用 Linear/LayerNorm/SiLU。critic 使用特权 rep 与完整 goal，输入和目标分别归一化。
  normalizer 更新显式进行，forward 保留输入梯度。
- 真实奖励和想象奖励共享 Manager term；沿用参考六项 tracking 权重、action rate
  -0.1 和 joint limit -2。硬终止按 root 高度/重力 Z 差；clip 尾部为 truncation。
- 同步参考路径使用同一 Manager program，想象消费任务声明的 sigmoid soft survival，
  真实路径使用同阈值的硬终止；真实 proprio 使用参考噪声幅度，想象侧沿用 deterministic shadow RNG。
  此差异及 MotrixSim 动力学、机器人参数、reset sampler
  和未启用的 startup COM/friction/encoder-bias 随机化必须在质量报告中单独列出。
- policy checkpoint 只含推理网络、normalizer 与输入规格；training checkpoint 另含
  WM/critic/优化器、replay RNG 与课程状态。恢复重新采集 replay，episode 不跨恢复点。
- ONNX 为单一 flat 输入 `[history(frame-major), actor_goal]`，不包含 privileged position。
  调用方负责维护历史，play 使用相同历史维护器。

验收分为 CPU 契约测试、固定输入数学/梯度验收和真实仿真/训练验收；前两项不证明
训练收敛或跨仿真器兼容。无可用 GPU 时应明确记录未执行的质量/性能验收。


## 单 GPU 训练实现

### 想象观测噪声对照

性能对照将 modeling 的数值 term 求值与生命周期绑定分离：绑定、工作区更新仍走
原路径；可显式编译固定 program 的 term 求值和 observation。编译路径必须先通过
同输入的奖励、终止、观测及输入梯度对照，再以实测耗时决定是否接入 trainer。

真实 proprio 含任务定义的噪声，而确定性 shadow 默认取分布中点。APG 可显式请求
带独立 seed 的随机 modeling fork，使用同一 term 和 `RandomSource` 抽样接口，避免在 trainer
复制噪声幅度。默认 fork 仍为确定性；随机 fork 的 RNG 状态随 training checkpoint
保存，互不污染真实环境。北京 5090 实验保持当前网络、batch、horizon 和全量数据，
比较原配方及随机想象候选，按固定间隔完整评估 40 个 clip，报告首次实测 40/40 的
训练 wall time 与评估间隔，不宣称精确连续时间阈值。

训练实现保持 Manager 单源奖励、
真实硬终止、clip truncation、FinalState 与 provider/logging/checkpoint 契约。
模型采用显式增量输出归一化；actor 使用小输出初始化，critic 输入和回报目标分别
归一化。混合精度只用于神经网络，状态积分、FK、奖励与归一化保留 float32。

TerminationTerm 可声明同参数的 `probability_dispatch`，由 resolver 校验、modeling
执行；真实 executor 仍执行原硬终止。modeling 的 `evaluate(soft_termination=True)`
以 `1-product(1-p)` 组合概率，未声明软形式的 term 回退硬概率。任务拥有阈值、
温度与误差数学，trainer 只消费概率，截断仍保持布尔 bootstrap 语义。

新训以全 LAFAN1、现有奖励与硬终止条件验收，不复用训练权重；replay 窗口就绪后
执行联合模型与策略更新。效果需与固定时间预算的真实 rollout
对比，不能从想象奖励或训练 loss 推断收敛。

优化器按参数职责分组：中间 Linear
权重使用 PyTorch 公开 Muon（`match_rms_adamw`、无 weight decay），首尾投影、bias
和归一化参数使用 Adam；critic 仍用 Adam。不跳过非有限梯度。
默认使用 Muon 和学习率 4e-4；每个采集步积累 0.4 轮更新额度，每轮包含
10 次 WM 与 1 次 policy 更新。本机 task 保留较小网络、batch 与 replay 容量。

critic 的各时刻 bootstrap 合并为一次 batch forward，保留 survival 梯度、真实
截断终态和原目标；优化前后的目标、梯度及一次参数更新已作数值对照。Torch CPU
线程预算由 worker 的显式 CPU 配额或启动环境控制，APG 不持有独立线程数超参数。

世界模型使用速度增量加解析积分，goal 使用十帧、间隔五帧的未来窗口。


### MotrixSim read freshness 验收

当前 MotrixSim 原生 `QueryProgram.execute` 只收集缓存。后端 read program 对 link、
geom、site 位姿/速度查询先调用 `SceneModel.forward_kinematic`；partial read 仅刷新
请求行。DoF、control 与接触报告查询不额外刷新；刷新不积分、不改变 q/qd。
真实 post-step 查询必须与同一 q/qd/root 状态的可微 FK 相符，不能仅在 reset 后验证。

## APG 配置与调度契约

配置与 FastSAC 共用 `device / agent / trainer` 分层：网络、AMP、replay 与更新参数归
agent，运行预算归 trainer，CPU 预算归 worker 放置，日志/checkpoint/seed 保持框架根级。WM 的监督
参数单独置于 `agent.world_model`。`num_updates` 为每个采集步对应的联合 learner 轮数，
允许小数；每轮执行 WM 的 num_updates 次优化及一次 policy/critic 优化。累计配额调度
保持长期 UTD，避免将参考 0.4 舍入成固定每 4 步一轮。计数与 FastSAC 一致：
num_learning_iterations、日志与保存 interval 均以采集步为单位。

当前 Hydra 配方：历史 10、WM 窗口 8、policy horizon 32、
Muon 4e-4、UTD 0.4、每轮 10 次 WM、policy EMA 0.995，以及 250000 learner 轮
余弦学习率衰减至 3%。填满 max(history, WM horizon) 后开始积累更新配额，不设
WM-only 阶段；显式 learning_starts 可延长 replay warmup，恢复后重新填充 replay。
EMA 只用于推理 artifact，采集与优化使用实时 actor；training checkpoint 保存实时
权重、EMA/优化器、完成轮数与剩余配额。旧 checkpoint 的推理规格仍可读取。

本机 recipe 只缩小网络、batch 和 replay 容量；物理、奖励和目标窗口沿用本仓库
既定契约。同步路径采用 FastSAC collect/learn 分阶段计时，标量在日志窗口统一读取，
异步路径只暴露已实现的单 collector host transport 参数，不暴露未实现的 IPC/DDP 开关。

### 同步与异步采样执行

APG 的同步与异步执行共用采样、序列组装、learner 和配额计算。异步模式
`algo.asynchronous=true` 将环境与独立推理 actor 放入 spawn collector 进程，主进程负责
learner、日志和 checkpoint。它复用 `motrix_rl.transport` 的 `HostTransitionRing`、
`Control` 与 inference bundle/weight channel，不复制 FastSAC 的 SAC replay 或 DDP worker。
真实环境数据在 CPU 产生，先采用 host ring；不新增未实现的 IPC/DDP 配置开关。
小型日志元数据、低频课程/RNG 快照走单向 Pipe，复用 ring 的有界背压，不另建带信号量的 Queue。
退出沿用 FastSAC 的 join → terminate → kill 阶梯，避免异常路径遗留 GPU 进程。

ring 的容量限制采样超前；消费按原始 step 顺序推进，不能跳过终态或合并 learner 配额。
每条 transition 携带原始 observation 用于更新归一化，actor 参数和两个 normalizer 通过同一
版本发布。collector 参数使用公共 `bind_flat_params` 绑定连续存储，receiver 一次 H2D
加载完整 actor，避免逐参数阻塞传输。warmup 阶段也发布归一化，学习率、EMA 和 UTD 时钟保持 learner/collector 原有口径。
checkpoint 的课程状态在 collector 对应 step 截取，不能从主进程的建模环境保存旧状态。
主进程只创建单 lane 的建模环境，实际采样 lane 属于 collector；resume 重新填充 replay。
collector 的动作噪声 RNG 随对应步的课程快照保存/恢复；模拟器状态与 history 仍从 reset 开始，
因此 resume 不承诺逐轨迹续接。权重版本滞后、ring 填充量及 collector 实际采样耗时单独记录。
权重 publish 计入 learner 阶段总时间，同时单列 publish 和 collector load 指标。
异步模式不支持交互 render；同步模式保留这一用途。异常、提前停止必须关闭并回收 collector。

性能验收需要比较相同 seed、语料、网络、batch、UTD 的同步/异步 wall time 与成功率，
不能以采样 SPS 替代达到同等质量的耗时。异步模式为显式选项，在实测通过前不改变任务默认模式。

### 文件职责与保留依据

以下覆盖 APG 分支相对上游新增的运行时、配置、测试与文档。文件以实际责任划分，
不为尚未实现的多 learner/GPU IPC 再建目录。`collector.py` 同时拥有采样时钟的
`UpdateSchedule`，避免仅为一个短小计数器单独保留文件。

| 文件（相对对应 package 的 `src/<package>/`） | 单一职责及使用方 |
| --- | --- |
| `motrix_rl_builtin/apg/__init__.py` | Python package 标识；不聚合内部符号。 |
| `apg/provider.py` | 插件注册与 trainer/exporter 创建边界。 |
| `apg/config.py` | 类型化 agent/trainer 配置及用户输入校验。 |
| `apg/train.py` | 统一训练主循环、日志/live panel、checkpoint/resume、play；不实现 IPC 协议。 |
| `apg/collector.py` | 真实环境采样/history、采样时钟 UTD 配额、可选 spawn collector 生命周期；两种执行模式共用采样实现。 |
| `apg/bridge.py` | 声明式 query 到 GenCoord/FK 的映射，以及真实终态/外生状态的 transition 组装。 |
| `apg/dynamics.py` | 结构化世界模型、积分和状态监督损失。 |
| `apg/networks.py` | Motion actor、critic、history 和可微归一化。 |
| `apg/optim.py` | Muon/Adam 参数分组与组合 optimizer 状态；不能直接以 FastSAC Adam 替代。 |
| `apg/learner.py` | WM 学习、想象 BPTT、critic target、EMA 与学习率时钟。 |
| `apg/export.py` | ONNX/artifact 导出适配，不参与训练调度。 |
| `motrix_rl/numa.py` | FastSAC/APG 共用 CPU 配额解析、进程绑定、spawn 初始配额恢复及预算日志；模块导入不加载 Torch。 |
| `motrix_rl/replay.py` | 通用、episode 边界正确的 sequence replay；与 SAC transition replay 语义不同。 |
| `motrix_rl_builtin/apg/networks.py` | 算法内部的归一化统计与可微输入契约。 |
| `motrix_rl/console/metrics.py` | 公共 live metric 字段与系统指标；FastSAC/APG 共同使用。 |
| `motrix_env_core/manager/modeling.py` | 从 Manager term/query 声明派生可微建模程序；不依赖 APG。 |
| `manager/exogenous.py` | 外生 channel 的声明、时间推进契约。 |
| `manager/truncations.py` | 与失败 termination 分离的时间/片段结束生命周期。 |
| `motrix_envs/locomotion/wbt/g1/supermimic.py` | G1 WBT 内的 SuperMimic 环境配置及注册。 |
| `locomotion/wbt/mdp/supermimic.py` | 跟踪观测、奖励、termination、motion command 与课程采样。 |
| `motrix_rl/configs/algo_base/motrix.apg.yaml` | 与 dataclass 对应的完整算法默认配置。 |
| 根 `configs/task/g1-wbt-supermimic/motrix.apg.yaml` | 当前机器规模的 task 差异，不复制完整算法默认值。 |
| 根 `configs/experiment/g1-wbt-supermimic/apg-server.yaml` | 显式服务器全量网络、batch/replay 与换算后的训练预算；不承载硬件编号或新 provider。 |

`motrix_env_core/manager/rand.py` **不是 APG 新增文件**，保留。`mdp/state.py` 建立每 lane RNG，
`manager/context.py` 向 term 提供随机数，`manager/modeling.py` 选择独立随机轨迹或确定性中点。
删除它会破坏真实环境 reset/noise 以及 imagination RNG/checkpoint 契约。它的 SplitMix64
可编译整数溢出修正属于共享运行时修复，不能改成依赖全局 Torch RNG 的临时替代。

长期测试职责：`motrix_env_core/tests/test_manager_modeling.py` 保护派生/外生/梯度契约；
`motrix_envs/tests/test_supermimic_terms.py` 保护任务的可微跟踪语义；
`motrix_rl/tests/test_replay.py` 保护序列边界；
`motrix_rl_builtin/tests/test_apg.py` 保护归一化统计、梯度、数值、inference/checkpoint，以及标为 integration 的
spawn 数据顺序、课程快照和异常退出。GPU 吞吐/收敛比较是专用实验，不混入默认单元测试。

文档职责：本文维护 trainer 设计；[Manager 建模](manager/modeling.md) 维护环境框架边界；
[序列数据平面](rl-sequence-data-plane.md) 维护 replay/transport 数据契约；
[实施计划](../plan/apg-supermimic.md) 记录 TODO 和实际验收证据。索引只负责导航。

### 服务器全量配置与资源分析

本机 task recipe 保留缩小网络与 replay/batch 的默认值；服务器通过显式 Hydra experiment
设置全量网络、batch、4096 envs 和 1000-step replay。预算为 12000 learner rounds；
按 UTD=0.4 与 10 步 warmup 换算为 30010 collector steps。
experiment 只表达当前全量训练需求，不增加 trainer 类型或伪造 train_backend 变体；公共算法字段仍在 agent/trainer。
硬件编号、CPU affinity 不写死在任务 YAML 中，由运行时约束选择同 NUMA 节点内的资源。

性能改动以 CPU/CUDA profiler 为依据，区分 policy imagination/BPTT、WM、归一化/ingest、
collector 和发布。保持 UTD、horizon、奖励和终态语义，比较相同规模下的数值、梯度、吞吐与显存。
优先减少想象 step 的 kernel 碎片与多余 host 同步，不以牺牲更新次数提高 SPS。

FK 编译路径把固定 3×3 matrix/vector 乘法展开为可融合的逐元素运算，eager 路径保留 BLAS。
这属于共享 `math/kinematics.py` 的执行优化，不新增第二套 FK 或 APG 专用运动学实现；
以完整位姿/速度输出及所有输入梯度与 eager 路径的一致性保护数学契约。

### 更新编译与工作状态生命周期

WM/critic 的数值 forward、loss 与 backward 构成更新编译边界；采样、梯度裁剪、
优化器 step、EMA 与指标记录留在 eager 编排。CUDA Graph generation 仅在上一次
backward 完成且跨调用输出取得独立所有权后推进。模型对象仍是 checkpoint 和
optimizer 的唯一所有者，compiled callable 不成为第二套模型。

想象单步复用已解析 Manager term：静态字段结构在 fork 时确定，可变 command/action
与 RNG 显式沿步骤传递，保护先前步骤 autograd 所需的存储。优化不得通过 detach 或
任务侧奖励副本绕开单源数学。单步普通编译先于整段 horizon capture；单步结果在整个
BPTT 期间存活，因此不能逐 imagined step 重置 CUDA Graph generation。

CPU 预算属于 worker 放置。APG 采用与 FastSAC 同名的 learner_cpu_cores /
collector_cpu_cores；显式 CPU 配额设置进程 affinity 及 Torch 线程预算，未指定时保留
启动环境设置。公共 helper 归 motrix_rl，provider 不互相导入私有 worker 实现。


`ModelingProgram.transition` 接收 command/action/RNG 工作状态，返回 observation、reward、
termination、truncation 与下一工作状态。固定树的复制构造器及外生 channel 访问器在
workspace 构建时解析；真实 term 调用不变。action receiver 在 fork 时复制，调用中临时
绑定工作状态并恢复，避免在 compiled region 中调用 Python copy.copy。输入工作状态不被
原地推进，轨迹完成后显式提交 RNG 与工作状态供 checkpoint 使用。

WM/critic 输出在 compiled region 外 clone 后才进入指标窗口，以跨 CUDA Graph generation
保留所有权。imagination 单步只启用普通 fullgraph 编译：其输出需要存活到整个轨迹反向传播。
`agent.compile=false` 仍执行同一数值路径的 eager 版本。


归档 run 的类型化恢复在框架 read_task_config 边界调用 provider 配置类的
migrate_run_config；APG 仅移除旧 trainer.torch_threads 字段。迁移先于用户覆写与
schema 校验，不修改归档文件，也不接受旧 CLI 字段别名；播放、导出与恢复不因配置清理失效。


公共 CPU 放置的实现位于 `motrix_rl/numa.py`：解析与约束 CPU 集合、应用 affinity/线程配额、
spawn 时临时恢复父进程原始 CPU allocation，随后恢复 learner 配额。FastSAC 使用相同 helper，
保留其原有解析与 best-effort 绑核行为。APG collector 在子进程入口记录自己的实际预算；
未改变 affinity 时 spawn 不需要调用绑核 API。线程数不是 APG 算法正确性的前提，默认继承
环境，显式资源隔离按 CPU 配额生效。性能数字和质量验收统一记录在实施计划，不成为默认配置常量。


### Actor 数值轨迹与输出所有权

actor 数值轨迹与训练编排分离：rollout、bootstrap 和 loss 构成一个数值函数，backward
在单独入口执行；采样、空批次判断、参数冻结、裁剪、optimizer/EMA 与 critic 更新保留在
外部。数值内核仍使用单步编译，不编译整个 horizon 或多步块；完整执行捕获见下节。

每次 actor 更新入口推进一次 graph generation。返回的 critic 数据、指标和 Manager 可变
工作状态取得独立存储后，再进入后续 critic graph generation。提交工作状态不保留已完成
backward 的 autograd 图；保持原有 critic normalizer 更新时序和 checkpoint keys。


### 整轨迹 CUDA Graph 执行捕获

复用单步 Inductor 前向/反向内核，先在独立 CUDA stream 上 warmup 完整 actor backward，
再捕获整条轨迹的执行，不扩大 AOT 图。固定 aux/rep/history/observation/accepted 与 Manager
工作状态的可变 Tensor 地址，每次 replay 前填入新值；不可变 motion assets 共享原存储。
捕获区域不包含采样、optimizer、EMA 或 critic 更新。第一次 capture 后也必须 replay；
warmup 不提交 workspace、不更新参数或归一化。每轮返回结果仍按已建立的所有权规则复制。
`agent.compile=false` 与 CPU 保持普通执行，不增加自动失败降级或配置开关。

cuBLAS 工作区必须在 actor 图池内分配。capture 前后使用本轮验证的 PyTorch 2.11 所用的
`torch._C._cuda_clearCublasWorkspaces()` 协议清理缓存，与 Inductor CUDAGraph Trees 一致：
否则后续 WM/critic 自动图录制会释放预热工作区，actor 第二次 replay 访问失效地址。
此调用只出现在一次性捕获阶段；升级 PyTorch 时必须运行真实 CUDA 图交替录制回归。
运行时捕获与 staging 留在 `apg/learner.py`，Manager 工作状态地址/资产共享由
`manager/modeling.py` 管理；不增加独立运行时文件或新的配置分支。

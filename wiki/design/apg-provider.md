# APG provider 与 sync trainer 设计（P3.2）

## 摘要

三线在此汇合：`motrix_rl_builtin` 新增 `apg/` provider（算法核心自
`MotrixLab_mbrl_manager` 前一尝试回收，传输栈与 TaskModel 层不回收），
`MotrixFramework` 聚合双 provider，WBT 家族升格外生注记并新增
g1-supermimic 单文件变体。按 S-C 冻结为**单 commit**；S-A：algo 名 `apg`
（rllib `motrix`），metrics 标量键 `apg.` 前缀，dashboard family
`rl.worldmodel`（P3.1 已注册）。依赖 P2（modeling view / FinalState /
truncation）与 P3.1（schema / SequenceReplay / EmpiricalNormalization）。

## 前一尝试回收边界

| 前一尝试（`mbrl/`） | 处置 | 去向 |
|---|---|---|
| `networks.py`：MlpDynamics / FloatingBaseDynamics（so(3) 对数/指数映射、heading-frame 不变特征、error floors、in/out RunningNorm、零初始化输出层）、MlpActor / MlpCritic | **回收** | `apg/networks.py` 原数学保留 |
| FloatingBaseDynamics 内嵌的 exogenous 推进（increment/copy/constant 写 state span） | **删除** | 职责移交 `ModelingProgram.advance_exogenous`（P2.3 D-B）；dynamics 只预测物理块 |
| `learner.py`：三阶段梯度契约（WM 监督 detached targets；actor BPTT 冻结 WM/critic 参数但梯度过输入；critic lambda-return 回归 detached rollout）、reward/gate 扁平化单遍（`(H*B)` 行一次 evaluate）、survival 加权 horizon return、critic epochs/minibatches、共享 cosine lr 衰减（→3%） | **回收** | `apg/learner.py`，`task.*` 调用替换为 program API |
| `train.py`：TrainerBase 骨架、UTD 交错循环（fractional steps 累加器）、checkpoints/panel/事件 | **回收** | `apg/train.py`（现契约已与该骨架一致） |
| `collector.py`：actor 面向循环（obs 帧栈、episode 起点广播填充、高斯探索噪声×action_limit、norm 仅真实采集更新、episode 窗统计、deterministic evaluate） | **回收重写** | `apg/collector.py`，recorder/adapter 换 modeling view 装配（见 P-D） |
| `GoalConditionedActor`、`RunningObsNorm`、`normalizer.py` | **不回收** | goal 经 obs 组进入 actor 输入（单源 observe）；归一器用 P3.1 共享 `EmpiricalNormalization`（D7） |
| `TaskModel`/adapter/recorder（`motrix_env_core.modeling` 包）、`async_collection`/`async_transport`、`ActorRolloutGraph`/compile、`panels.py` | **不回收** | 派生式取代（P2.3）；async 属 P4；compile 见 P-H；panel 按新指标名另写 |

## 冻结决策

### P-A provider 身份与装配

`ApgProvider(AgentProvider[ApgCfg])`：`agent_name="apg"`、
`train_backend="torch"`、`checkpoint_format="pt"`。`MotrixFramework` 聚合
`(FastSacProvider(), ApgProvider())`；删除空置 `mbrl/` 目录。默认任务配置
`configs/task/g1-wbt-supermimic/motrix.apg.yaml`。metric 标量键 `apg.`
前缀（`apg.wm_loss`、`apg.policy_loss`、`apg.imag_reward`、`apg.value_loss`、
`apg.rollout_*`、`apg.eval_*`）——替换前一尝试的 `wm/loss` 式命名。

### P-B 动力学状态组装（FieldDesc 驱动 + FK 派生）

modeling state 是**命名字段集**（声明 read 集），dynamics 预测**物理块**：

- 物理块布局沿用前一尝试约定 `[pos(3) | quat(4) | lin_vel(3) | ang_vel(3) |
  qpos(N) | qvel(N)]`。组装规则：`LinkPosition/Quaternion/LinearVelocity/
  AngularVelocityQuery` 的 batch 字段按 **FK 树根行**切出 root 块——行号 =
  tracked 集内 base link 的行（g1 为 row 0 `pelvis`；装配期以
  `tree.link_names[0] ∈ tracked` 校验，跨四个 body query 行号一致）；
  `JointPosition/VelocityQuery` 字段给关节块（任务关节序 → 树 q 序按名置换，
  未覆盖且影响 tracked 行的树关节报错）。`apg/assemble.py` 产出布局描述
  （纯数据），networks 按其构造。
- **其余声明字段不进 dynamics**：tracked body 全行等 FK 可达字段由
  **KinematicTree**（#163：MotrixSim loader `kinematic_tree(model, base_link,
  device)`，D-A 共享常量语义）从物理块**精确可微地派生**，并入完整
  sim_state dict 供 `program.evaluate/observe` 消费。WM 只学动力学，
  运动学不学（FK parity fixture 锁定派生值 == sim read 视图）。外生通道
  字段（P-D）同样不是 dynamics 输出——由 `advance_exogenous` 规则推进，
  窗口监督在物理块上。
- `scheme ∈ {residual, floating_base}` 保留：`residual` 为平铺 MLP 直接
  预测全部字段（无 FK 结构任务）；`floating_base` 为上述结构（g1 用）。

### P-C 想象循环（program 消费）

每想象步：`actor(norm(history))` → `action_limit * raw` → `dynamics(物理块,
a)` → FK 并回完整 state dict → `program.evaluate(state, a)`（reward /
terminated / truncated；survive gate = `~(terminated | truncated)`）→
`program.advance_exogenous(a)` → `program.observe(state)` push 帧栈。H 步
unroll 后 reward/gate 扁平化为 `(H*B,)` 一次 evaluate（FK 重的 term 数学
批量化，前一尝试已验证的吞吐形态）。

**想象起点重建**：起点 state 来自 `SequenceReplay.sample_starts`；其外生通道
字段值（如 `steps`，schema 状态字段之一，见 P-D）直接写入 shadow command
host 对应叶（`program._host.commands[...]` 的 per-env 叶按 channel 名对拷），
shadow action 工作记忆同理取自起点槽的 `previous_action` 语义（断链置零 +
`intact` 掩码折叠进接受计数，P3.1 契约）。此后 `evaluate` 的 command update
hook 在想象态上重算派生叶（yaw-align 等），`advance_exogenous` 从正确帧位
推进——`sync_from_env()` 仅服务"从当前真实态续想"的场景，历史起点不走它。

### P-D 转换装配器（collector，无 recorder）

sync trainer 构造 env 时注入 `capture_final_state=True`（S-B）。每步：

1. pre-step 克隆声明 read 集全行（`sim_data.view(key).clone()`）为 `s_t`；
2. `env.step(action_t)` → `state`（reward/dones）+ `FinalState`（done 行终态）；
3. `next_*` = post-step `sim_data`（活行）∪ `FinalState.sim`（done 行）；
4. **外生通道字段**：`state_fields` 除 read 集外含外生通道（P3.1 T-D 的
   "read 集 + 外生通道"）——装配器每步自常驻 command term 的 per-env 叶
   （如 `steps`）取值，`next_*` 取 post-step term 值（advance 已执行），
   done 行取 `FinalState.commands`（reset 前终值）；
5. `wm_transition_schema(state_fields, act_dim, obs_dim=policy 组宽)` 装配
   batch → `SequenceReplay.extend`；policy obs 存 `state.obs[组名]`。

actor 输入 = `policy` 组 obs 帧栈（cfg `actor.history`）；critic 输入 =
`value` 组（两组均经 `program.observe` 在想象侧同源生成，P-E）。

### P-E obs 组与归一化

APG cfg 声明组名（默认 `policy`/`value`）。想象侧 obs 由
`program.observe` 生成，噪声确定性（D-A rand 语义：`next_uniform`→零）；
真实侧 obs 噪声照常（收集即含噪）。`EmpiricalNormalization`（obs / value
两套）只从真实采集更新，想象与评估冻结使用。obs 帧栈断链语义复用
P3.1 `history()`（episode 头广播填充、缺帧重复最早连续帧）。

### P-F WBT 家族升格 + supermimic 变体

- `WbtMotionCommand.steps`：`metric()` **升格为普通 per-env 状态叶**（它
  本就是 reset/advance 写的状态，不是仪表）；`advance` 内一行镜像进新增
  `steps_gauge` metric，保持 `motion_step` 指标发布连续性（基线回归可见面
  不变）。term 函数体不动。
- 家族 cfg 注记 `exogenous = {"steps": ExogenousRule(INCREMENT, step=1.0)}`
  （帧索引每步 +1；clip 结束处真实侧的分枝语义在想象侧由 truncation 门控
  掩除，见下）。
- 新增**家族级** truncation term `motion_clip_end`（读 command `steps` 与
  clip `frame_clip_end`，D5 语义：clip 结束是时间性截断而非失败）；
  supermimic 变体启用（基线变体不声明则行为不变）。
- `g1/supermimic.py` 单文件变体（与 dance.py 同构）：supermimic 语料目录、
  启用 clip-end truncation、**移除 `undesired_contact_forces` 查询与
  `undesired_contacts` reward**（contact 不可想象，P2.4 类型级拒绝的
  任务侧对应——reward 权重回填由变体 cfg 决定）、`sampling_mode="return"`
  课程默认值。
- `reference_index` 等组装元数据由 provider 读常驻 term（P-B），任务侧零
  python。

### P-G checkpoint / play / ONNX（D8）

`POLICY` = 无状态 actor + obs normalizer 统计（复用现有 ONNX exporter 模式
与 parity 校验流程，`apg/export.py` 挂接）；`TRAINING_STATE` = dynamics /
critic / optimizers / normalizers / replay 采样 RNG / iteration / env_steps /
sampler 课程状态（经 command term 的序列化视图，trainer 不感知内部结构）。
resume 语义沿用前一尝试：学习状态精确恢复、replay 重灌（fresh warm-up）。

### P-H compile 不在本 commit

eager v1（对应 D-C 的想象路径决策；前一尝试的 `ActorRolloutGraph`/
CUDA-graph 捕获与 `torch.compile` 窗口损失为后续独立优化域，回收源保留在
前一尝试）。`trainer.compile` 字段预留为 `False` 不生效。

### P-I eval profile（D9）

`trainer.eval_interval / eval_steps / final_eval`（trainer cfg，非框架
概念）+ deterministic 评估（无探索噪声、norm 冻结、`track_norm=False`
语义沿用）。评估走独立 env（eval 覆盖项 = `for_play()` 式 cfg 变体的
train-mode 拷贝）。

## 文件改动

| 文件 | 动作 |
|---|---|
| `motrix_rl_builtin/.../apg/__init__.py` | 新增：包导出 |
| `motrix_rl_builtin/.../apg/config.py` | 新增：`ApgCfg`（自 MbrlCfg 改造：去 `task_adapter`/async 字段，加 obs 组名与 eval profile） |
| `motrix_rl_builtin/.../apg/assemble.py` | 新增：FieldDesc→物理块布局、KinematicTree FK 派生、完整 state dict 组装 |
| `motrix_rl_builtin/.../apg/networks.py` | 新增：Dynamics/MlpDynamics/FloatingBaseDynamics（去内嵌 exogenous）、MlpActor、MlpCritic |
| `motrix_rl_builtin/.../apg/collector.py` | 新增：转换装配器 + actor 循环 + evaluate |
| `motrix_rl_builtin/.../apg/learner.py` | 新增：三阶段学习者（program 消费） |
| `motrix_rl_builtin/.../apg/train.py` | 新增：sync Trainer（UTD 交错循环、checkpoint、panel） |
| `motrix_rl_builtin/.../apg/export.py` | 新增：ONNX exporter 挂接 |
| `motrix_rl_builtin/.../apg/framework.py` | 新增：`ApgProvider` |
| `motrix_rl_builtin/.../fastsac/framework.py` | 修改：`MotrixFramework((FastSacProvider(), ApgProvider()))` |
| `motrix_rl_builtin/.../mbrl/` | 删除：空置目录 |
| `motrix_envs/.../wbt/mdp/command.py` | 修改：`steps` 升格 + `steps_gauge` 镜像 |
| `motrix_envs/.../wbt/mdp/terminations.py` | 修改：`MotionClipEndTruncationCfg` |
| `motrix_envs/.../wbt/g1/common.py` | 修改：exogenous 注记（family 级） |
| `motrix_envs/.../wbt/g1/supermimic.py` | 新增：单文件变体 + env 注册 |
| `configs/task/g1-wbt-supermimic/motrix.apg.yaml` | 新增：默认训练配置 |
| `motrix_rl_builtin/tests/test_apg_*.py`、`motrix_envs` 升格回归 | 新增：见验收 |

## 验收

- **FK parity fixture**（P3.2 验收线，P2.4 遗留）：真实 env 上
  tree FK(物理块) == `tracked_body_*` sim read 视图（逐行 allclose）；
- **升格回归**：基线 WBT env（dance/backflip/largebox）model-free
  （fastsac）训练行为一致，`motion_step` 指标发布连续；不声明 exogenous
  的既有任务零改动；
- **g1-supermimic 端到端**：有限损失收敛（uniform 起步，切
  `sampling_mode=return`），`mxlab play` 与 ONNX parity 通过；
- **装配器契约**：done 行 `next_*` 为 pre-reset 终态（FinalState），
  活行为 post-step 读集；replay 窗口监督闭环（P3.1 契约在真实任务上消费）；
- **想象循环**：program.evaluate/observe 在 `(H*B)` 批量下数值与逐步
  调用一致；survive gate 来自 terminated/truncated（D5 区分）。

## 实施修订（落地时定案）

1. **modeling.py 扩展**（`motrix_env_core`，P-C 的必要前提）：想象批次宽度
   B ≠ env lanes，`ModelingProgram.resized(B)` 派生批量尺寸的兄弟 host；
   `seed_exogenous(values)` 从 replay 存值种入外生通道；BPTT 下 shadow
   工作记忆的 in-place 写触发 autograd 版本冲突，
   `rebind_shadow_state()`（`detach` 可选）每想象步换全新克隆——每张量
   写入恰好一次，梯度仍过克隆回传。int64 通道的 increment 步进取整校验。
2. **想象 v1 逐步执行**：未做 (H·B) 扁平化 evaluate（eager 下为吞吐优
   化，非正确性项；copy_action 型通道的扁平化还需逐步推进语义），随
   P4 compile 域一并评估回收。
3. **FK parity 定形**：步进态**位置**精确（≤2e-2，求解器容差内）；
   步进态**速度**视图与位置视图存在积分器半子步相位差（backend 采样
   性质：推入重构快照后速度 parity 精确）——fixture 双线锁定，速度
   相位差记录为 MotrixSim 性质，不阻塞（想象侧由物理块组合出的速度
   与块自身相位一致）。
4. **D8 偏差**：motion sampler 无序列化 API，课程统计不随 checkpoint
   持久化——resume 以 fresh warm-up 重启课程统计（一处已知偏差）。
5. **外生注记落点**：`WbtMotionCommandCfg` 默认（family 一处，全部变体
   继承），非 g1/common.py；`steps_gauge` 由 cfg `__call__` 显式构造
   （直接构造不经 manager 也成立）。
6. **supermimic 语料**：目标语料未入 assets 前，变体默认指向 dance
   目录（cfg 可覆盖）；端到端验收跑真实语料时替换。
7. **P-A 落地**：metrics 键 `apg.` 前缀（`apg.wm_loss` 等），dashboard
   family `rl.worldmodel`（P3.1 T-F）。

## 与 P3.3 / P4 的边界

- **P3.3 go2**：零 trainer 改动的第二任务（`residual` scheme + quadruped
  家族，Direct/Manager 变体并列）；本设计的组装规则不得含 g1 特例；
- **P4 async**：`async_collection`/`async_transport` 按新契约重写（WM schema
  ring + 块式入栈），sync/async 共用本设计的装配器与 learner；
  `ActorRolloutGraph`/compile 届时一并评估回收。

# Manager 双路径建模设计（P2.3）

## 摘要

任务数学单源的机制落地：resolver 产出的 `ResolvedManagerProgram` 是纯数据（typed
call records + 参数描述符），**同一份 call records 可以绑定第二个 shadow
`ManagerContext` 再执行一次**——真实路径绑定常驻 context（sim 视图 = read program
arena、commands = 常驻 term 实例），想象路径绑定 modeling state 张量束支撑的
shadow context。派生产物 `ModelingView` 经 `ManagerEnv.modeling_view()` 暴露
（D3：APG 可训练性是 env cfg 的派生属性，任务侧零 python）。

本设计是 [APG 路线图](../wm-bptt-trainer.md) P2.3 的实施前定案，冻结 P2 评审
遗留的 shadow 规则；执行边界与 term 协议见
[Manager Runtime 设计](./runtime.md)与 [Torch 执行器设计](./torch-executor.md)。

## 冻结决策（P2 评审）

### D-A shadow kernel_data 拷贝/共享规则

按**结构判定**（与 FinalState 的行判定同构，零注记）：

| kernel_data 叶子 | 判定 | 想象侧处理 |
|---|---|---|
| 前导维 == num_envs 的张量 | per-env 可变状态（reset/advance 会写） | **shadow 拷贝**：独立同形张量，想象侧写不回漏真实侧 |
| 共享张量（clip 表 `(F,J)`、FK chain）与标量超参 | episode 无关常量 | **同对象只读共享**：不复制、不重采样 |
| `metric()` 声明字段 | 诊断仪表，非状态 | shadow 侧独立零缓冲作可写 sink，执行后丢弃 |
| ActionState 张量叶子（queue、共享 cursor、自定义行） | 全部为工作记忆（非 episode 常量） | **全量拷贝**——共享 cursor 若共享，想象侧推进会污染采集侧历史 |

- 共享只读不靠运行时拦截（写入共享对象不会报错），由验收兜底：想象侧执行后
  真实侧常驻状态必须逐位不变（fixture 强制）；
- `ctx` 控制面：`rand` 确定性（`next_uniform` 返回零、`uniform_range` 返回区间
  中点，即"obs 噪声关闭"语义；reward term 内消费 `ctx.rand` 的罕见情形同此
  语义）；`dt`/`env_id` 同值；`sim_reset_requested` 恒 False；
  `reset_ids`/`reset_episode` 为空（想象侧不跑 reset kernel）；
  `terminated`/`truncated`/`reward` 镜像为 shadow 独立缓冲。

### D-B 外生通道注记 API

`CommandCfg` 增 `exogenous: dict[字段名, ExogenousRule]`（默认空，per-field
声明）。`ExogenousRule` 是**单一 configclass + `ExogenousKind` 枚举判别**
（`increment`/`copy_action`/`constant` 三型，`step`/`action_term` 为各型参数）
——不取 union of classes 形态：env cfg 树会被转为 schema-driven structured
config，其值类型不能是 union（实施期由回归发现并定案）：

- `increment(step)`：每步递增（motion phase / 帧索引）；
- `copy_action(action_term)`：由 policy action 决定（velocity command 类）；
- `constant`：episode 内常量、reset 时重采样（终值随 FinalState 捕获，
  learner 接续）。

increment 的想象侧推进由 `ModelingProgram.advance_exogenous(action)`
执行，与真实 command `advance` **数值等价**（fixture：H 步递增对齐真实 env）。
注记描述的是想象语义，不改变真实路径行为；`constant` 型的 reset 采样注记
落点（reset/randomization term cfg）随 P3.2 的家族升格需要再落。

### D-C 编译策略

本 commit 只实现 **eager** 想象路径（正确性优先）；`torch.compile` 为后续
独立编译域优化（guard 落在 shadow 宿主对象 identity 上，与真实 kernel 编译
缓存互不干扰），不阻塞验收。

### D-D dry-run 报错契约

派生失败在 `modeling_view()` 调用处抛出，首个失败点即止，信息格式：

```text
modeling derivation failed: reward.<term> accesses ctx.sim["<key>"]
  which is not part of the modeling state (declared read set + exogenous channels)
```

包含 group、term 名、被访问字段名；不聚合多个错误（保持定位简单）。

### 已冻结的配套决策（S 项）

S-A provider id `motrix.apg`、metrics family `apg.`（P3 生效）；S-B
FinalState 捕获为 APG trainer 构造注入（已落地，见
[终态结果契约](../array-env-final-observation.md)）；S-C P3.2 按路线图单
commit；S-D 外生注记取 per-field 映射（即 D-B）。

## 派生机制

### shadow context 组装

`manager/modeling.py` 构造第二个 `ManagerContext` 实例：

- `sim`：modeling state 张量束的寻址视图（字段名 ≡ 声明 read 集，形状取自
  read program 声明；语义类型由 query 类携带，见 FieldDesc）；
- `commands`：常驻 command term 的 shadow 拷贝树（按 D-A 规则逐叶判定），
  Map 键与真实侧一致——dispatch 代码对 ctx 的访问路径不变；
- `actions`：shadow ActionState（action_queue 等 per-env 叶子拷贝；想象步的
  原始动作由 trainer 写入后经同一 `action_process` 处理）；
- `metrics`：D-A 的零缓冲 sink；
- `rand`：确定性实现（零填充，原地推进语义保留但不影响数值）。

### group 级 all-or-nothing

可想象性以 **group 为原子**：reward 组、各 observation 组、termination 组、
truncation 组要么整体可派生，要么整组不可想象（混合会使 reward 加权和残缺、
观测不全）。**v1 取严格语义**：任一 group 派生失败即中止整个 `modeling_view()`
——不存在部分可想象的 view（部分 view 的消费者尚不存在，不为其预扩展契约）。
command `update` hook 一并参与双路径（WBT 的 yaw-align 读 root pos/quat，均在
read 集内则可派生）；command `advance`/`reset` hook 不参与——想象侧由外生
规则与 trainer 的 episode 语义替代（clip 结束映射为 truncation 而非重新采样）。

v1 不引入 `@modeling` 显式标记：执行器直接从既有 program 尝试派生，dry-run
成功与否即 capability。

### dry-run enforcement

哑 modeling state（零张量、形状同 FieldDesc）上执行全部派生调度一次：

- 字段访问核验：访问 shadow 未提供的字段（非 read 集字段、非外生 command
  字段）按 D-D 格式抛错；
- group 完整性、obs 宽度、reward 权重聚合（复用 cfg reward group 权重表，
  同一张量，不复制数值）一并校验；
- 派生不检测数值等价——等价性由 fixture 数值一致性测试兜底。

### ModelingView 与访问器

```python
@dataclass(frozen=True)
class ModelingView:
    fields: Mapping[str, FieldDesc]        # read key → query 语义类型 + shape + dtype
    exogenous: Mapping[str, ExogenousRule] # 外生通道名（含所属 command）→ 规则
    program: ModelingProgram               # evaluate / observe / advance_exogenous（eager）
    action_process: Mapping[str, Callable] # 复用 env.action_terms[].process（同一函数对象）
```

`ManagerEnv.modeling_view()` 首次调用触发派生 + dry-run 并缓存；无需子类
override。trainer 消费面：networks 按 `FieldDesc` 组装动力学结构（D4 项下），
learner 用 program 在想象态上算 reward / obs / survive。

### ModelingProgram 执行语义（eager）

- `evaluate(sim_state, action)` → `(reward, terminated, truncated)`：先把原始
  action 经同一 `process` 实现写入 shadow 动作历史，再按真实 evaluate 的前半
  语句序执行（command update hooks → rewards → terminations → truncations →
  加权总量聚合），**不含 advance**；`sim_state` 是 read-key → 张量的映射，
  每次调用重绑 shadow ctx 的 `sim` Map（trainer 的张量直通，autograd 无阻）；
- `observe(sim_state)` → 各组 obs：与真实 observe 同组同序，噪声确定性；
- `advance_exogenous(action)`：按 D-B 规则推进常驻 shadow 外生通道，数值等价
  真实 advance（fixture 锁定）；
- `sync_from_env()`：想象起点——真实常驻 per-env 状态（command 行 + 全部动作
  工作记忆）拷贝进 shadow，此后想象侧逐步镜像真实侧。

## 与 P2.4 的边界

read query 语义类型的**可想象性声明已落地**：`SimDataQuery` 子类强制声明
`modelable: ClassVar[bool]`（`__init_subclass__` 在类定义期拦截漏声明），
派生在 dry-run 之前做类型级拒绝——read 集含不可想象类型即报错（key + 类型
名）。可微 FK 本体（`KinematicTree`、MotrixSim loader、`KinematicTreeQuery`
声明通道、write-program 烘焙的 FK 刷新）已由 #163 上游化进 core，本设计直接
消费：树作为 D-A 共享常量进想象侧，device 需求经 `KinematicTreeQuery(device=)`
声明满足。"树 FK(模型态) == sim FK(read 视图)"的 parity fixture 随 P3.2 的
第一个消费者（supermimic）落地。

## 文件改动

新增 `manager/modeling.py`（FieldDesc/ExogenousRule/ModelingView/
ModelingProgram + 派生与 dry-run）与 `tests/test_manager_modeling.py`；修改
`manager/commands.py`（exogenous 注记字段）、`manager/env.py`（modeling_view
访问器）、`manager/resolver.py`（暴露 per-term 消费的 SimArg key 集合等派生
元数据；真实路径解析零改动）、`torch_manager/runtime.py`（eager 调度复用：
语句序与参数物化）、`manager/rand.py`（确定性实现）、`manager/__init__.py`
（导出）。`manager/context.py` 与 `motrix_envs` 任务层零改动。

## 验收

- **双路径数值一致**（P2 phase 级验收线）：fixture 任务上，真实 kernel 与
  modeling program 在相同输入下 reward / obs / termination / truncation 输出
  一致（含权重聚合、group 校验、dry-run 全部生效）；
- 外生规则数值等价：increment 通道想象侧 H 步推进与真实 env 对齐；
- shadow 隔离：想象侧执行后真实侧常驻状态（command/action term、read arena）
  逐位不变；
- group all-or-nothing 拒绝与 dry-run 报错定位（term 名 + 字段名）；
- obs 噪声确定性；升格注记不改变真实路径行为（现有 env 全量回归）。

# ArrayEnv 终态结果契约

## 摘要

ArrayEnv 在自动 reset 前捕获当前 transition 的终态，通过 `ArrayEnvState` 发布稀疏行快照，两个契约共用同一捕获点（evaluate 之后、任何 reset 写入之前）：

- **`FinalObs`**（既有，默认行为）：truncated 行的 pre-reset 观测。服务 model-free 消费者（FastSAC bootstrap）；有截断即捕获，`step()` 返回契约不变。
- **`FinalState`**（构造参数 `capture_final_state=True` 显式请求）：全体 done 行（terminated ∪ truncated）的 pre-reset 终态，内容为声明 read 字段行 + 常驻 command/action term state 行。服务序列模型消费者（WM trainer 的转换装配）。请求是 **runtime 注入**——序列 trainer 在构造 env 时传入（与 `seed`/`backend` 同通道），任务 cfg 不声明该行为；未请求时零成本，现 FinalObs 语义原样保留。

两者均不绑定 RL provider；ManagerEnv 实现捕获，provider 只负责传递和消费，不修改环境结果。本文覆盖 Torch 张量前端（DirectEnv/ManagerEnv）。

## 结果结构与生命周期

### FinalObs

`FinalObs` 为普通 dataclass，包含 `env_ids: torch.Tensor`（int64）和 `obs: EnvObs`（policy/value 均为 torch 张量）。观测行顺序与 env_ids 一致，policy/value 保持原始观测结构；没有独立 value 时仍为 None，消费者使用 `value_or_policy`。行张量为独立快照（高级索引生成 owned copy），可跨 step 保留；不需要重复复制 policy 作为 value。消费者不得修改共享结果。

`ArrayEnvState.final_obs` 为可选字段。初始化为 None，每次 step 在推进环境前清空当前结果；此前返回的 state 持有的终态快照不因此清空。step 在 compute_transition、episode_steps 和 truncation 更新后、任何 auto-reset 写入前调用 `_capture_final_obs(env_ids)`。仅有 truncated 行时调用；没有截断或实现不支持捕获时为 None。读取无消费副作用。

ArrayEnv 的默认捕获 hook 返回 None，明确表示当前实现不提供快照；不尝试通过重复调用可能含副作用的通用 compute_observation 自动兼容所有子类。ManagerEnv 使用已有 compiled observe kernel 实现。sim-reset-only 行不进入 final_obs。

### FinalState

`FinalState` 为普通 dataclass：

```python
@dataclass
class FinalState:
    env_ids: torch.Tensor                  # 全体 done 行（terminated ∪ truncated），int64
    sim: dict[str, torch.Tensor]           # 声明 read key → (K, ...) 行快照
    commands: dict[str, torch.Tensor]      # "<command-term>.<leaf路径>" → (K, ...) 行快照
    actions: dict[str, torch.Tensor]       # "<action-term>.<leaf路径>" → (K, ...) 行快照
```

所有行张量与 `env_ids` 对齐（高级索引生成 owned copy，可跨 step 保留，只读）。值的时序语义是 **post-evaluate、pre-reset 真值**：command `advance` 已执行、reset kernel 未执行，command/action 行恰好是外生通道与动作历史的终值——reset 之后这些行即被销毁，这是它们唯一必须在 env 内捕获的原因。

`ArrayEnvState.final_state` 为可选字段，生命周期与 `final_obs` 相同：step 开始时清空，捕获点写入，无 done 行或未请求时为 None。ArrayEnv 默认 hook 返回 None；构造参数 `capture_final_state`（默认 False）为 False 时不调用捕获——请求是 runtime 消费者的构造期注入，任务 cfg 不声明，未请求的环境零成本，现 FinalObs 语义原样保留。

## Manager 捕获语义

捕获读取 compute_transition 已刷新的 simulator 数据，并发生在 transition kernel 之后、command reset 之前。它与同一 pre-reset 状态的正常 observe 使用相同输入；不回退 command，不额外刷新 simulator。

### FinalObs（观测捕获）

Observation term 应仅读取上下文并写观测输出，不推进 command 或历史；现有噪声 term 可以消费 ctx.rand。终态捕获直接使用共享 per-lane RNG，不备份或恢复；额外 observe 消耗噪声随机数，因此可改变同 seed 下后续 reset 和正常观测的随机轨迹，但不要求保持捕获前后的随机序列一致。终态捕获会额外执行一次整批 observe，因此不能用观察函数维护每步只应更新一次的状态。捕获使用独立预绑定 output slot，不写正常 observation 双缓冲；torch 高级索引已生成独立行张量，无需额外 copy。

当前 compiled observe 不支持行子集，因此少量截断仍有整批执行成本。性能优化必须先分别测量 kernel 与 compact row gather 成本，不为了潜在用途新增编译入口。

### FinalState（终态捕获）

- **捕获计划构建一次**：ManagerEnv 构造期（配置请求时）遍历 read program 的全部声明 key（`sim_data.keys` 的稳定 arena 视图）与常驻 command/action term 的 kernel-data 叶子，缓存 `(key, tensor)` 引用。term 是 frozen dataclass、字段构造后不替换，read 视图生命周期与 program 一致，故引用长期有效。
- **行判定**：per-env 行 = 前导维度等于 num_envs 的张量叶子。共享常量（clip 表、FK chain）与标量叶子没有行语义，不捕获；声明 `metric()` 的诊断张量按对象身份排除——payload 携带的是状态通道，不是仪表。
- **无额外 kernel**：FinalState 不运行 observe/evaluate，直接 gather 既有 arena 与常驻张量；成本为每 done 行一次 gather，稳态量级可忽略，大规模同步 reset 步（初始化/课程切换）以基准确认。
- **键空间**：`sim` 以声明 read key（含 term-arg 自动收集的 key）为名；`commands`/`actions` 以 `"<term名>.<点号叶子路径>"` 为名。消费侧（provider 转换装配器）只允许经此契约与声明 read 集取数，禁止绕过声明直读 backend 数据。

### 与 per-term 任务的共存

FinalState 是框架级机制，字段来自任务本就为 term 而声明的 read 集与常驻 term state：per-term 任务零改动、零感知。`step()` 返回契约不变：model-free 消费者消费 obs/reward/dones 与 final_obs 完全不受影响；APG 消费者额外消费 `sim_data`（活行，post-step）+ FinalState（done 行）+ 自持 action，两者共存于同一 env 实例。

## FastSAC 边界

FastSAC adapter 的 step 返回 `FastSacStepResult`，将本步 obs、critic_obs、reward、terminated、truncated、final_obs 与 info 放在同一个结果中；统一的 torch wrapper 直接传递 state 的 final_obs 引用，不在 wrapper 上缓存旁路状态。reset 仍返回 obs/critic_obs 二元组。collector 将 FinalObs 转换为传输所需的 env_ids、policy 与 value_or_policy。异步传输、背压和 replay 存储的生命周期不属于本次 Env 结构整理，见 [FastSAC 异构训练器设计的终态观测 payload](./fastsac-async-heterogeneous-trainer.md#终态观测-payload)。FastSAC 不消费 final_state。

## 验收

FinalObs：初始化、无截断、截断前后观测差异、多行选择、policy-only/value 观测、快照跨步寿命、command 时序和只读重复访问。性能测量区分无截断路径、少量/整批截断的 capture 成本；使用预热后的重复采样，不包含编译时间。

FinalState：默认关闭零足迹（不构建捕获计划、不产生字段）；done 行覆盖 terminated 与 truncated 两种来源；捕获行为 post-advance、pre-reset 真值（reset 后常驻张量已改写、快照不变）；声明 metric 与共享常量不进入 payload；行张量为 owned 快照且跨步存活；sim 行与 read program 声明视图一致；`step()` 返回契约与 model-free 消费者行为不变。

# RL 序列数据平面设计（P3.1）

## 摘要

世界模型（WM）训练所需的**序列数据平面**落在 `motrix_rl`：WM transition 的
schema 模板、episode 边界正确的序列回放共享模块、满足 `NORM_KEYS` 契约的共享
归一器、`apg` 的 metrics family 注册。无 provider、无 env 侧改动，独立可合入。
本设计是 [APG 路线图](./wm-bptt-trainer.md) P3.1 的实施前定案（D6/D7；后继态
语义承接 D4 的 FinalState 定案）。

## 前一尝试的回收边界

`MotrixLab_mbrl_manager` 的 `mbrl/replay.py` 是数学源头，按下表取舍：

| 前一尝试 | 处置 | 理由 |
|---|---|---|
| 加权槽采样（float64 cumsum + searchsorted 全 device） | **回收** | 采样不过主机，已验证 |
| 窗口 chaining / valid mask（首 done **包含**、reset 后尾步 mask、ring-wrap 同法 mask） | **回收** | WM 监督语义核心 |
| `previous_action` / obs 帧栈的 contiguity 语义（断链置零 + `intact` 掩码、历史向 episode 头钳位、缺帧重复最早连续帧） | **回收** | 想象起点契约 |
| `rng_state()/load_rng_state()` | **回收** | D8 TRAINING_STATE 的"序列回放采样器"项 |
| fused `state_dim` 平铺存储 | **改为 schema 命名字段** | 字段集任务派生（read 集 + 外生通道），平铺把布局知识分散到 provider |
| 私有 numpy `TransitionBlock` 进程队列 | **不回收** | D4 定案：传输走 schema-driven `motrix_rl.transport`（sync/async 同一装配路径） |
| collector 侧 episode_id/step_id（worker offset） | **改为 replay 内部派生** | 见 T-C |
| `add`（pinned staging + event 纪律） | **回收、按字段重组** | 复用传输纪律，命名 staging 替代 fused floats 行 |
| `add_batch` 块式入栈 | **缓至 P4** | 消费者（async ring drain）在 P4 才出现 |

## 冻结决策

### T-A 模块落位

- `motrix_rl/replay.py`：`SequenceReplay`（learner 侧 device-resident 存储，
  非 IPC 原语，不进 `transport/`）；
- `motrix_rl/normalize.py`：共享归一器（自 fastsac 升格，见 T-E）；
- `transport/schema.py`：WM schema 模板工厂（传输面声明，与
  `fastsac_transition_schema` 并列）。

`transport/` 维持"进程边界原语"的包语义不变。

### T-B 后继态显式入 schema

WM schema 对每个 modeling state 字段携带**同形 `next_*` 列**：ring slot 自包含
（无跨槽派生），done 行真后继（pre-reset 终态）由 P3.2 装配器从 `FinalState`
写入——装配器本就一步滞后组出完整 transition（`s_t` 缓冲、`s_{t+1}|s_T` 当步
可得）。对照 fastsac 的"后继由后继槽派生 + FinalObsStorage"路线：WM 单步监督
在 done 步需要真终态，显式列是诚实编码，`sample_windows` 免 chaining 派生、
async 传输免稀疏侧带。带宽代价（state 双宽）记入 P4/P5 的打包优化域。

### T-C episode/step id 由 replay 内部派生

`episode_id`/`step_id` **不进 schema、不过传输**：replay 从
`terminated | truncated` 逐 lane 派生（done 即 episode 边界，D4）。理由：

- GPU-IPC fused ring 的 float32 融合对宽计数有损（其模块契约明言宽计数应走
  host ring），int64 ids 过环会被精度截断；
- sync `step()` 与 async ring 槽两路消费者都能看到 dones，派生无信息损失；
- 前一尝试把 ids 放 collector 是 recorder 架构的遗留义务，此处收敛。

### T-D 模板形状

```python
def wm_transition_schema(
    state_fields: Iterable[TransitionField],   # modeling state：read 集 + 外生通道，名字=字段名
    act_dim: int,
    *,
    obs_dim: int | None = None,                # 可选：actor 帧栈历史用 policy obs
) -> TransitionSchema
```

字段序：`*state` → `*next_state` → `actions (act_dim,) f32` →
`rewards () f32` → `terminated () i64` → `truncated () i64` →（`obs`）。dones
双列分开（D5：lambda-return 的 bootstrap 语义区分终止/截断），episode 边界取
或。provider 特有 extra 字段（如 goal/noisy obs）由 provider 在返回值上拼接
`TransitionSchema(fields=tpl.fields + extras)`——模板不开任意 extras 参数。

### T-E 归一器升格（单实现，非新写）

`EmpiricalNormalization`（含 `_mean/_std/_var/count` 注册 buffer、可选 local
accumulators 跨 rank merge）**原类移动**至 `motrix_rl/normalize.py`，
`fastsac/buffer.py` 保留 re-export（仓库内 import 兼容）。APG 直接消费同一类，
D7（NORM_KEYS 属性面 → `WeightChannelShared` async 权重同步）天然满足，不在
`motrix_rl` 新写第二份统计数学。fastsac 全量回归护航等价移动。

### T-F metrics family 注册（新族）

`runner.py` `_METRICS_FAMILIES["apg"] = "rl.worldmodel"`。family 值是 live
dashboard 选曲线族的依据（外部服务）；APG 的曲线族特征（WM 多步损失、想象
BPTT、UTD 节奏）与 `rl.offpolicy` 差异大，**单设新族**而非复用——dashboard
侧需同步登记 `rl.worldmodel`（登记前的过渡期由 dashboard 的未知族处理决定，
不阻塞训练侧合入）。

## SequenceReplay 契约

```python
class SequenceReplay:
    def __init__(self, *, schema, num_envs, capacity, device, seed) -> None
    def extend(self, batch: Mapping[str, Tensor]) -> None   # 一步全批次，字段名=schema 名（含 next_*）
    def sample_windows(self, batch_size, length) -> dict     # WM 监督窗
    def sample_starts(self, batch_size) -> dict              # 想象起点
    def history(self, envs, positions, field, length) -> Tensor  # 连续帧栈（含钳位/缺帧语义）
    def rng_state(self) -> Tensor; def load_rng_state(state)
    def __len__(self) -> int
```

- 存储：每 schema 字段一块 `(num_envs, capacity, *per_env_shape)` device 张量
  + 内部 `episode_id/step_id` int64 同形；
- `extend` 校验字段名集与形状（廉价 assert 级），CUDA 走 pinned staging +
  event 纪律（前一尝试已验证的复用安全）；
- `sample_windows` 返回：各 state 字段 `states/next_states (B,L,*shape)`、
  `actions/rewards/terminated/truncated (B,L)`、窗首 `previous_action`、
  `(B,L)` `valid` 掩码（chained 的 cummin 语义，首 done 包含、reset 尾与
  ring-wrap 掩除）；
- `sample_starts` 返回：各 state 字段当前槽行、`previous_action`（断链置零）、
  `intact` 掩码；
- 空池/窗不足显式 `ValueError`（沿用前一尝试文案语义）。

## 文件改动

| 文件 | 动作 |
|---|---|
| `motrix_rl/src/motrix_rl/replay.py` | 新增：`SequenceReplay` |
| `motrix_rl/src/motrix_rl/normalize.py` | 新增：`EmpiricalNormalization`（自 fastsac/buffer.py 移入） |
| `motrix_rl/src/motrix_rl/transport/schema.py` | 修改：`wm_transition_schema` 模板 |
| `motrix_rl/src/motrix_rl/runner.py` | 修改：`_METRICS_FAMILIES` 注册 `apg` |
| `motrix_rl_builtin/.../fastsac/buffer.py` | 修改：类移出、re-export 保留 |
| `motrix_rl/tests/test_replay.py` | 新增：模板 + 回放契约测试 |
| `motrix_rl/tests/test_normalize.py` | 新增：升格后 NORM_KEYS 面与状态字典往返 |

单 commit（roadmap P3.1 即一项），无 env 侧、无 provider 改动。

## 验收

- 模板：字段序/形状/dtype 契约，provider 拼接 extras 可行；
- 回放：extend 逐 leaf 拷贝（后续步/reset 不改已存样本）；episode/step id
  派生在 terminated 与 truncated 双路正确；窗口 valid 掩码边界包含式（首 done
  有效、其后 mask）、ring-wrap 掩除；`next_*` 列在 done 行为真终态（装配器
  契约，fixture 直写验证）；起点 `previous_action`/`intact`/帧栈钳位与缺帧
  语义；rng 往返确定性；空池/窗不足报错；
- 归一器：`NORM_KEYS` hasattr 探测路径可发布（weight-channel 契约）、
  state_dict 往返、fastsac 侧 import 兼容 + 全量回归一致；
- `runner`：`family == "rl.worldmodel"`。

## 与后续阶段的边界

- **P3.2**：转换装配器（FinalState → `next_*` done 行；modeling view 字段 →
  `state_fields`）与 provider extra 字段拼接是 provider 侧消费方，本设计只
  定 schema 与回放；
- **P4**：async ring drain 的块式入栈（`add_batch` 同型）在同一 `replay.py`
  内追加；GPU fused 环的带宽优化属传输域；
- **P5**：packed 多 clip 的窗口语义（多 clip 同 lane 交错）届时扩展，本设计
  的 id 派生与 chaining 不预设单 clip 假设。

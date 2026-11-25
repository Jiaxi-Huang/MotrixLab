# APG Trainer 框架层优化点评估（基线视角）

## 摘要

本文区分 APG 的结构适配与后续算法研究。当前上游基线为 `main@6dc21c4`，
APG rebase 基点为 `5a9908d`；`4a42758` 是历史 APG 分支，不是上游 main。
命名观测布局与 history 收敛采用 [APG 设计](../design/wm-bptt-trainer.md#观测布局与-main-契约)；
传感器重建、critic 校准和任务指纹仍为独立后续需求。
Torch 2.7 对默认 Muon/compiled imagination 的支持缺口仍待独立解决，
结构适配完成不代表默认安装环境的合入条件已满足。

## 定位与方法

本文是 instinct 分支（`codex/instinct-g1-apg`）`wiki/research/apg-generality-instinct.md`
（下称**证据文档**，尚未合入 main，合入后本处补全为正式相对链接）的去任务化提炼：
以**历史 APG trainer**（`4a42758`）为证据对象，把 instinct/G1 任务支持
（11 轮训练台账）暴露的问题泛化为框架本身的优化点，按 RL 侧
（motrix_rl / motrix_rl_builtin）与 env 框架侧（motrix_env_core / 任务模板层）
组织。任务专属细节（相机参数、片段长度、轮次编号等）只留在证据文档，本文
正文只保留对"下一个任务会再撞上"的判断；文末附录的 F→P/Q 映射表是证据
条目在 main 内的可用摘要。

每条按：**基线现状 → 泛化教训 → 优化方向**，不含实施方案——按工作流，实现
走设计文档。未明确标注当前实现的 file:line 指历史 `4a42758` blob 行号。任务侧现象、
代价与台账见证据文档对应 F 条目。

## RL 侧（motrix_rl / motrix_rl_builtin/apg）

### P1 命名观测布局与派生状态布局分开维护

**当前实现**：main 的 `ObservationSpec` 描述 group/frame/history，Manager layout
提供 term 名称和切片。APG 的不可变布局对象据此解析 `policy.proprio`、
`policy.goal`，并贯通 learner、collector、normalization 与 play。term 换序不改变
网络输入语义；缺失、额外 policy term 和非单帧组在构建边界报错。
history 使用 main 的可微窗口函数，仍只堆叠 proprio。
checkpoint 和 ONNX 保持 canonical history+goal 输入，尺寸由同一布局派生。

**剩余边界**：GenCoord aux/rep 仍是算法内部的固定物理表示，其维度与
encode/decode 由同一实现维护。增加新传感器或状态通道需要具体任务驱动的设计，
不能从命名观测适配直接推导出通用 latent/provider 框架的需求。

**证据**：历史证据文档 F1+F3；当前输入契约见
[APG 设计](../design/wm-bptt-trainer.md#观测布局与-main-契约)。

### P2 结局归因遥测缺口：回合怎么死的在传输层被丢弃

**基线现状**：trainer 把回合结束坍缩为标量 `terminated | truncated`
（train.py:176），EpisodeWindow 只记 mean return/length（train.py:146/178/
215-216）。批次级的 per-term 触发率遥测**已存在**（process_metrics →
Collection.metrics → `metrics/<name>`，train.py:231/274），真正缺的是：
(a) time-limit 截断是不进任何 per-term mask 的独立折叠（env.py:833-842），
"死于封顶"在任何遥测里都不可归因；(b) FinalState 只带
env_ids/sim/commands/actions（array/env.py:44-61），transition schema 的
done 字段只有标量 terminated/truncated（schema.py:108-115）——**回合级**
"哪个原因结束了哪个回合"在传输层丢失；(c) 无周期性 policy 评估 hook。

**泛化教训**：训练曲线无法区分"能力不足"与"回合预算/数据分布问题"，
长回合任务的调试会长期依赖会话外手写脚本；trainer 自身指标平台化误导
方向。任何长 horizon / 带成功边界的任务都会撞。

**优化方向**：done-reason（含 time-out 归因）作为 env→trainer 的标准遥测
通道，随 FinalState/transition schema 携带回合级归因；checkpoint 间隔触发
的 policy 评估 hook，指标由任务声明。

**证据**：证据文档 F7。

### P3 续训/日程无 extend 语义：计数器入档了，锚点与总量没有

**基线现状**：LR 余弦衰减的时钟 `world_model_iterations` **已入 checkpoint
并正确恢复**（learner.py:434/439-440），续训时进度无缝衔接——这一半是对的。
缺口在另一半：衰减锚点 `learning_rate_decay_steps` 与训练总量
`num_learning_iterations` 是纯启动配置，progress 在 1.0 处钳制（下限
0.03×base，learner.py:136-146）；延长一个已衰减的 run 必须同时手改两个全局
数，且余弦形状被静默重整形。全仓无 extend 语义；亦无 warmup/阶段调度概念
（policy 从第 0 轮起就对随机初始化的世界模型做 BPTT）。

**泛化教训**：任何需要"再练一段"的 run（长任务、多阶段交付）都要付手工
算术与尾段性能回落的双重代价，且最终交付物常不是 run 末尾。

**优化方向**：日程锚点/总量入 checkpoint 或提供连贯的 `resume --extend`
入口；warmup→train 的阶段调度一等化。

**证据**：证据文档 F5。

### P4 优化器参数所有权语义分叉：既成事实，不是新问题

**基线现状**：`make_optimizer(model, networks, ...)` 的参数语义随 optimizer
字符串分叉——Adam 路径收编 `model.parameters()` 全体，Muon 路径只取
networks 隐层 ∪ 余集（optim.py:14-27/:53）；critic 优化器在基线就绕过工厂
（learner.py:98-99 裸 `torch.optim.Adam`）。梯度裁剪范围按 module 容器传
（learner.py:374 传 `actor.parameters()`），与优化器实际持有集是两套口径。

**泛化教训**：这不是"加了视觉才触发"的分叉，而是基线既成的三处不一致
（工厂内分叉 / critic 旁路 / 裁剪口径）。任何调整网络结构归属的任务
（子模块独立 LR、冻结部分网络）都会撞。

**优化方向**：子模块粒度的所有权显式声明；裁剪集从所有权派生；全部优化
器统一走工厂。

**证据**：证据文档 F2。

### P5 Critic 只在想象分布上拟合：无真实数据校准通道

**基线现状**：critic 的输入与 TD 目标全部来自想象 rollout 链
（learner.py:252-282）；replay 的真实后继转移从不进 critic 训练——replay
只通过 `sample_windows` 为世界模型提供起始真实状态（learner.py:34-40 一带）。

**泛化教训**：这是 Dyna 式"critic 只见想象分布"的结构性缺口，世界模型
不完美之处皆存在（原 motion-tracking 任务模型误差小所以没显形）。价值
高估在标准遥测中不可见，发现依赖手工对比真实/想象 TD。

**优化方向**：真实数据 critic 校准成为一等机制——正式配置面、文档与
real TD bias 类标准遥测指标，而非外挂补丁。

**证据**：证据文档 F4（前半）。

### P6 特权特征声明通道只覆盖 env 既有 value obs

**基线现状**：特权观测的声明通道**已有**——`value_observation_space`
（array/env.py:240，resolver.py:711-717 校验 shape/dtype）。缺口是：trainer
侧凭空合成的特权特征（世界模型依赖的物理场扫描一类）没有声明路径，只能
在算法内部按维度算术拼进 critic 输入；配套数据以文件路径形式写进 algo 层
配置，属层级违规（只能靠校验和兜底一致性）。

**泛化教训**：仅当任务需要 trainer 合成特权特征时撞——频率低于 P1/P2，
但一旦撞上就是层级混乱加配置面污染。

**优化方向**：任务声明特权特征 provider（value obs 的扩展点），critic
输入维度从声明派生；algo 配置与任务数据解耦。

**证据**：证据文档 F4（后半）。

### P7 Checkpoint 的任务定义指纹仍是独立需求

**当前实现**：checkpoint 已携带 proprio/goal/action 尺寸；Policy 的 `low/high`
是注册 buffers，随 policy state 保存，`restore_policy` 恢复这些动作量程。
resume/play 核对环境与 checkpoint 的输入输出尺寸。因此“动作量程未入档”
不构成当前缺陷。

**剩余边界**：同维度下 actuator 顺序、增益及任务定义变更的语义兼容性尚无
任务指纹保护。不能据此断言所有历史 checkpoint 都不可回放；是否需要指纹及其
覆盖范围，应由具体兼容性需求单独定义。

**证据**：证据文档 F6；当前 Policy 与 restore_policy 的 state-dict 契约。

## env 框架侧（motrix_env_core / 任务模板层）

### Q1 世界模型可建模性边界：只有"拒绝"，没有"由谁重建"

**基线现状**：`derive_modeling_view` 对非 modelable query 类型直接 raise
（modeling.py:692-704）；非 modelable 集合 = 接触力 / SensorValues /
RaycastCameraImage / GeomPairColliding / ActuatorCtrl（read.py 声明处）。
ModelingView 的表面是 fields/exogenous/program/action_process
（modeling.py:58-73），没有 per-field 的重建接口——非 modelable 字段的
"重建责任"没有抽象，只能由算法侧做布局手术（instinct 分支的
`modeled_queries` 旁路即为此打的补丁：调用方自行以原单位/形状重建）。

**泛化教训**：每种新传感器（深度、激光雷达、高程、RGB）进入世界模型都
要在算法内部重做一次 encode/decode 布局手术；P1 facet B 的魔法索引正是
这个手术的痕迹。

**优化方向**：per-query-type 的重建 provider 接口（状态向量槽位分配 /
encode / decode）+ 声明式 latent 槽位布局。与 P1 同属一个设计簇（布局
描述符），但修复 locus 在 env 框架侧；`modeled_queries`"能力由调用方
声明"的设计本身干净，值得作为 provider 的入口保留。

**证据**：证据文档 F3。

### Q2 WBT 训练/评估回合语义结构性不一致，且无契约校验

**基线现状**：`max_episode_seconds = 10.0` 是 WBT 基础默认（cfg.py:215）；
`for_play()` 才设置无上限 + uniform + 帧零全链（cfg.py:244-248，另含
sequential_clips/hold_at_clip_end），但它只连到回放路径（train.py:59-68，
训练构建 `play=False`）；全仓无 `for_training()` 对应物。于是训练环境的
回合语义与评估环境结构性不一致：评估可达的成功边界（clip_end），训练
可能结构性不可达，且无任何告警。另外无"回合预算 ≥ 成功边界帧数"的契约
校验，起点采样器把 reset 质量泼在参考-场景穿透帧上也无告警。

**泛化教训**：静默默认值陷阱的表现是"一切正常只是学不会"——训练曲线、
reward 分项、系统指标全部正常。任何长片段 / 带明确成功边界的任务都会撞，
且定位成本极高（本次靠对照上游才定位）。

**优化方向**：env 契约校验层（回合预算 vs 成功边界；采样起点处的穿透
告警）；训练路径配方一等化（`for_training()` 或经校验的默认值），使训练
与评估的回合语义可对齐；起点课程升为 trainer 可见的一等概念。

**证据**：证据文档 F8。

## 已验证足够通用的部分（正面清单，防误改）

基线已具备且经受过任务检验：

- **FinalState 终态捕获**：pre-reset 终态随传输层定案进入 replay
  （bridge.py:144-145、train.py:67），终值不污染。
- **EMA/raw 双变体 checkpoint + `restore_policy`**（learner.py:69-71/
  416-428）：跨长程训练与续训无格式事故。
- **异步采集架构**：env 数、buffer、horizon 扩容均为纯配置。
- **配置覆盖链**（task yaml > algo_base）与 RNG 全量入档。
- **per-term 批次触发率遥测**（P2 所述通道，注意其粒度局限即可）。

instinct 分支改动中验证、值得沉淀为通用模式：

- 新能力键一律默认关闭的扩展模式（非新任务零影响）。
- 按 query 类型的存储分层（如浮点精度分级）。
- 数据-场景配对的完整性校验（哈希兜底）。
- `modeled_queries`"能力由调用方声明"的旁路设计（Q1 的入口）。

## 优先级建议

按"下一个任务会立刻再撞上"的频率排序：

| 优先级 | 条目 | 理由 | 建议产物 |
| --- | --- | --- | --- |
| 1 | Q1 与 P1 剩余边界 | 命名观测适配已覆盖当前 motion 输入；新传感器重建须由具体任务驱动 | 按任务需求另行设计 |
| 2 | P2 | 每个新任务的调试成本直接取决于此；回合预算类陷阱的免疫手段 | 设计文档：遥测通道 + 评估 hook |
| 3 | Q2 | 实现成本低、拦截收益高（校验即在 env 层） | 可并入 P2 文档或独立小改 |
| 4 | P3 ≈ P4 | 前者影响交付质量与续训效率，后者是既成分叉、改网络结构必撞 | 各一份小设计文档 |
| 5 | P5 | 机制已在 instinct 分支验证可用，主要是正式化 | 小文档 |
| 6 | P6 | 低频，层级整理为主 | 并入 P1 文档 |
| 7 | P7 | 同维度任务语义变更需要明确兼容性要求 | 独立设计 |

## 附录：与任务侧证据文档的映射

证据文档 = instinct 分支（`codex/instinct-g1-apg`）`wiki/research/apg-generality-instinct.md`
的 F1–F8 条目（现象/根因/代价详见该文档）：

| 证据文档条目 | 本文条目 |
| --- | --- |
| F1 视觉通路一次性特例 | P1（facet A 为主） |
| F2 优化器所有权 | P4 |
| F3 可建模性边界 + 布局手术 | P1（facet B）+ Q1 |
| F4 critic 校准 + 特权特征 | P5 + P6 |
| F5 续训/日程 | P3 |
| F6 checkpoint 指纹 | P7 |
| F7 结局归因遥测 | P2 |
| F8 WBT 静默默认值 | Q2 |

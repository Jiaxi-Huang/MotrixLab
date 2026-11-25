# APG Trainer 设计开发总结（相对 main 基线）

日期：2026-11-19 ｜ 状态：快照文档（随开发推进会过期）

## 范围与口径

- **基线 → 当前**：`1ddbb5d`（APG 设计文档落地前一 commit）→ `4a42758`（HEAD），另含
  工作区未提交的任务/服务器 YAML 修订（本机 recipe 调整、server preset 增加
  `ring_capacity: 4`）。合计 85 个文件，+9931 / −700 行。
- **职责边界**：本文是跨层总结快照。设计权威以
  [APG Trainer 设计](./wm-bptt-trainer.md)、[Manager 双路径建模](./manager/modeling.md)、
  [RL 序列数据平面](./rl-sequence-data-plane.md) 为准；全部验收证据、原始日志与数据
  位置以[实施计划](../plan/apg-supermimic.md)为准。本文不引入新结论，数字均出自上述
  文档与当前代码。
- **参考实现**：SuperMimic `abc124c`（任务语义：观测/奖励/网络 I/O）、
  XMimic `54860e1`（加速策略参考）。
  有意差异（速度增量 + 解析积分而非全状态增量、十帧间隔五帧 goal 等）记录在设计文档。
- **硬件口径**：本机 = i5-14600KF + RTX 5060 Ti 16 GiB（缩小网络）；服务器 = 北京 5090
  双路 Xeon 8380，单卡训练（全量网络）。两端 CPU 仿真/调度能力不同，SPS 不可直接互比。

## 总览

![APG 总体架构与异步执行拓扑](./apg-architecture.svg)

四层结构：任务数学全部落在 `motrix_envs`（真实与想象单源）；环境框架
`motrix_env_core` 提供派生建模、终态捕获与可微 FK；`motrix_rl` 沉淀 FastSAC/APG
共享的传输、序列回放、归一化与 CPU 配额契约；`motrix_rl_builtin.apg` 以
`motrix.apg` provider 接入，trainer 不按任务名分支。异步模式将环境与推理 actor 放入
spawn collector 进程，主进程独占 learner / replay / checkpoint，三通道互联：
共享内存 transition ring（上行数据）、单向 Pipe（上行元数据）、seqlock 权重通道（下行）。

---

## 1. RL 侧：设计、优化与训练效果

### 1.1 算法本体（APG = world model + 冻结 WM 的想象 BPTT）

| 部件 | 设计 |
| --- | --- |
| 状态表示 | GenCoord：`rep = [z, gravity_b, v_b, w_b, q, qd]`，`aux = [p, quat_xyzw]`；由声明 read 集派生（`bridge.py: GenCoordLayout`） |
| 世界模型 | MLP（LayerNorm/SiLU）输出速度增量 `[Δv_b, Δw_b, Δqd]` + 解析积分（姿态按 decimation 系数推进，sinc 用 Taylor 分支保持可微）；输出增量按真实转换统计归一化；监督为 8 步窗口六类状态加权 MSE |
| actor | goal encoder（十帧 `[q,qd]` + 十帧 root-relative 6D 旋转）→ tanh latent → decoder 拼接 10 帧 proprio history；末层 0.01 小初始化 |
| critic | 特权输入（rep + value obs），输入与回报目标分别归一化；一次批量 forward 合并 H 步 bootstrap |
| 终止语义 | 想象侧消费任务声明的 sigmoid soft survival；真实侧同阈值硬终止；clip 尾为 truncation，在真终态 bootstrap（`FinalState` 后继），auto-reset 值不可能泄入 lambda-return 目标 |
| 优化器 | actor/WM 隐层 Muon（betas 分别 0.9/0.95 与 0.9/0.999）、首尾投影/bias Adam、critic fused Adam |
| 调度 | UTD = 0.4（`Fraction` 累计配额，每 2–3 collector step 一轮）；每轮 WM×10 + policy×1 + critic×20；cosine LR 锚 250000 learner rounds 衰减至 3%；EMA 0.995 仅推理 artifact；warmup = max(learning_starts, history 10, WM horizon 8) |
| 回放 | `SequenceReplay`：episode 边界正确的 device 驻留环形库，`next_*` 显式列（done 行为装配器写入的真终态），加权抽样全 device（float64 cumsum + searchsorted 不过主机） |
| 交付物 | policy checkpoint（含推理网络/normalizer/维度指纹）+ training checkpoint（WM/critic/优化器/RNG/课程状态/UTD 余数）；ONNX flat 输入 parity < 1.5e-6 |

规模配方：本机 task（16 GiB）= 2048 envs、actor [2048,1024,512]、WM [1024]×3、
batch 1024；服务器 experiment（`+experiment=g1-wbt-supermimic/apg-server`）= 4096
envs、actor [2048,1024,512,256]、WM [2048]×3、policy batch 4096 / WM batch 2048、
replay 1000。物理、奖励、horizon（policy 32 / WM 8 / history 10）、UTD 两端一致。

### 1.2 执行设计：同步参考路径 → 异步单 collector

同步与异步共用同一份采样实现、装配器、learner 与配额计算（`collector.py` 中
`Collector` 两种模式复用，异步仅把 env+actor 移入 spawn 进程）。异步要点：

- **有界背压**：ring 容量限制采样超前，消费按原始 step 顺序推进，不跳过终态、不合并配额；
- **UTD 时钟**：`UpdateSchedule` 以 Fraction 累计，resume 后余数恢复、replay 重新填充
  期间不消耗更新额度；
- **权重发布**：actor 参数与两个 normalizer 同一版本原子发布（fused 连续向量），
  `weight_version_lag` / `ring_fill` 单列遥测；publish 计入 learn 阶段计时；
- **课程状态**：checkpoint 的课程/RNG 在 collector 对应 step 截取（主进程只有 1 lane
  建模环境），collector 动作噪声 RNG 随课程快照保存恢复；
- **退出阶梯**：join → terminate → kill，异常路径不遗留 GPU 进程；元数据通道从
  Queue 改为单向 Pipe，消除 semaphore 清理告警。

### 1.3 性能优化主线（每项均独立验收，保持网络/UTD/数学不变）

| # | 优化（提交） | 机制 | 实测收益 |
| --- | --- | --- | --- |
| 1 | Manager program / WM loss 编译（预验收期） | 固定数值 program 编译 + 整段 WM loss fullgraph + 消除动作环形缓存 Python 读取 | batch 1024 Manager 前反向 10.8→2.6 ms；batch 128 WM 34.5→6.4 ms（梯度 L2 3.5e-6） |
| 2 | FK 小算子融合（`909344b`） | 共享 `math/kinematics.py` 3×3 矩阵乘展开为可融合逐元素运算（不建第二套 FK） | 本机 +13.1%（10829→12242）；服务器全量 +2.9%（9288→9555） |
| 3 | WM/critic 更新编译 + Manager 显式工作状态 + 公共 CPU 放置（`956996a`/`8be31c3`/`7b10c05`） | backward 构成编译边界；imagination 单步 fullgraph；command/action/RNG 工作状态显式传递；`numa.py` 共享配额 | 本机 +22.3%（12487→15300）；服务器全量 **+18.5%**（8818.6→10447.7，三组配对均提升） |
| 4 | learner/collector CPU 配额分离 | 同 NUMA 内 learner 1–4 核 + collector 8–16 核（`async_options.*_cpu_cores`） | 相对共享配额 +9.3–17.2%（13064 SPS @4+16）；更多核无收益（31 核反而更慢） |
| 5 | **整轨迹 actor CUDA Graph 捕获**（b58afd2 起） | 复用单步 Inductor 前后向内核，独立 stream warmup×3 后 capture 整条 32 步 rollout+backward；输入固定地址 staging；cuBLAS 工作区归属本图池 | 服务器全量 **+74.0%**（13195→22961 SPS）；policy CPU span 737→184 ms；kernel launch 23591→4949（−79%）；GPU 中位 35%→63% |
| 6 | WM 按需投影采样 | `sample_windows(state_fields, initial_state_only, include_metadata=False)` 只 gather WM 实际消费字段 | WM sample 39.99→21.96 ms（−45.1%） |
| 7 | 状态编码融合 + 归一化 device region | encode 与五组统计更新合并编译 | learn 123.3→117.0 ms（单次） |
| 8 | 重复 H2D 消除（借用契约） | `SequenceReplay.extend` 返回只读借用 device batch，归一化复用已上传数据 | 25647 SPS、GPU 64.7%（单次） |
| 9 | ring 容量 2→4 | 匹配 0.4 UTD 的生产/消费节奏 | **+18.8%**（25667→30487 SPS），collect 16.9 ms |
| 10 | borrow 租约消除 host clone | ring 槽视图直通 replay staging（H2D 源仍是 replay 私有 pinned staging） | collect 15.9→8.73 ms，31941 SPS |
| 11 | 梯度检查 CUDA 顺序断言 + lambda-return 编译 | 消除每次更新的 host 同步；递推合并编译 | 微基准 1.09→0.053 ms；5000-step 正式训练 11m33s（前版 13m46s） |

被拒绝的候选（证据保留，不进默认路径）：Muon 直接编译（BF16 舍入差异 + 无稳定收益）、
完整 32 步 / 8 步块 actor AOT 编译（全量首编译 17 分钟超时退出 124）、31 核 collector、
独立 optimizer CUDA graph（2048 方阵仅 3.84→3.68 ms）。

### 1.4 5090 服务器吞吐演进（全量配置 4096 envs，同卡对照）

| 阶段 | 稳态 env-steps/s | 备注 |
| --- | ---: | --- |
| 同步参考路径（预验收） | ~4100 | host 调度为主瓶颈，非 GPU 算力 |
| 异步单 collector（512-step） | 4779 | 同步 4090 → +16.8%（此行为 2048 envs 缩小配置同卡对照） |
| FK 融合（`909344b`） | 9555 | +2.9%；learn −10.8% |
| 完整编译边界组合 | 10448 | +18.5%（三组配对 15.0–18.5%） |
| CPU 配额分离 4+16 | 13064 | 相对共享 +17.2% |
| 整轨迹 CUDA Graph | 22961 | +74.0%；learn 258.7→124.6 ms |
| 投影采样 + 编码融合 + H2D 复用 + 单线程 learner | ~25300 | 单次 512-step |
| ring=4 + borrow + 顺序断言 | 30487–31941 | 短跑峰值；正式 5000-step 稳态 27–28k |
| 正式 5000-step（第二轮优化后，三组） | 27–28k | GPU 平均 65.6% → 78.3% → 81.3%（300 连续样本）；三组均 34/40 |

瓶颈定性（多次 profiler + 栈采样）：主瓶颈是**串行 host kernel launch 与 autograd
调度**（数万次微秒级小 kernel，中位 1.25 μs、89.6% < 10 μs），不是 GPU 算力或整机
CPU 核数—— learner 实耗不足 1 核、collector 约 13.4 核（16 线程配额）、50.4%
collector 时间在 native `SceneModel.step_n`。 learner 加速后等待移向 collector 供给，
GPU 平均利用率 41%（historyfix 配方稳态）说明 collector/host 已是当前上限；90% 利用率
目标未达成（最高 81.27%），继续加核/加 ring 无证据支持。

### 1.5 训练效果（LAFAN1 全 40 clip，四进程、首帧→末帧、单次 rollout、无重试）

演进（均为固定协议完整评估，非中间挑优）：

| 配方 | 步数/预算 | 成功数 |
| --- | ---: | ---: |
| 本机 WM10 长训（100000 collector steps） | 最佳点 57344 | 36/40 |
| 服务器 WM10（主动停止） | 最佳 24576/32768 | 33/40（第五轮后停止，最新 31/40） |
| Muon 新训（本机 16384） | 最终 | 27/40 |
| 实现验收（本机 8192） | 最终 | 35/40 |
| 整轨迹 capture（服务器 8192） | 最终 | 37/40 |
| 优化后 5000-step ×3（streamed/devicecheck/borrowed） | 最终 | 34/40 三次 |
| XMimic 参数子集（WM×5、WM 1024×3、batch 2048） | 2974/3974/5000 rounds | 34/34/34 |
| history 修复 A（mix 0.5/0.4/0.1 + WM×5） | 5000 rounds | 39/40 |
| history 修复 B/D（mix 0.8/0.1/0.1） | 5000 rounds | 36/38 |
| **history 修复 C（mix 0.5/0.4/0.1 + WM×10）seed1** | 5000 rounds | **40/40** |
| C seed2（复验） | 4474/5000 | 39/39 |

**当前最佳**：C seed1 在 5000 learner updates 首次观测到 40/40，启动至 checkpoint
23m12s（1392.1s，北京 5090 单卡、xmcli 网络配方：WM 1024×3、policy/WM batch 2048、
4096 envs）。首次 ≥39 在 3974 rounds（18m51s）。C seed2 连续两个 39/40 但失败动作
不同——**40/40 是有限搜索的单点结果，不是稳定结论**；未实现连续 40/40 或独立 seed
复现。失败集中在 fallAndGetUp / fight 类片段（起身接触阶段），是后续研究优先级。

关键质量修复：motion 起点采样器的 history 项更新缺口（xmcli 三个 checkpoint 的
history 全部等于初始化均匀值 → 修复为每次 rebuild `0.999·history + 0.001·soft`
归一化混合；A 相对 xmcli 同预算 34→39 支持保留该修复，但无多 seed 因果量化）。

### 1.6 learner 更新流水与编译边界

![Learner 单轮更新流水与编译边界](./apg-learner-compile.svg)

---

## 2. env 侧：框架设计与任务支持

### 2.1 双路径建模：任务数学单源

![env 侧双路径建模与终态捕获](./apg-env-modeling.svg)

核心机制：resolver 产出的 `ResolvedManagerProgram` 是纯数据，**同一份 call records
可以绑定第二个 shadow `ManagerContext` 再编译一次**。真实路径绑定常驻 context
（sim 视图 = read program arena、常驻 term 实例、随机 rand）；想象路径绑定 modeling
state 张量束支撑的 shadow context（确定性 rand、kernel_data 按结构判定拷贝或只读
共享）。任务侧零 python、零感知：

- **派生即能力**（D3）：`ManagerEnv.modeling_view()` 构造期 dry-run，失败即显式报错
  （term 名 + 字段名）；派生失败的任务配 APG trainer 在配置期报错，YAML 中无 python 路径；
- **语义类型来自 query 类**：`SimDataQuery` 子类声明 `modelable`，关节/旋转/速度可想象，
  接触力/相机/raycast 在类型系统层面拒绝——新增传感器时"可否想象"是框架级义务；
- **group 级 all-or-nothing**：reward/obs/termination 组整体可派生或整体不可想象，
  避免加权和残缺；
- **外生通道声明式注记**：`increment`（motion phase，可声明尾帧 upper_bound 保持）/
  `copy_action` / `constant`，想象侧推进与真实 command `advance` 数值等价（fixture 锁定）；
- **shadow 隔离**：想象侧执行后真实侧常驻状态逐位不变（fixture 强制）；clip 结束在
  想象侧映射为 truncation 而非重新采样。

### 2.2 配套框架机制

- **FinalState**（P2.2）：唯一必须发生在 env 内的捕获——auto-reset 会销毁 done 行后继。
  捕获点在 evaluate 之后、reset 之前（单次 gather），覆盖 terminated + truncated 全体
  done 行，内容为声明 read 字段行 + 常驻 command/action term state 行（恰为外生通道
  终值）。`step()` 返回契约不变，model-free 消费者零感知、零成本。
- **Manager truncation 一等化**（P2.1）：时间性截断与失败终止分离，服务 lambda-return
  的 bootstrap 语义；删除任务侧 `_update_truncate` 覆写。
- **可微 FK 上游化**（#163）：`KinematicTree` device-mobile；修复 MotrixSim 原生
  post-step link 查询缓存陈旧（修复前 body position 误差 ~0.025 m、角速度 ~9.2 rad/s；
  修复后与相同状态的可微 FK 各项 < 1e-5），刷新不积分、不改变 q/qd。
- **课程采样器族**（P1）：`MotionStartSampler` 统一 uniform / adaptive / return，多
  clip 统计归 sampler 所有；return 模式经 `ctx.reward` 累积 episode 回报；history
  混合项见 §1.5 的修复。课程状态随 command 的 `training_state_dict` 序列化，trainer
  不感知内部结构。

### 2.3 不同类型任务的接入方式

任务 = WBT 家族内一个 cfg 文件（`g1/supermimic.py` 与 backflip/dance 并列）+ 必要的
term 数学文件；差异经配置继承表达，数学差异新增 term（真实/想象共用）。trainer 通过
派生能力而非任务名识别可训练性。当前 `GenCoordLayout` 的结构性约束（声明集不满足即
构造期报错）：

- 恰好一个声明的 `KinematicTreeQuery`（单 floating base 链）；
- root 的四类查询（position/quaternion/linear/angular velocity）各一个；
- joint position/velocity 查询各一个且顺序一致，与 actuator 目标顺序一致；
- 其余字段仅接受 batch body 查询（经可微 FK 重建）；
- 单个无延迟 joint-position action term。

支持新任务类型的路径：语义上等价的任务（不同机器人/不同 clip 集/不同奖励权重）纯
配置即可；需要新传感器/新观测项的任务会撞上已知边界——非 modelable query 直接拒绝、
跨层布局靠位置魔法数、trainer 合成特权特征无声明通道。这些已按"下一个任务会再撞上"
的频率整理为 [APG 框架层优化点评估](../research/apg-framework-optimization-points.md)
（RL 侧 P1–P7、env 侧 Q1–Q2，P1+Q1 布局描述符簇优先级最高），instinct/G1 任务
（跪登台阶）已在该分支完成 11 轮训练台账验证。

---

## 3. RL ↔ env 数据传输与 H2D

![数据传输与 H2D 优化全景](./apg-data-plane.svg)

### 3.1 三条通道

| 通道 | 方向 | 载体 | 内容与语义 |
| --- | --- | --- | --- |
| ① transition ring | collector → learner | `HostTransitionRing`（host 共享内存，SPSC，capacity 4） | 每 slot 一个 env-step 全批次；schema 声明字段集（state/next_*/actions/rewards/dones/obs/collector_observation）；有界背压；read_span 零拷贝视图 + commit 释放；单写者 int64 游标，x86/TSO 免屏障 |
| ② 元数据 Pipe | collector → learner | 单向 `mp.Pipe`（pickle bytes） | reward terms / process metrics / 课程快照 / RNG / 分阶段计时 / 异常 traceback；内联序列化使元数据独立于子进程张量 FD 生命周期 |
| ③ 权重通道 | learner → collector | `WeightChannelShared`（seqlock 双缓冲 host 槽） | fused flat params（actor 全参数一维连续）+ fused normalizer stat 行；odd=写入中/even=已发布；发布即已落地（D2H 先 sync）；CUDA-IPC 变体已在 transport 实现但 APG 走 host 变体 |

### 3.2 H2D 设计原则

1. **pinned staging + event 纪律**：`SequenceReplay.extend` 每字段一块 pinned staging，
   `non_blocking` H2D 后 record event，下一次 extend 先 synchronize——staging 复用安全；
2. **借用契约替代重复上传**：`extend` 返回只读 device batch（CUDA 上为 staging 别名），
   同一 stream 立即消费（归一化等），下次 extend 即可覆盖；调用方需要保留先 clone；
   replay 存储始终独立复制，生命周期契约记录在序列数据平面设计；
3. **连续融合下行**：发布侧 flatten 成单一向量经 pinned staging D2H；接收侧
   `bind_flat_params` 把 collector 模块参数绑定进一块连续 CUDA 存储，加载 = 每模块
   一次连续 H2D（替代逐参数阻塞传输）；
4. **标量窗口化**：训练指标 device 累计、每个日志窗口一次 `stack→tolist` D2H；
   episode 统计在 CPU ring 视图上累计，无逐轮设备读；
5. **租约零拷贝**：`borrow()` 持有 ring 槽视图直到 replay 入库完成才 commit——消除
   整批 host clone，且 H2D 源是 replay 私有 staging，不异步读已释放 ring。

### 3.3 已优化的传输（对应上图绿色）

- replay 入库 pinned staging + event（前一尝试已验证的纪律回收）；
- 归一化复用借用 batch：state/next_state/actions/obs(proprio) 不再从 CPU 各搬一次
  （修复前归一化逐字段 `.to(device)` 重复上传）；
- collector 权重加载一次连续 H2D + normalizer stat 行一次 H2D（实测加载 0.7–1.5 ms/步，
  publish 0.9–2.9 ms/步）；
- publish 侧 pinned D2H non_blocking + stream sync 后再翻 seq；
- borrow 租约消除整批 host clone（collect 15.9→8.73 ms）；
- 采样完全 device 驻留：加权抽样、窗口 gather/mask（编译融合）、history 帧栈均无 H2D；
- 标量/日志窗口化 D2H。

### 3.4 尚未优化的传输（对应上图橙色虚线）

1. **goal 归一化输入**（`learner.py update_normalization`）：异步模式下
   `collected.observation`（CPU ring 视图）的 goal 切片每步一次 `.to(device)`——
   同批数据其实已随 `collector_observation` 字段经 replay staging 上了 GPU，未复用。
   这是最直接的剩余项（goal 维度大于 proprio：十帧 `[q,qd]` + 十帧 6D 旋转）；
2. **collector 推理小张量**：obs/done `.to(device)` 未 pinned、同步拷贝；actions
   `.cpu()` D2H 每步一次（栈采样实测仅约 0.1% collector 时间，量级小）；
3. **上行链路两次 CPU 中转**：env 输出（CPU）→ ring（共享内存 memcpy）→ staging
   （CPU 拷贝）→ H2D。`GpuIpcTransitionRing`（CUDA-IPC 设备槽）在 transport 层已
   实现且与 host ring 共享游标协议，但 APG 未启用——真实环境数据在 CPU 产生，IPC
   环只能省掉 ring→staging 一跳，收益有限且未做验收；
4. `assembler.snapshot()` 每步全声明字段 CPU clone（内存带宽开销，非 PCIe）；
   checkpoint `torch.save` 全量 D2H（非热路径，仅 checkpoint 间歇）。

### 3.5 量化参考（全量配置，512-step 稳定窗口）

collect（learner 侧等待+接收+入库+归一化）8.7–16.9 ms/step、learn 107–126 ms/step、
publish 约 2 ms/step、collector 权重加载约 1.4 ms/step；ring_fill 均值 0.4–0.5、
weight_version_lag 均值 1.2。learner 加速后瓶颈移向 collector 供给——WM 1024×3
较小配方（historyfix 阶段）稳态 GPU 平均仅约 41%，继续扩大传输容量无证据支持。

---

## 4. 当前状态与后续

**已确立**：同步参考路径与异步单 collector 路径均可用且契约一致（同装配器/learner/
配额）；整轨迹 CUDA Graph + 按需采样 + H2D 复用 + ring=4 的组合使服务器全量稳态从
~4100 提升到 27–31k env-steps/s（同卡配对最高 +74% 单项）；LAFAN1 单 seed 单点 40/40
（23m12s）；history 采样修复有效（34→39）。

**未确立 / 保留**：连续 40/40 与多 seed 稳定性（seed2 为 39/40）；优化实现的全量
time-to-quality 配对（历史 37/40 与 5000-step 34/40 预算不同不可比）；90% GPU 利用率
目标（最高 81.27%，当前上限在 collector/host 供给而非 GPU）；同步/异步从零长训的
成功率与达标时间对照。这些均为 plan 中未勾选项，不因吞吐数字视为完成。

**下一步优先级**（沿用 plan 与优化点评估）：fallAndGetUp 失败阶段归因、采样集中/滞后
与固定验证分布上的 WM 误差（研究向）；goal 归一化 H2D 复用（工程向，最小改动）；
P1+Q1 跨层布局描述符设计簇（新任务通用性，需先出设计文档）。

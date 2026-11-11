# 三方实现对照：supermimic（上游）/ supermimic_vibed / MotrixLab APG

## 摘要

本文是旧两方对比文档的**整体重写**（不保留旧文，直接覆盖），把对照对象从两方扩为三方：

- **上游 supermimic**（`~/code/supermimic`）：HybridRobotics 研究代码，构建在 mjlab（MuJoCo/MJX warp、IsaacLab 风格 manager 体系）之上，G1 29DoF 全身运动跟踪的 world-model + 想象 BPTT 训练；
- **supermimic_vibed**（`~/code/supermimic_vibed`）：独立复现（IsaacLab 3 + Newton/MuJoCo-Warp 后端），README 自述对论文（arXiv 2610.28480）有若干记录在案的偏离；
- **本仓库**（MotrixLab）：MotrixSim 之上的 `g1-wbt-supermimic` 任务 + `motrix.apg` provider。

对照只看**语义**（env 与 trainer 的数学/数值/时序），不看实现形式。三方源码级核查结论（2026-11-11，含对关键差异点的直接源码复核）：

1. **核心任务语义三方一致**：奖励逐项（权重与 σ）、终止阈值、reset 噪声数值、return 起点课程配方、93 维本体观测与噪声半宽、动作缩放公式、50Hz/4 子步节拍。
2. **本仓库与上游逐项对齐**：goal 窗构成（actor 640 / critic 侧 +rel_pos 30）、帧时序（reward 对转移前帧计、goal 含当前帧 offset 0）、软终止公式、GAE critic 形式与 v_max 裁剪范围、WM gencoord 配方（rep-state 特征、加速度 delta、c=5/8 半隐式积分、Bryson 名义误差）。
3. **vibed 忠实复现但带 6 项自述/实证偏离**（§3）：全局 1 帧相位超前（reward 对 t+1 计、goal 从 t+1 起）、actor goal 额外携带参考根位移/速度（730 vs 上游 640）、软终止形状不同（双 sigmoid 之积 vs max 复合）、encoder offset 每 episode 重采样（上游 startup 常量）、bf16、WM-only 预热 500 迭代 + cosine lr 衰减。
4. **误继承已纠正（本文落地的第五轮对齐）**：上条最后一项（预热 + cosine 衰减）与"探索噪声按 action_limit 缩放"此前被本仓库当作上游配方继承（实为 vibed 自创），本轮全部改回上游语义——删除 `warmup_iterations` 选项、固定 lr（去 cosine）、探索噪声改回 raw 命令空间 σ=0.1（详见 §3、§6）。obs history 10 帧经确认与上游实际训练配置一致，保留。
5. **剩余不对齐项**收窄为：域随机化与 sim_ctx 条件化（结构性缺口）、家族关节保护终止、dynamics 零初始化输出层、Muon/fp32 工程差、num_envs 2048、语料占位等（§4）。

对照的主要源码入口：

- 上游：`supermimic/representations.py`（goal 窗/维数）、`supermimic/tasks/{task_spec,env_cfgs}.py`、`supermimic/utils/tracking_reward.py`、`supermimic/envs/world_model_env.py`、`supermimic/learners/{policy_learner,wm_learner}.py`、`supermimic/runners/{supermimic_runner,ppo_runner,reset_sampling}.py`、`supermimic/sim_context.py`；基础环境为 mjlab 的 `unitree_g1_flat_tracking_env_cfg`（`~/code/mjlab`）。
- vibed：`supermimic/{env,mdp,motion,robot,networks,train,buffer,reset_sampling}.py`；偏离自述见其 `README.md` "Deviations" 节。
- 本仓库：[g1/supermimic.py](../../motrix_envs/src/motrix_envs/locomotion/wbt/g1/supermimic.py)、[wbt/mdp/](../../motrix_envs/src/motrix_envs/locomotion/wbt/mdp/)（term 实现）、[apg/learner.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/learner.py)、[apg/networks.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/networks.py)、[apg/config.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/config.py)、[apg/collector.py](../../motrix_rl_builtin/src/motrix_rl_builtin/apg/collector.py)、[motrix.apg.yaml](../../configs/task/g1-wbt-supermimic/motrix.apg.yaml)、[manager/modeling.py](../../motrix_env_core/src/motrix_env_core/manager/modeling.py)、[manager/exogenous.py](../../motrix_env_core/src/motrix_env_core/manager/exogenous.py)；设计文档见 [wm-bptt-trainer](../design/wm-bptt-trainer.md) 与 [apg-provider](../design/apg-provider.md)。

## 0. 三方代码库一览

| | 上游 supermimic | vibed | MotrixLab（本仓库） |
|---|---|---|---|
| 定位 | 研究原始实现 | 论文独立复现 | 上游主路线的 provider 化移植（WM+BPTT，明确不移植 PPO 副路线） |
| 仿真后端 | mjlab（MuJoCo/MJX warp），manager-based | IsaacLab 3 kit-less + Newton（MuJoCo-Warp solver，implicitfast） | MotrixSim，自有 env_core manager 体系 |
| 机器人/节拍 | G1 29DoF；sim 0.005s × 4 → ctrl 0.02s | 同（substeps 4） | 同（`SimCfg(dt=0.005)`，decimation 4） |
| 语料 | mjlab `MotionLoader` npz（gencoord + maxcoord 双表示，LAFAN1） | csv 30fps→50fps 重采样（slerp/中心差分），LAFAN1 40 clips | NPZ v1（一文件一 clip、fps=50 校验、全局帧轴 + `frame_clip_end`）；当前 dance 单 clip 占位 |
| episode | 500 步上限；clip 尾 clamp 续跑（WM 侧）；`motion_end` 截断 | 500 步 / clip 尾，done-not-term（bootstrap） | 10s 上限 + `motion_clip_end` 显式截断（truncated 语义 bootstrap） |

## 1. 任务（env）语义对照

### 1.1 动作与 PD

三方完全同构：29 维位置目标 = `默认姿态 + raw · scale_j`，`scale_j = 0.25 · effort_limit / kp`（逐关节数值一致：hip_pitch/yaw 与 waist_yaw 0.5475、hip_roll/knee 0.3507、ankle/waist_roll/pitch/shoulder 0.4386、wrist 0.0745）；PD 增益由 natural frequency 10Hz、damping 2.0 从 armature 推出（kp 40.18/99.10/28.50/14.25/16.78 一致）；默认姿态（KNEES_BENT）一致。上游与本仓库动作无裁剪（控制范围钳位是唯一边界）；vibed 探索后 clamp ±20（多余但无实质影响）。

上游有 encoder_bias 通道：obs 为 `(q + bias) − default`、执行目标为 `proc − bias`（编码器坐标架一致）；vibed 以 `xi = −bias` 等价实现；本仓库无 DR 故无此通道（见 §1.6）。

### 1.2 观测与 goal 窗

**本体感知（policy 组，93 维）**：三方同集合同噪声——`[projected_gravity(3, 噪半宽0.05), base_ang_vel(3, 0.2), dof_pos_rel(29, 0.01), dof_vel(29, 0.5), actions(29, 0)]`（项序不同，对 MLP 无语义影响）。归一化在 trainer 侧（RunningNorm/EmpiricalNormalization），env 不裁剪不缩放。obs history 栈：上游 `history_length` 可配、实际训练 10 帧；vibed 10；本仓库 10（一致）。

**goal 窗（关键差异区，已源码复核）**：

| | 上游 | vibed | 本仓库 |
|---|---|---|---|
| 窗口索引 | `{t, t+5, …, t+45}`（含当前帧，`representations.py:186-189`，clamp 当前 clip 末帧） | `{t+1, …, t+46}`（`motion.py:108`，**整体超前 1 帧 = 20ms**） | `{0,5,…,45}`（`mdp/observations.py`，clamp-and-hold） |
| actor goal | **640** = 每帧 `[qj(29), vj(29)]` + rel_ori rot6d(6×10)（`representations.py:169`） | **730**：额外加参考根位移 dp（参考 heading 系）+ 参考根速度 v/w，且 vj×0.1（`motion.py:130`） | **640**（`goal_terms=[motion_future_joint, motion_future_ref_ori_b]`）✅ 同上游 |
| critic 侧 goal | 670（+rel_pos 3×10，机器人根系） | 760（+rel_pos） | 670 等价构成嵌入 890 维 value group ✅ |
| 超界 slot | clamp 重复末帧（速度保留真值） | 末姿态 + 速度清零（`recorded` 掩码语义） | clamp 重复末帧 ✅ 同上游 |
| rel_ori 定义 | 参考根姿态相对机器人当前根系，rot6d | 同 | 同 ✅ |

**value 组（critic 特权输入）**：上游 = rep_state(68) + wm_goal(670) + **sim_ctx(33)**；vibed = INV(71，height+6D tilt) + extras 107（关节误差、tracked body 误差、根误差、xi 33）+ wm_goal 760；本仓库 = 890 维 value group（goal 670 + soft_termination(1) + 14 body 位置(42)/姿态 6D(84) + base 线/角速度 + dof_pos/vel + actions），无归一化裸输入（同上游 critic）。已知 caveat：`robot_body_ori_b`（84 维）仍是家族历史 bug 兼容实现（body 0 的 6D 矩阵元素被后续 body 当参考四元数用），不是干净的上游等价物。

多任务：上游 actor 为 motion + velocity 双任务多编码器（slot_mix 批内分槽 + latent BC）；vibed 与本仓库均单任务（本仓库单 encoder——多 encoder / latent BC / slot_mix 随第二任务落地再加，第八轮删去未使用的 `goal_encoder_splits`）。

### 1.3 奖励

三方逐项一致（本仓库 `supermimic.py:79-84` 覆盖家族默认）：

| 项 | 权重 | σ |
|---|---|---|
| 全局锚点（pelvis）位置 | 0.5 | 0.3 |
| 全局锚点姿态 | 0.5 | 0.4 |
| 身体相对位置（14 body 均值） | 1.0 | 0.3 |
| 身体相对姿态 | 1.0 | 0.4 |
| 身体线速度（世界系） | 1.0 | 1.0 |
| 身体角速度（世界系） | 1.0 | 3.14 |
| action_rate（raw 命令差分） | −0.1 | — |
| 关节软限位（0.9 soft） | −2.0 | — |
| 自碰/接触 | 均无（上游删；本仓库接触力 query 不可建模连带删） | — |

- 高斯核 `w·exp(−err²/σ²)`，每控制步 ×ctrl_dt(0.02)；正权重和 **5.0 → 每步上限 0.1、v_max = g_max = 0.1/(1−0.99) = 10**（更正旧文 11 的残留值）。
- re-anchor 语义一致：相对目标的 xy+yaw 取机器人实际根、z+roll/pitch 取参考根（上游 `update_relative_body_poses`、vibed `dR = yaw(R·rRᵀ)`、本仓库命令 update hook 的 `align_targets`）。
- 帧时序：上游与本仓库 reward 对**转移前帧 t** 计（物理 → reward → 帧推进）；vibed 对 **t+1** 计（与 goal 窗同一 1 帧相位超前）。
- 架构差异（不回退）：上游 WM 环境里镜像重实现一份 tracking reward；本仓库双路径编译让真实 env 与想象侧消费**同一份 term program**（任务数学单源），一致性由 fixture 锁定。

### 1.4 终止、截断与软终止

- 硬终止：`|z_ref − z_pelvis| > 0.25` 与倾角差 `> 0.8` 三方逐字一致。本仓库另保留家族关节保护 `bad_dof_pos`(0.5)/`bad_dof_vel`(100)（有意保留的防爆护，上游/vibed 无）；上游另删 ee_body 终止（本仓库同样删）。
- 截断：clip 尾/时间上限均为 truncation 语义（bootstrap）；上游 WM 侧 clip 尾 clamp 续跑，本仓库想象侧以 `steps` 外生通道 INCREMENT + `clamp="clip_end"` 等价实现。
- 软终止（想象侧生存 gate 的来源）：上游与本仓库逐字相同 `p = σ((max(z_err/0.25, tilt_err/0.8) − 1)/0.05)`；vibed 为双 sigmoid（半宽 0.02/0.04）之积——形状不同。本仓库软终止由任务以 value 组 obs term 单源声明，learner 按名列取通道，gate = 1 − p。

### 1.5 reset 与起点课程

- reset 噪声数值三方逐字一致：root pos ±(0.05,0.05,0.01)、Euler ±(0.1,0.1,0.2)、线速度 ±(0.5,0.5,0.2)、角速度 ±(0.52,0.52,0.78)、关节位置 ±0.1 后钳位、关节速度不加噪。
- 起点采样默认 return 课程：`0.5·uniform + 0.4·softmax(−25·V/g_max) + 0.1·history`，1s bins、γ=0.99、timeout 用均值回报 bootstrap、terminated 不 bootstrap——三方一致（统计细节小差异：上游 history 项带 memory cap 256，vibed EMA 0.999/0.001，本仓库 history_mix 0.1）。
- 想象起点均从 replay 采样（RSI 式）。历史帧处理有实现细节差异：上游存干净传感器、使用时加新噪声（padding 槽共享单次噪声）；本仓库存采集期已带噪 obs、向 episode 头钳位重复——分布近似等价。

### 1.6 域随机化与 sim_ctx（三方差异最大处）

| | 上游 | vibed | 本仓库 |
|---|---|---|---|
| 摩擦 | foot friction U[0.3,1.2]，startup | U[0.3,1.2]（机器人 shape 材质，地面固定 0.3），startup | **无** |
| torso COM | ±(0.025,0.05,0.05)，startup | 同，startup | **无** |
| encoder/关节偏置 | ±0.01，**startup 常量** | ±0.01，**每 episode 重采样**（自述偏离） | **无** |
| sim_ctx 条件化 | 33 维（friction+COM+bias，归一化）输入 dynamics 与 critic——WM 学参数化动力学 | xi 33 维同角色 | 无（dynamics 输入少 33 维，学标称动力学） |

本仓库若引入 DR，需扩展外生通道注记体系（见 [manager/modeling](../design/manager/modeling.md)）——**剩余最大结构性差异**。

### 1.7 运动参考数据

三方均纯当前帧整数 gather、不插值；未来窗在查询侧 clamp 当前 clip 末帧。表示：上游 npz 存 gencoord + maxcoord 双表示（root 世界四元数含朝向，heading 不变性由 reward 侧 re-anchor 达成）；vibed 存 gencoord（csv 重采样 + 可微 FK 预计算 14 body 状态）；本仓库 NPZ v1 存 body 世界位姿 + 关节 q/v。本仓库语料为 dance 单 clip（999 帧）占位，supermimic 目标语料（LAFAN1）尚未入 assets。

## 2. Trainer 语义对照

### 2.1 算法族谱

- 上游两条路线：① `SuperMimicRunner` 主路线（真实采集 → replay → WM 监督 → 想象 BPTT 策略学习；默认 actor-critic 变体 = bootstrap + GAE critic 拟合；另有 γ=1 无 bootstrap 的纯 BPTT 变体）；② `PpoRunner` 副路线（想象 rollout 上 rsl_rl PPO，自标注 test 脚本）。
- vibed：仅主路线（actor-critic 形态）。
- 本仓库：仅主路线 ① 的 actor-critic 变体完整移植；无 PPO 组件（无 ratio/clip/KL/entropy）。

### 2.2 训练循环与 UTD

三方稳态配方一致：短 on-policy ring（1000 步/lane，均匀窗采样、窗内无 done）→ **utd 0.4**（每 4096-lane 步 0.4 次策略更新）→ 每次更新 10 个 WM 窗 batch（2048×8 步）+ 1 次 actor 更新（想象 batch 4096）+ critic 回归（5 epochs × 4 minibatches）。粒度不同：上游每 1 步采集按 Ratio 0.4 触发；vibed/本仓库每迭代 2.5 步（2/3 交替）触发——稳态等价。

- WM-only 预热：上游**无**；vibed 500 迭代（自述偏离）；本仓库**本轮删除**（选项与 YAML 项一并移除）。
- replay 预填充（64 步，为凑 8 步监督窗）是数据结构必需，非训练语义预热，保留。

### 2.3 策略损失、critic、WM、actor

| 项 | 上游 | vibed | 本仓库 |
|---|---|---|---|
| actor 损失 | `G = Σ γ^t(∏gate)r_t + γ^H(∏gate)V̂(s_H)`，V̂ clamp v_max=10；冻结 WM/critic 反传 | 同 | 同（`learner.py`）✅ |
| critic | GAE(γ0.99, λ0.95) soft-dones，5×4 MSE；clamp 仅 bootstrap 与回归目标 | TD(λ) 等价形式；**逐步 V 也 clamp**；norm 更新期继续更新 | GAE δ 形式（=上游），clamp 仅 bootstrap+targets ✅ |
| WM 配方 | MLP 2048³；输入 rep_state `[z,g_b,v_b,w_b,qj,vj]`(68)+action(29)+**sim_ctx(33)**；输出加速度 delta `[Δv_b,Δw_b,Δvj]`(35)；c=5/8 半隐式积分；Bryson 名义误差 0.01/0.02/0.1/3.14；8 步窗 | 同（输入 INV 71 + 29 + xi 33；6D tilt 代替 3D gravity） | 同上游特征（68+29=97，**无 sim_ctx**）✅ |
| WM 输出初始化 | 默认 | 零初始化（匀速延续护栏） | 零初始化（第七轮曾改默认、GPU 首跑实证发散后**恢复**——本仓库 action_limit 乘法约定下必要，见 §6-7） |
| actor 网络 | per-task encoder MLP([2048,1024,512,256])→tanh latent 32 + decoder 同隐藏；LayerNorm+GELU；确定性输出 | 同 | 同（`GoalConditionedActor`）✅ |
| 探索噪声 | σ=0.1 加在 **raw 动作**（逐关节 scale 前），不裁剪 | σ=0.1 raw + clamp ±20 | **本轮改为** σ=0.1 加在 raw 命令（per-joint scale 前），不裁剪 ✅（旧实现 0.1×action_limit，关节空间量级偏小） |
| 梯度裁剪 | norm 1.0 全部 + actor 逐元素 value clip 0.1；actor Adam(0.9,0.95)，WM/critic (0.9,0.999) | 同 | 同 ✅ |

### 2.4 优化器、lr、精度

| | 上游 | vibed | 本仓库 |
|---|---|---|---|
| lr | **固定 2e-4，无 schedule** | cosine → 3%（250k 迭代） | **固定 2e-4**（第五轮删 cosine；restore 后 `reset_learning_rates()` 回基准）✅ |
| 优化器 | 隐藏 Linear 权重 Muon（默认 momentum 0.95 / wd 0.1，`adjust_lr_fn="match_rms_adamw"`）+ 其余参数 fused capturable Adam，逐模块 betas | Adam（Muon 选项存在但默认关、未测） | **同上游**（第六轮 `_build_optimizers`：torch 2.11 内置 `torch.optim.Muon`，调用逐参一致；lr 恒定天然满足上游"捕获时冻结"语义）✅ |
| 精度 | fp32（全局 TF32 matmul；reward 重锚定/IK 的旋转数学用 `small_matmul` 元素乘绕开 TF32） | bf16 autocast（仅 MLP matmul） | **fp32**（第六轮 YAML `amp: null`；TF32 经 `float32_matmul_precision("high")` 与上游一致；小旋转矩阵防护见 §4-15）✅ |

**CUDA graph 更新捕获（未移植，纯性能优化、零语义）**：上游 `CapturedUpdate` 把整个更新（WM 批、策略更新）在 3 次 eager 预热后录成 CUDA graph 回放——全设备共享一个 stream/pool、按录制顺序回放、要求捕获函数无 host sync、形状固定、输入经静态 TensorDict 缓冲。本轮对齐后其**前置条件已全部就绪**：actor 无条件 step（无非有限梯度 host-sync 门控）、lr 恒定、优化器 capturable 化、归一器更新期冻结。剩余工作 = 把 `update_world_model`/`policy_update`/`_update_critic` 的**采样与计算拆分**（采样留宿主侧，计算读静态输入缓冲）+ 移植 `CapturedUpdate`（约百行，输入需支持嵌套 dict 而非 TensorDict）。

### 2.5 数据面与工程质量

本仓库增强（上游无对应契约）：FinalState 终态捕获（auto-reset 前真终态入 replay）、SequenceReplay valid/intact 掩码、terminated/truncated 分列 schema、学习状态精确恢复（已知偏差：sampler 课程统计不序列化）、ONNX 导出把 obs/goal 归一化 + action_limit 缩放折进图内（parity 校验）。上游为研究代码无 resume/export 契约；vibed 有原子 checkpoint + 完整 resume + ONNX 导出。

## 3. vibed 偏离清单与误继承纠正

vibed 相对上游的偏离（其 README "Deviations" 多有自述，本次源码逐一核实）：

1. **全局 1 帧相位超前**：goal 窗从 t+1 起（`motion.py:108`）、reward 对转移后帧 t+1 计（`env.py:204-205`）；上游/本仓库均含当前帧、对转移前帧计。等效 vibed 比 upstream 严 20ms。
2. **actor goal 组成**：730（多出参考根位移 dp + 根速度 v/w，vj×0.1）；上游/本仓库 640。
3. **软终止形状**：双 sigmoid(0.02/0.04) 之积；上游/本仓库 σ((max−1)/0.05)。
4. **DR 重采样节奏**：joint offset 每 episode 重采样；上游 startup 常量。
5. **bf16**（上游 fp32）。
6. **WM-only 预热 500 迭代 + cosine lr 衰减 → 3%**：vibed 自创（README 自述），上游固定 lr、无预热。

**误继承纠正（本轮第五次对齐落地）**：上表第 6 项与探索噪声缩放曾被旧版对比文档误标为"上游 SuperMimicConfig 配方"并被本仓库继承（`warmup_iterations`、`lr_decay_iterations`、噪声 ×action_limit）。本轮处置：

- `ApgCfg.trainer` 删除 `warmup_iterations` 与 `lr_decay_iterations`（provider 不再有此选项），训练循环去预热门控，lr 恒为配置值；
- `learner.set_learning_rate_scale` → `reset_learning_rates`（仅在 optimizer 状态恢复后调，防旧 checkpoint 残留衰减 lr）；`_lr_scale_at` 删除；
- 探索噪声 std 直接作用于 raw 命令（per-joint scale 前，关节空间量级 = std·scale_j，上游语义）；
- 任务 YAML 与 algo_base 基线同步删除相应键；
- obs history 10 帧保留（上游实际训练同为 10；runner 侧默认值 1 仅为库缺省）。

**第六轮处置（优化器/精度对齐）**：bf16（vibed 偏离 #5）与纯 Adam 同为 vibed 继承（上游为 Muon + capturable Adam + fp32）。本轮已对齐：优化器按上游 `build_optimizers` 拆分——每模块（dynamics/actor/critic）隐藏 Linear 权重走 `torch.optim.Muon`（默认超参 + `adjust_lr_fn="match_rms_adamw"`），其余参数走 fused capturable Adam（lr 设备张量）；任务 YAML `amp: bf16` → `amp: null`；同时移除 actor 的非有限梯度跳过门（vibed 继承，上游无条件 step——该检查每步强制 host sync，也是 CUDA graph 捕获的阻断项）。CUDA graph 本身未移植（前置已就绪，见 §2.4）。

**第七轮处置（讨论后分项裁决）**：① dynamics 零初始化输出层——确认为 vibed 继承（上游默认初始化），且等效一个隐式 WM-only 预热、与第五轮决策矛盾，**已删除**（`FloatingBaseDynamics` 回默认初始化；residual scheme 本就是默认初始化，两 scheme 就此一致；仅影响新建运行，旧 checkpoint 不受影响）；② 历史帧噪声拆三项裁决——(a) 存储 vs 使用时加噪：边缘分布等价、仅跨更新相关性小差，**保留为已记录近似**（对齐需存干净 obs，与噪声进 obs term 的双路径架构冲突，收益比差）；(b) **想象步观测无噪 → 加噪：真实语义缺口**（上游 `_synth_sensor(noisy=True)` 全程带噪，joint vel ±0.5 幅度下 actor 在干净观测上优化、部署在带噪观测上执行），**设计修订已写入 [wm-bptt-trainer](../design/wm-bptt-trainer.md) P2（shadow rand 真实采样），待确认后实施**；(c) episode 头 padding：两侧机制等价，不动；③ sampler 课程统计不序列化——**非 upstream 差异**（上游无 resume，checkpoint 仅权重），属 D8 未完成部分，影响为恢复后课程短暂回 uniform（2048 lanes 快速回填），**挂起**，触发条件 = 长训中断恢复成为现实或 Phase 4 async 落地。

## 4. 剩余不对齐清单（本仓库 vs 上游，截至本文）

| # | 项 | 性质 | 处置 |
|---|---|---|---|
| 1 | sim_ctx 域随机化条件化（friction/COM/encoder bias + 33 维上下文输入） | 结构性缺口 | 待 DR 基建（§1.6） |
| 2 | 家族关节级保护终止（bad_dof_pos/vel） | 有意保留 | 仿真防爆护，上游无 |
| 3 | 想象侧任务数学双路径单源（上游两处镜像实现） | 架构改进 | 不回退 |
| 4 | dynamics 零初始化输出层 | 实证必要偏离（第七轮两度裁决） | GPU 全宽默认初始化首跑即发散；零初始化保留，upstream 不需要的原因=其无 action_limit 乘法 |
| 5 | CUDA graph 更新捕获（上游 `CapturedUpdate`） | 性能优化（零语义） | Muon/fp32/恒定 lr/capturable 优化器/无条件 step 前置已就绪；剩余=采样与计算拆分 + CapturedUpdate 移植，待做 |
| 6 | num_envs 2048（YAML）vs 上游 4096 | 配置差 | 训练预算决定，按需调整 |
| 7 | maxcoord 表示（每刚体 + analytic IK） | 范围 | 待议 |
| 8 | velocity 多任务 / slot_mix / latent BC | 范围 | 随第二任务一并落地（多 encoder 机制已在第八轮裁撤，届时按需重建） |
| 9 | PPO 副路线 | 范围 | 明确不要 |
| 10 | 历史帧噪声：(a) 本仓库存带噪 obs、噪声实现冻结在 replay（上游存干净、使用时新抽） | 实现细节 | 边缘分布等价、仅跨更新相关性小差，保留为已记录近似（第七轮裁决） |
| 10b | 历史帧噪声：(b) 想象步观测无噪（上游 `_synth_sensor(noisy=True)` 全程带噪） | **语义缺口** | P2 修订已写入设计文档（shadow rand 真实采样），待确认后实施（第七轮） |
| 11 | resume 的 sampler 课程统计不序列化（D8 未完成；**非 upstream 差异**——上游无 resume） | 框架特性 | 挂起；触发 = 长训中断恢复成为现实或 Phase 4 async（第七轮裁决） |
| 12 | value 组 body_ori 84 维 bug 兼容 | 已知 caveat | 需显式迁移观测契约 |
| 13 | 语料为 dance 占位 | 资产 | 待入库 |
| 14 | critic 输入构成（上游 rep_state+goal+sim_ctx；本仓库 value group 含 body 位置/姿态特权块） | 实现细节 | 特权信息集合近似，语义等价性未逐项论证 |
| 15 | 小旋转矩阵 TF32 防护（上游 `small_matmul` 用于 reward 重锚定/IK/maxcoord；本仓库 env_core 数学未防护，TF32 "high" 下 3×3 级 matmul 有舍入） | 数值细节 | 候选后续（动共享 math 层，需单独评审） |

## 5. 对比实验注意事项

- episode 长度分布：本仓库真实 env 有 10s 上限截断 + clip 尾显式截断（truncated，bootstrap）；上游 clip 尾 clamp 续跑、episode 主要由终止决定——影响 episode 长度可比性（v_max/g_max 三方同为 10，不受影响）。
- 与 vibed 对比 return/成功率时注意其 1 帧相位超前（更严 20ms）与 actor goal 差异（730，信息更多），不构成公平基线。
- 双重 matrix6d bug（第三轮发现并修复）之前的 checkpoint，goal 姿态通道输入无效，不宜沿用。
- 第四轮改动后 checkpoint 格式变化（`goal_norm_{i}` 键、`goal_columns`、dynamics 结构）；**本轮第五轮**移除预热/lr 调度后，旧 checkpoint resume 时 optimizer lr 一律重置为基准值（上游语义），历史衰减中间态不再延续。
- 本轮起 `warmup_iterations`/`lr_decay_iterations` 从 schema 消失，引用这两个键的历史 YAML/脚本需同步清理。
- **第六轮**起 checkpoint 的 `optimizers` 变为按模块的 state_dict **列表**（Muon+Adam 拆分），旧单 state_dict 格式不兼容（load 时 strict zip 显式报错）。

## 6. 对齐落地记录（简史）

1. **第一轮**：任务侧数值对齐（锚点 pelvis、奖励权重 0.5/1.0/−0.1/−2.0、终止 0.25/0.8、return 课程启用、未来窗口 obs、projected_gravity）。
2. **第二轮**（外部评审后）：goal 平铺不入栈（policy 组仅本体感知 93 维）、goal 经 encoder MLP→tanh latent(32) 进 decoder、未来关节窗逐帧块布局、v_max 裁剪收窄为仅 bootstrap 与回归目标；修复 unroll 时序 bug（evaluate/observe 落在 s_{t+1}）。
3. **第三轮**：goal 窗 obs 契约测试，抓出并修复 `motion_future_reference_orientation_obs` 双重 matrix6d bug。
4. **第四轮**：想象侧对齐上游 WM 语义——软终止 gate、clip 尾 clamp 续跑（外生通道 clamp）、critic 改 GAE + 裸输入、actor 泛化多 goal encoder、`FloatingBaseDynamics` 重写为 gencoord 配方、探索噪声去裁剪。
5. **第五轮（2026-11-11 三方对照后）**：纠正对 vibed 的误继承——删除 WM-only 预热选项、固定 lr（去 cosine 衰减，restore 重置基准 lr）、探索噪声改 raw 命令空间 σ=0.1；确认 history 10 与上游实际训练一致后保留；v_max 表述修正为 10（Σw⁺=5.0，旧文 11 为历史权重残留）。一次性验证：trainer 端到端测试通过（3 passed）、导入/配置/YAML 校验通过。
6. **第六轮（优化器/精度对齐）**：优化器移植上游 `build_optimizers`——每模块隐藏 Linear 权重走 `torch.optim.Muon`（默认超参 + `adjust_lr_fn="match_rms_adamw"`，torch 2.11 内置、CPU/GPU 均可 step），其余参数走 fused capturable Adam（lr 设备张量，restore 后 `reset_learning_rates` 原地重置）；actor 移除非有限梯度跳过门（上游无条件 step，且该检查强制 host sync、阻断 CUDA graph）；任务 YAML `amp: bf16` → `amp: null`（上游全程 fp32，TF32 政策两侧本就一致）。CUDA graph 捕获未移植——前置条件已全部就绪，剩余采样/计算拆分 + `CapturedUpdate` 移植（§2.4）。验证：trainer 端到端测试 3 passed。
8. **第八轮（配置参数对齐审计）**：删 `actor.output_activation`（upstream 恒 unsquashed 确定性输出，网络层去掉激活选项）、`actor.updates_per_iteration`（upstream 每轮恒 1 次更新，Ratio 只调频率）、`trainer.compile`（惰性项）、`goal_encoder_splits`（单任务未用，遵循"不为假想未来任务预扩契约"）；`collect_steps_per_iteration` → `trainer.utd`（upstream 稳态语义 0.4 = 每 2.5 步一次更新，algo_base 基线 0.005 等价原 200 步/更新）；`goal_latent_dim` → `latent_dim`（upstream 命名；旧 checkpoint 的 dims 键有 fallback）。评估后**不加**：`latent_bc_weight`（多任务辅助损失，无第二 encoder 即死参数）、upstream 的 `replay_ratio` 公式（Ratio = num_envs/(wm_batches·batch·horizon)·replay_ratio 耦合其 4096-env 宽度，本机 512 下套用会得出 0.05 的错误 utd——只移植 utd 数值语义）、`slot_mix`（多任务）。保留：`world_model.scheme`（任务无关兜底，设计文档决策）、`soft_termination_term`（upstream 软终止的 env 侧单源等价物）、`amp`/eval 三项（repo 侧/D9 设计）、`replay.warmup_steps`（8 步窗的数据结构必需）。验证：CPU 10 passed（trainer e2e/provider schema/ONNX 导出）、ruff 全绿；GPU smoke 因正式训练占用显卡让位（utd 路径算术等价已由 CPU e2e 覆盖）。
7. **第七轮（三项差异讨论后裁决 + GPU 首跑实证修订）**：初裁"删 dynamics 零初始化"被 GPU 全宽首跑**推翻**——默认初始化 + 无 warmup 下首个 unroll 发散（imag_reward −13/步、梯度 NaN、NaN 权重进而 panic 仿真后端），零初始化恢复并改注为实证偏离（upstream 无 action_limit 乘法约定是其不需要护栏的关键差异）；同时恢复 actor decoder 输出层 ×0.01 近零初始化（同因：action_limit ≈5.5 放大默认初始化输出至多弧度级目标，action_rate 惩罚爆到 −8/步）；修复 `rotation_distance` 的 asin 反传 NaN（范数恰为 1.0f 时 `asin'(1)=∞`→全图 NaN，钳至 1−1e-6，前向偏差 <3e-3 rad）；影子宿主设备化（`modeling_view(device=)`：真实 env 的 term 程序在 CPU，影子宿主/kernel_data/静态 term 参数整体随 learner 设备迁移——CPU 测试与 GPU dry-run 均暴露不了的混载）。语料接线（LAFAN1 40 clips 软链入 assets/lafan，gitignore）与本机 16GB 配置（num_envs 512、WM [1024]³、actor [1024,512,256,128]、pi_batch 2048；显存主项=32 步 eager fp32 BPTT ≈ batch×horizon×75KB 与 replay ring）。验证：GPU 2 迭代 + final eval + checkpoint 全通（~1.7s/迭代），CPU 回归 36 passed。想象步加噪（P2 修订）仍未实施。

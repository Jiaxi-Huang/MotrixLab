# SONIC G1 任务迁移设计

## 目标与边界

本设计将 SONIC 的 G1 动作跟踪能力接入 MotrixLab 当前九包 workspace，同时保持现有
Manager、simulator registry、Hydra Task 和 FastSAC 公共契约。不引入旧仓库的兼容层，
也不改变 workspace package 版本。

迁移包含：

- Manager-based G1 环境与 canonical 10 future-frame 时序契约；
- SONIC motion 语料装载（schema v1 npz + `ext_smpl_*` 通道，见 [sonic-bones-seed-corpus](./sonic-bones-seed-corpus.md)）；
- 通过 `PolicyVariant` 注册的 SONIC FastSAC actor、命名辅助损失和同步/异步训练；
- MotrixLab checkpoint 保存、恢复与回放；
- 行为测试（合成语料 fixture）和双语用户文档。

## 环境与数据

环境注册名为 `g1-sonic`，没有 profile 或兄弟变体。配置使用 typed
Manager group，term factory 返回 `ObsTerm`、`RewardTerm`、`TerminationTerm` 或
`ResetTerm`；所有 fused-kernel 入口使用 `@dispatch`。

`SonicMotionClip` 在通用 `WbtMotionClip` 数组之外保存 SMPL reference 和逐帧 clip
边界。运行时只读取带 `ext_smpl_joints` / `ext_smpl_root_quat` 通道的 schema v1 npz 语料，
四元数统一为 `xyzw`，关节和 body 数组按 name-based 重排为 task contract。环境在未设置变量时
缺省指向 converter 的缓存输出 `~/.cache/motrixlab/bones_seed_npz/g1`——`download_bone_seed.py`
+ `convert_bones_seed.py` + 训练零额外参数即可串起；`SONIC_MOTION_DIR` 可覆盖（原 packed store
与内置 smoke 语料已移除，集成测试改用合成语料 fixture，见
[sonic-bones-seed-corpus](./sonic-bones-seed-corpus.md)）。
`configs/task/g1-sonic/motrix.fastsac.yaml` 是唯一 task recipe，直接内联
`algo.variant`。`model.num_future_frames`（smoke 配置为 5，release 值 10 见 yaml 注释）
同时驱动环境 observation 布局和 SONIC actor 输入；小规模验证只覆盖 CLI 的环境数、
播放环境数、checkpoint 间隔和迭代数。

足部 acceleration reward 所需的速度历史属于 SONIC command state。transition kernel 在
reward 求值前更新该状态，不扩展通用 `ActionTerm` 或 Manager 生命周期。

## FastSAC

`FastSacCfg.policy_variant` 通过中性 `policy_variant_registry` 选择专用 actor；
`FastSacCfg.variant` 是由 SONIC 变体自解析的映射，FastSAC 不理解其字段。普通
FastSAC 路径不变。SONIC observation 最后三个 multi-hot encoder mask 维度绕过经验归一化。
action 链路与 g1-wbt-dance 同构：环境声明按关节限位反推的紧致 action space
（限位残差 ÷ term scale），`SonicPolicyVariant.build_actor` 把 env 推导的
`action_scale`/`action_bias`（及 `use_tanh`）转发给 policy head，tanh ±1 恰好张满每个
关节的可达行程。action term 直接复用 wbt 的 `WbtJointPositionActionCfg`（2026-09-30
对齐）：物理 scale 由 g1_sonic.xml 模型增益按同一公式 `0.25·effort/kp` 推导，独立的
SONIC 增益表 `_SONIC_ACTUATOR_PARAMETERS` 与 reset 期 kp/kd 写入已删除，增益以模型
文件为单一来源（policy 关节序同日改为 `g1_sonic.xml` 模型 actuator 序，见下方
"关节序切换"小节）。
SONIC 专属的
辅助损失加权、指标命名和旧 checkpoint 校验由 variant hook 实现，通用 agent 只组合
variant 返回的额外 loss 与 metrics。
训练 checkpoint 写入 `policy_variant` 与 `policy_variant_metadata`，用于恢复时校验 variant
并记录模型配置。训练 task recipe 不包含
`g1_control_decoder_hidden_dims`：本地 `SonicActor` 不构造该 decoder，并用
FastSAC policy head 替代上游 PPO control head。

异步 collector/learner 继续沿用 FastSAC 的参数与 observation-normalizer 发布通道；
`PolicyVariant` 只负责构造两端一致的 actor，不改变 weight-channel 的 seqlock、slot handshake
或 host/CUDA-IPC 契约。

MotrixLab 训练 checkpoint 的回放继续依赖 run metadata，与其他任务一致。

## MotrixSim 兼容依据

当前 workspace 固定 `motrixsim==0.10.1` 正式版，迁移按 `v0.10.1` 发布的 API/MJCF
文档核对。控制写入必须保持 environment 轴和声明的 actuator 轴顺序；完整 actuator
写入会先转换为 native 顺序和 C-contiguous 布局。

## 发布约束

仓库不再捆绑任何 SONIC 动作数据（内置 smoke 语料已移除，测试改用合成 fixture）。
BONES-SEED / GEAR-SONIC 下载子集仅供内部训练；公开再分发相关数据前需补齐可审计的
授权引用、权利人和具体许可文本。

## 与上游 gear_sonic 的逐项对照（2026-09 复查）

对照基准：`GR00T-WholeBodyControl/gear_sonic` 的 `sonic_release.yaml` 发布实验。
以下记录每处偏离的定性（上游一致 / 有意移植 / 已修复 / 保留差异）。

### action 链路（本分支已改为对齐 g1-wbt-dance）

- 上游 policy 是无 tanh 的无界 Gaussian（std 独立参数），wrapper 端 clip ±20；
  到位控为 `default + (0.25·effort/kp)·action`，可达包络约 `5·effort/kp`。
  本 env 曾声明的 `Box(-20, 20)` 正是照抄上游 clip 值，不是策略输出空间。
- 现行实现（2026-09-30 起完全对齐 wbt）：env 复用 wbt 的 `WbtJointPositionAction`，
  `action_space` 用共享的 `joint_position_action_space_from_ctrl_ranges`
  从关节限位反推紧致空间，Sonic actor 转发 env 推导的 scale/bias。tanh ±1 =
  `default ± 0.25·effort/kp`；scale 与物理增益同源于 g1_sonic.xml（hip_pitch
  kp 99.1 / ±139 Nm），公式与 g1-wbt-dance 完全一致。曾独立维护的
  `_SONIC_ACTUATOR_PARAMETERS` 增益表与 reset 期 kp/kd 写入已删除——三者合一后
  "增益与 scale 不一致"类回归被结构性消除，模型文件发布值由
  `test_sonic_model_hip_pitch_release_gains` 继续钉住。
- **行为变化声明**：手腕可达范围由 ±0.596 rad（行程 37%）恢复到 ±1.61 rad（全行程），
  tanh log-prob 修正与 action-rate 惩罚的作用域随之变化；2026-09-23 验证跑的 scale
  结论不再直接适用，全量训练前需重新验证。

### 关节序切换：模型 actuator 序（2026-09-30）

原生训练不加载上游权重，关节内容序是自由变量。policy 关节序由上游交错的
`G1_ISAACLab_ORDER` 改为 `g1_sonic.xml` 的模型 actuator 序（左腿 6 / 右腿 6 /
腰 3 / 左臂 7 / 右臂 7，与 wbt 系环境及上游 MuJoCo 平面同思路）：

- `G1_SONIC_JOINTS` 按模型序声明，
  `test_sonic_declared_joint_order_matches_model_actuator_order` 钉住声明与模型同步；
  语料（MotionLibrary 按名重排）、dof 查询、action term 全部跟随该单一声明，
  链路内无需任何置换。
- 三组关节选择改为按名解析、不再硬编码下标：teleop 腿块 `HYBRID_LEG_JOINTS`
  （"左腿 6 在前、右腿 6 在后"的上游语义序保留，在模型序下恰为连续段）、
  SMPL 腕尾 `SONIC_WRIST_JOINTS`（改为按肢体分组 L(roll,pitch,yaw) /
  R(roll,pitch,yaw)）、足部差分 `SONIC_FOOT_JOINTS`。
- 布局等价性：encoder 输入的关节内容序不再与上游逐字节一致；帧交错 bug 兼容
  布局不受影响（`test_sonic_split_reproduces_upstream_tokenizer_flat_order`
  只锁定特征/帧维度，仍有效）。上游 release checkpoint 的 state_dict key/shape
  对应依旧成立，但语义互通需显式置换表（参考上游 `mujoco_to_isaaclab_dof` 的做法）。
  **此变更使旧本地 SONIC checkpoint 失效（形状兼容但关节语义错位），需重训。**

### G1 参考观测布局：上游 bug 兼容（勿"修复"）

上游 `command_multi_future` 先拼 feature-major `[全部 P | 全部 V]` 再 reshape 成
`(F, 2J)`，帧被交错打乱；上游在 `trl/losses/token_losses.py` 自注 "temporal axis is
incorrectly flattened" 且 decoder 依赖该布局。本仓库 env 的 feature-major term 布局 +
`SonicActor._split` 的 reshape 逐字节复刻上游 flat 顺序（G1 stride 5 ≙ 0.1 s，
SMPL stride 1 ≙ 0.02 s，与上游 `dt_future_ref_frames` 一致）；
`test_sonic_split_reproduces_upstream_tokenizer_flat_order` 锁定该等价。

### 模型层 infra 对齐（2026-09-29，A/B/C/D 全量收敛）

对照结论：核心图（encoder MLP 拓扑、FSQ universal token、g1_kin decoder、G1/SMPL 布局、
路由优先级）本已语义一致；本次将四类剩余语义差异全部对齐上游 release：

- **teleop/hybrid 输入流（C）**：重写 `SonicHybridReferenceObservation` 为上游 release
  flat 布局，总宽 `24F+27`——`[12 腿关节 pos × F（G1 stride 网格、含当前帧）| 同网格
  12 腿关节 vel × F | 当前帧 VR 3 点位置 9（reference-anchor 系）| 当前帧 VR 3 点四元数
  xyzw 12（anchor 系）| 当前帧 anchor 6D（ref 相对 robot anchor）]`。腿部关节改为上游
  语义序（左腿 6 关节在前、右腿在后；`HYBRID_LEG_POLICY_INDICES=(0,3,6,9,13,17,
  1,4,7,10,14,18)`）；本地 quat helper 约定即 xyzw，与上游 IsaacLab 一致。
  `test_sonic_hybrid_reference_matches_release_layout` 逐块锁定。
- **encoder_index 列序（D）**：改为上游 `encoder_sample_probs` 键序 `[g1, teleop,
  smpl]`；`ENCODER_SAMPLING_MODES=("mixed","g1","teleop","smpl")` 使 pinned mode 与
  mask 列一一对应。**此变更使旧本地 SONIC checkpoint 失效。**
- **multi-hot mask（B）**：reset 采样改为上游 legacy 行为——先按 native 分布抽 one-hot，
  smpl-native 行再置 g1=1，并以 `HYBRID_TELEOP_PROB_WHEN_SMPL=0.5` 置 teleop=1（形成
  三热行）；teleop-native 不激活 g1。模型侧删除 `g1_required=ones` hack（multi-hot 下
  g1 mask 已覆盖 smpl 行，aux 损失也只读各自 mask 的行）。
- **aux 损失（A）**：命名与权重对齐上游 `aux_loss_coef`——`g1_recon 0.01`、
  `g1_smpl_latent`/`g1_teleop_latent`/`teleop_smpl_latent`/`reencoded_smpl_g1_latent`
  各 1.0；新增 g1-teleop / teleop-smpl 两个对齐损失（三热行上计算，空行集为 0）。
  全部双向不 detach——经核实这恰是上游 release 实际接线（`G1SmplLatentLoss` 等读
  non-detached `encoded_latents`）；上游的 pre-detach 只存在于无任何 config 使用的
  compliance/CHIP 死码路径。

配套调查结论（排除分阶段训练假设）：multi-hot + 从零联合训练是上游 release 语义
（`optimize_encoders_ratio_for_CHIP` 全仓无 yaml 启用；`sonic_bones_seed` 的 4 encoder
亦从零联训；全仓 config 无 `pretrained_model`/freeze/`active_encoders`，finetune 路径
是 strict 全架构恢复）。

state_dict 兼容性：release 几何下 g1/smpl encoder 与 g1_kin decoder 的 key 可与上游
一一对应（需改前缀 `actor_module.`→`backbone.`、`decoders.g1_kin.`→`g1_kin_decoder.`）；
teleop encoder 输入宽度现为同函数 `24F+27`，理论上亦可对应（ quat 帧约定同为 xyzw）。

### 已核对一致项

- 失败终止四族（anchor z/ori、ee z、feet xyz）阈值 0.15/0.75/0.5 与上游一致；
  `bad_dof*`/`bad_body_z` 上游本就不存在（wbt 特有）。
- **clip_end 终止已移除，改为到尾换段（2026-09-25，对齐 wbt 语义）**：起始帧恢复
  全区间自适应采样；`advance()` 在跟踪完 clip 末帧后就地重采一个新帧并置
  `ctx.sim_reset_requested`，reset kernel 对该 lane 跑 `reset_env`（权威重采 +
  足速重播种 + encoder 重抽）加 sim reset teleport，episode 记账与 action 状态
  不变，回合只以 time_out 或失败收场。`clip_end`/`clip_ended`/`hold_at_clip_end`
  及 g1 env 的 terminated→truncated 改道一并删除。play 无时间上限，
  `start_at_timestep_zero_prob=1.0` 使每次换段回到第 0 帧，回放从头循环。
- low_reference 的 0.1 EMA 与终止放宽契约一致。
- reward 权重与 std 全部对齐 release（含 feet_acc -2.5e-6）。
- sonic 无 `simulate_action_latency` 上游对应物；本地曾有该选项（默认 False），
  2026-09-30 随 action term 复用 wbt 一并移除，行为仍与上游一致。

### 已修复的偏离

- **feet_acc 重置尖峰**：上游在 articulation 层用 reset 速度播种 previous velocity；
  本地曾清零后差分，产生 `(v-0)/dt` 假加速度。现 `reset_env` 用 teleport 目标帧的
  `clip.joint_vel` 播种。
- **NPZ 路径 fps 防护**：`SonicMotionClip.from_corpus`（MotionLibrary 校验）要求 50 Hz。上游对非 50 fps 数据做
  重采样而非报错，本仓库选择 fail-fast；换全量数据时请用 BONES-SEED / LAFAN converter
  重采样转换。

### 保留差异（有意决定，暂不跟进上游）

- **tracking_vr_3point_local 已对齐上游 sonic_release 三点集（2026-09-30）**：
  默认从上游 base 的 5 点集（骨盆+腕 0.18 offset+双踝）改为 release 覆盖值——
  torso 上方 0.5 m 虚拟点 + 双腕无 offset；reward term 字段随点集更名
  （`tracking_vr_5point_local` → `tracking_vr_3point_local`）。
- **观测不加噪声**：上游 actor obs 有 gravity±0.05 / ang_vel±0.2 / dof_pos±0.01 /
  dof_vel±0.5，参考观测 ±0.05；本地全无。保留现状。
- **adaptive sampling 无尾部 bin mask**：上游同样没有（wbt 的 mask 是 wbt 特有）；
  本地 EMA+kernel 平滑与上游 failure-rate 比例式亦不同，两者均为合法实现。
- **joint_limit 惩罚**：本地加 soft_limit 0.9 与 cap 5.0（上游为裸 L1 超限）；
  **anti_shake** 用 torso_link 替代上游 head_link（g1_sonic 无 head）。

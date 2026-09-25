# SONIC G1 任务迁移设计

## 目标与边界

本设计将 SONIC 的 G1 动作跟踪能力接入 MotrixLab 当前九包 workspace，同时保持现有
Manager、simulator registry、Hydra Task 和 FastSAC 公共契约。不引入旧仓库的兼容层，
也不改变 workspace package 版本。

迁移包含：

- Manager-based G1 环境与 canonical 10 future-frame 时序契约；
- 版本化、只读 mmap 的 SONIC packed motion store；
- 通过 `PolicyVariant` 注册的 SONIC FastSAC actor、命名辅助损失和同步/异步训练；
- MotrixLab checkpoint 保存、恢复与回放；
- 小型 smoke store、行为测试和双语用户文档。

## 环境与数据

环境注册名为 `g1-sonic`，没有 profile 或兄弟变体。配置使用 typed
Manager group，term factory 返回 `ObsTerm`、`RewardTerm`、`TerminationTerm` 或
`ResetTerm`；所有 fused-kernel 入口使用 `@dispatch`。

`SonicMotionClip` 在通用 `WbtMotionClip` 数组之外保存 SMPL reference 和逐帧 clip
边界。运行时只读取 `motrixlab_sonic_packed_v1`，四元数统一为 `xyzw`，关节和 body
数组按 task contract 排列。环境在未设置变量时回退到仓库内的小型 store，以满足
registry 和集成测试契约；有效训练所需的完整动作集由 `SONIC_PACKED_STORE` 外部提供。
`configs/task/g1-sonic/motrix.fastsac.yaml` 是唯一 task recipe，直接内联
`algo.variant`。`model.num_future_frames=10` 同时驱动环境 observation 布局和
SONIC actor 输入；小规模验证只覆盖 CLI 的环境数、播放环境数、checkpoint 间隔和迭代数。

足部 acceleration reward 所需的速度历史属于 SONIC command state。transition kernel 在
reward 求值前更新该状态，不扩展通用 `ActionTerm` 或 Manager 生命周期。

## FastSAC

`FastSacCfg.policy_variant` 通过中性 `policy_variant_registry` 选择专用 actor；
`FastSacCfg.variant` 是由 SONIC 变体自解析的映射，FastSAC 不理解其字段。普通
FastSAC 路径不变。SONIC observation 最后两个 encoder selector 维度绕过经验归一化。
action 链路与 g1-wbt-dance 同构：环境声明按关节限位反推的紧致 action space
（限位残差 ÷ term scale），`SonicPolicyVariant.build_actor` 把 env 推导的
`action_scale`/`action_bias`（及 `use_tanh`）转发给 policy head，tanh ±1 恰好张满每个
关节的可达行程；物理 scale 仍由 SONIC 增益表的 `2.0·effort/kp` 决定（不能改用 XML
增益推导，见 `_SONIC_ACTUATOR_PARAMETERS` 注释）。SONIC 专属的
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

Bundled smoke store 的公开再分发已由用户确认，但可审计的授权引用、权利人和具体许可
文本尚未提供。该状态记录在 `THIRD_PARTY_NOTICES.md`，补齐凭据前不得将相关数据纳入
公开 release。

## 与上游 gear_sonic 的逐项对照（2026-09 复查）

对照基准：`GR00T-WholeBodyControl/gear_sonic` 的 `sonic_release.yaml` 发布实验。
以下记录每处偏离的定性（上游一致 / 有意移植 / 已修复 / 保留差异）。

### action 链路（本分支已改为对齐 g1-wbt-dance）

- 上游 policy 是无 tanh 的无界 Gaussian（std 独立参数），wrapper 端 clip ±20；
  到位控为 `default + (0.25·effort/kp)·action`，可达包络约 `5·effort/kp`。
  本 env 曾声明的 `Box(-20, 20)` 正是照抄上游 clip 值，不是策略输出空间。
- 现行实现：`SonicJointPositionAction.action_space` 用共享的
  `joint_position_action_space_from_ctrl_ranges` 从关节限位反推紧致空间（wbt 同款），
  Sonic actor 转发 env 推导的 scale/bias。tanh ±1 = 每关节全行程；物理 scale
  `2.0·effort/kp`（SONIC 增益表）使包络为 `2·effort/kp`，约上游等效值的 40%。
- **行为变化声明**：手腕可达范围由 ±0.596 rad（行程 37%）恢复到 ±1.61 rad（全行程），
  tanh log-prob 修正与 action-rate 惩罚的作用域随之变化；2026-09-23 验证跑的 scale
  结论不再直接适用，全量训练前需重新验证。

### G1 参考观测布局：上游 bug 兼容（勿"修复"）

上游 `command_multi_future` 先拼 feature-major `[全部 P | 全部 V]` 再 reshape 成
`(F, 2J)`，帧被交错打乱；上游在 `trl/losses/token_losses.py` 自注 "temporal axis is
incorrectly flattened" 且 decoder 依赖该布局。本仓库 env 的 feature-major term 布局 +
`SonicActor._split` 的 reshape 逐字节复刻上游 flat 顺序（G1 stride 5 ≙ 0.1 s，
SMPL stride 1 ≙ 0.02 s，与上游 `dt_future_ref_frames` 一致）；
`test_sonic_split_reproduces_upstream_tokenizer_flat_order` 锁定该等价。

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
- sonic 无 `simulate_action_latency` 上游对应物，默认 False 与上游行为一致。

### 已修复的偏离

- **feet_acc 重置尖峰**：上游在 articulation 层用 reset 速度播种 previous velocity；
  本地曾清零后差分，产生 `(v-0)/dt` 假加速度。现 `reset_env` 用 teleport 目标帧的
  `clip.joint_vel` 播种。
- **NPZ 路径 fps 防护**：`SonicMotionClip.from_motion` 现要求 50 Hz（与 packed store
  一致）。上游对非 50 fps 数据做重采样而非报错，本仓库选择 fail-fast；换全量数据时
  请用 `scripts/motion/pack_sonic_data.py`（内含 wxyz→xyzw 转换与校验）重打包。

### 保留差异（有意决定，暂不跟进上游）

- **tracking_vr_5point_local 用上游默认 5 点集**（骨盆+腕 0.18 offset+双踝），而
  sonic_release 覆盖为 3 点（torso +0.5 m 虚拟点+双腕无 offset）。保留已验证行为，
  后续评估。
- **观测不加噪声**：上游 actor obs 有 gravity±0.05 / ang_vel±0.2 / dof_pos±0.01 /
  dof_vel±0.5，参考观测 ±0.05；本地全无。保留现状。
- **adaptive sampling 无尾部 bin mask**：上游同样没有（wbt 的 mask 是 wbt 特有）；
  本地 EMA+kernel 平滑与上游 failure-rate 比例式亦不同，两者均为合法实现。
- **joint_limit 惩罚**：本地加 soft_limit 0.9 与 cap 5.0（上游为裸 L1 超限）；
  **anti_shake** 用 torso_link 替代上游 head_link（g1_sonic 无 head）。

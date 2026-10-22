# g1-supermimic 任务设计

在 MotrixSim Manager 栈上复现 [SuperMimic](https://arxiv.org/abs/2610.28480)（参考实现
[UniSuperMimic](https://github.com/Motphys/UniSuperMimic)）的 G1 LAFAN1 全身跟踪任务。任务代码位于
`motrix_envs/src/motrix_envs/locomotion/supermimic/`，注册 Env ID `g1-supermimic`；用户文档见
`docs/source/{zh_CN,en}/user_guide/envs/whole_body_tracking/supermimic.md`。

## 数据平面：复用现有 LAFAN1 管线

LAFAN1 数据的构建与读取不复建，直接使用主分支已有链路：

- `scripts/motion/download_lafan.py` 从 HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset` 拉取 G1 重定向 CSV
  （与 UniSuperMimic 同源，`g1/` 下恰为上游 conf 列出的 40 段 clip）。本次为它补了 `--all` 批量模式，把整库转进
  `~/.cache/motrixlab/lafan/g1_npz`（任务默认语料目录，`SUPERMIMIC_MOTION_DIR` 覆盖，`:` 连接多目录——沿用 sonic 分支
  `SONIC_MOTION_DIR` 的约定）。
- `motrix_envs.motion.converters.lafan_converter` 做正向运动学烘焙，产出 schema v1 NPZ（50 Hz）。
- `MotionLibrary` 把语料目录拼接为带 `frame_clip_end` 边界的全局帧轴；`WbtMotionClip` 提供 kernel 侧数值视图。

## 任务平面

`SuperMimicMotionCommand` 继承 `WbtMotionCommand`：`update` 的逐帧重锚定与 `advance` 的 clip-end hold 语义原样复用；
`reset_env` 重写为 SuperMimic 的逐 clip 一秒分桶抽取（`build_start_bins` 从 `frame_clip_end` 推出桶表；kernel 从共享
`bin_cdf` 抽桶、桶内均匀抽帧，并把起始桶写回 `start_bin`）；`reset` 只推进 `reset_counter`（宿主侧采样器用它检测
episode 边界）。

- **观察**：策略组 = 上游 99 维 proprioception（q−default、qd、局部角速度、重力、上一步动作、相对参考旋转 6D，逐项
  噪声与上游一致）+ 10 帧未来目标（每帧 73 维，当前帧参考根系 yaw 系；未来帧在所属 clip 末帧截断）。价值组在此基础上
  去噪并追加当前帧参考关节（58）、参考根误差（3）与锚定跟踪误差（45）。
- **奖励/终止**：WBT 奖励核就是 SuperMimic 的六项跟踪配方，任务 cfg 按上游权重（1/1/1/1 + 0.5/0.5，action rate
  −0.1，soft-limit −2.0）；manager 对奖励统一乘 `ctrl_dt`，故每步回报尺度与上游 `* dt` 一致。终止 = 上游高度 0.25 /
  倾角 0.8 + `motion_clip_end`（episode 在所属 clip 末帧结束，配合 `hold_at_clip_end`）+ dof 数值安全网；10 s 时限。
- **重置**：WBT teleport reset 的默认噪声恰为上游量级，原样使用。

## 起点采样：回报驱动分桶混合

`SuperMimicStartSampler`（宿主侧，由 `SuperMimicManagerEnv.step` tick）移植上游
`start_bin_probabilities` / `update_start_bin_returns`：

- 逐 lane 累计折扣回报（γ=0.99）；reset 计数器变化检测 episode 结束（失败/时限/clip 末/auto-reset 全覆盖）。
- 失败（高度/倾角）终止不计剩余价值；clip 结束与时限按 `γ^L·r̄_L/(1−γ)` bootstrap（上游把 clip end 记作 time-out）。
- 桶回报 EMA `G_b ← 0.8·G_b + 0.2·G`（首次 credit 落在 0.2·G，与上游 index_put_ 语义一致）；混合分布
  `p = 0.5/N + 0.4·softmax(−25 G/G_max) + 0.1·w̄`，`G_max = 5·dt/(1−γ) = 10`，`w̄` 为 softmax 的 0.999/0.001 EMA
  留存。构造期 bookend（`attach` 早于首次 reset）只锚定起始桶、不 credit。
- 统计以 `supermimic_*` metrics 进入训练曲线。

## 物理域随机化（MotrixLab 重置项）

复用 humanoid-walk 随机化的写入通道（`GeomFrictionWrite` / `LinkComWrite`）：

- `SuperMimicDynamicsResetCfg`（`sim_reset.dynamics`）：每次 episode 重置采样地面摩擦缩放因子 U(0.3, 1.2)（乘名义
  摩擦矢量，上游对机器人碰撞几何设绝对值）与 torso_link 质心偏移 ±(0.025, 0.05, 0.05)；play 模式退化为名义值。
- 采样值同时写入 motion command 的 4 维仿真上下文 `[摩擦因子 | 质心偏移]`，供世界模型条件化与 critic 特征
  （`normalize_context`：摩擦 (x−0.75)/0.45，质心 /0.05）——上游 xi 的子集；名义值由 `GeomFrictionQuery` /
  `LinkCenterOfMassQuery` 查询。质量/惯量/增益/延迟未随机化。

## 可微世界模型（motrix.supermimic）

MotrixLab 环境是 NumPy 栈（ArrayEnv/ManagerEnv + numba kernel），torch 侧整体放在 RL 插件
`motrix_rl_builtin/supermimic/`（该包本就依赖 torch），环境不引入 torch。

**环境侧（NumPy 契约）**：

- `SuperMimicMotionCommand.update` kernel 每步发布展平状态 `[pos 3 | quat xyzw 4 | v 3 | w 3 | q N | qd N]`
  （71 维）与 4 维仿真上下文；帧号在 `steps`。学习器在 `env.step` 前后快照得 (s, s1)。

**插件侧**（注册为 `motrix` framework 的 `supermimic` agent，torch 后端，同步训练器）：

- `contracts.py`：NumPy 结构化 Protocol（motion command 的 clip/state/context/steps 缓冲 + env 表面），插件不
  import 任务包；
- `mdp.py`：`build_supermimic_mdp(env)` 从 NumPy 表面（语料 clip、机器人模型文件与默认 key pose、任务关节/跟踪
  body 序、ctrl_dt）构建 torch MDP；`SuperMimicKinematics` 从 MotrixSim 模型读运动链做可微 FK（与
  `forward_kinematic` 对齐：位置 <1e-4 m，旋转 <2e-3 rad，测试保障）；`SuperMimicTorchMDP` 提供解析奖励（与
  manager 奖励组同配方同锚定，×ctrl_dt）、软终止、99 维 proprio（上游 OBS_NOISE 表）、目标特征（与 kernel 逐项
  对齐）、critic 特征、imagination 起点有效性；四元数 xyzw，6D 取矩阵前两行；
- `networks.py` / `replay.py` / `geometry.py`：移植上游潜增量世界模型（in/out RunningNorm + 误差地板）、目标编码
  策略、λ-return critic、(time, env) 回放；四元数改为 xyzw 对齐环境布局；
- `learner.py`：显式接收 mdp；采集（解析 obs + 噪声 + 策略动作 clip 到 env Box）与训练（wm 多步窗口 MSE、想象
  rollout 策略目标 + 软终止折扣 + critic bootstrap、critic λ-回归）循环；
- `trainer.py`：同步训练器（TrainingLogger/checkpoints/续训）与确定性 play；上游 AMP/compile 保留为配置项
  （仅 CUDA 生效）。

任务配置 `configs/task/g1-supermimic/motrix.supermimic.yaml` + 算法基础
`motrix_rl/configs/algo_base/motrix.supermimic.yaml`（schema 由 framework 注册自动生成）。

**逐 clip 评估**：`SuperMimicEnvCfg.for_eval()` 给出确定性变体（关闭采样/噪声/随机化/时限，hold 在 clip 末）；
`SuperMimicMotionCommand.requested_frames` 宿主缓冲在 `reset_env` 中被消费（采纳置入帧并登记所属桶），评估把每条
lane 钉在其 clip 首帧后走 `env.reset`。`evaluate.py`（插件）持有纯 torch 评估循环（解析 obs 无噪声、策略确定性、
`failed = 终止 ∧ ¬clip_end`、逐 lane 回报/步数记账，归一上限 = (clip 长 −1)×5×dt）；checkpoint 携带策略超参
（`args`）。`scripts/eval_supermimic.py` 负责接线：registry 取 cfg → `for_eval()` → 每 clip 一条 lane → 构建插件
MDP → 读 checkpoint → 评估 → 写 `<ckpt>_eval.json` + 追加 `eval.csv`（支持 `--ckpt` 或 `--run_dir` manifest 解析）。

## 与上游的偏差（有意为之）

1. **拓扑**：`motrix.supermimic` 为同步单进程训练器（FastSAC 的异步 collector/learner 拓扑未移植）。
2. **上下文随机化 ξ**：实现了子集（地面摩擦缩放 + 躯干质心，且通过仿真上下文条件化世界模型/critic）；上游的重置
   关节偏移及其在动作目标（default+偏移+αa）与观察（q−偏移）中的耦合未实现。
3. **clip 结束的 RL 语义**：以 termination 实现，FastSAC 缓冲中不 bootstrap（仅起点采样器按截断处理）。根因是 core
   的 truncation 目前只来自 episode 时限；世界模型想象侧使用软终止概率，与上游一致。
4. **细节口径**：相对旋转 6D 用矩阵前两行（行主序，等价连续表示）；奖励对比当前帧（evaluate-then-advance），上游
   对比下一帧；地面摩擦随机化以缩放因子作用于地面几何（上游直接设机器人碰撞几何绝对摩擦）。

## 验收

- `motrix_envs/tests/test_supermimic_env.py`：桶划分、混合分布与上游公式逐项对齐（均匀初始、0.8/0.2 EMA、hard/soft
  bootstrap、构造 bookend 不 credit）、cfg 契约（奖励表、噪声表、for_play）、numba kernel（观察维度 829/935、目标帧
  逐特征对 scipy 参考验证、clip 末终止并重采样）。
- `motrix_envs/tests/test_supermimic_env.py` 另覆盖：随机化上下文范围与状态缓冲有限性（NumPy 侧，随 numba env 测试）。
- `motrix_rl_builtin/tests/test_supermimic.py`：注册契约、SO(3) 回环、回放窗口/起点语义、fake-env 学习器全循环
  （采集 + wm 更新 + 想象策略更新 + checkpoint 往返）。
- `tests/test_supermimic_integration.py`（numba+integration）：env 契约与 MDP 维度、torch FK 对齐 MotrixSim FK、
  目标/proprio/奖励对齐 manager 输出、imagination 起点有效性、真实 env + 真实 MDP 的端到端学习器循环与两轮
  trainer 运行（含 checkpoint manifest）。
- 任务配置经 `tests/test_task_configs.py` 全量组合验证（含 `motrix.supermimic`）；双语用户文档 sphinx `-W` 构建通过，
  poster 由 `docs/scripts/generate_env_snapshot.py` 生成并人工检查构图。

## 与 UniSuperMimic 的迁移完成列表

以 UniSuperMimic（`src/unisupermimic/`）为参考的逐组件迁移状态：

| 参考组件 | MotrixLab 对应 | 状态 |
| --- | --- | --- |
| MotionLib（CSV→torch 语料、一秒分桶、goal） | `download_lafan.py --all` + `lafan_converter` + `MotionLibrary`（NPZ 语料）+ `SuperMimicTorchMDP.goal` | 完成（数据源同一 HF 数据集，语料为 NPZ v1） |
| SuperMimicCommand（帧推进、episode 记账、重置噪声） | `SuperMimicMotionCommand`（继承 WbtMotionCommand；WBT reset 项噪声同上游） | 完成 |
| start_bin_probabilities / update_start_bin_returns（回报分桶采样） | `SuperMimicStartSampler`（宿主 tick） | 完成（公式逐项对齐，测试保障） |
| supermimic_obs / add_noise（99 维 proprio + OBS_NOISE/SCALE） | manager 观察组 + `SuperMimicTorchMDP.obs/add_noise` | 完成（OBS_SCALE 的输入归一化由 RunningNorm 承担） |
| supermimic_goal（未来帧目标） | manager goal 观察项 + `SuperMimicTorchMDP.goal` | 完成（6D 取前两行，等价表示） |
| MDP.reward（六核跟踪 + 正则 ×dt） | manager 奖励组 + `SuperMimicTorchMDP.reward` | 完成（逐项对齐，测试容差 0.02） |
| MDP.termination（高度 0.25 / 倾角 0.8） | `bad_ref_z` / `bad_ref_ori` 终止项 | 完成 |
| supermimic_timeout / supermimic_clip_end | 10 s 时限（truncation）+ `motion_clip_end` 终止项 | 完成（clip_end 在 FastSAC 缓冲按 terminated，偏差已记录） |
| SuperMimicContextEvent（ξ：摩擦/质心/关节偏移） | `SuperMimicDynamicsResetCfg`（摩擦缩放 + 躯干质心，写入仿真上下文） | 部分完成（关节偏移及其动作/观察耦合未做） |
| G1Kinematics（可微 FK） | `SuperMimicKinematics`（插件侧，从 MotrixSim 模型建树） | 完成 |
| MDP.critic_features / normalize_context | `SuperMimicTorchMDP.critic_features / normalize_context` | 完成（上下文 4 维而非 33 维） |
| networks（WorldModel/Policy/Critic/RunningNorm） | `motrix_rl_builtin.supermimic.networks` | 完成（xyzw 四元数） |
| ReplayBuffer（窗口/起点采样） | `motrix_rl_builtin.supermimic.replay` | 完成 |
| geometry（SO(3)/6D/invariant） | `motrix_rl_builtin.supermimic.geometry`（+ `mdp.py` 特征助手） | 完成 |
| SuperMimicLearner（wm/policy/critic 更新、调度、断点） | `motrix_rl_builtin.supermimic.learner`（显式 mdp 注入） | 完成 |
| env 发布 transition/state info | NumPy 缓冲契约（motion command 的 state/steps/context） | 完成（改为缓冲快照而非 info dict） |
| train_supermimic 入口 / Hydra conf | `mxlab train task=g1-supermimic/motrix.supermimic` + `algo_base/motrix.supermimic.yaml` | 完成（同步单进程） |
| evaluate.py（逐 clip 确定性评估、norm_return/失败统计） | `motrix_rl_builtin.supermimic.evaluate` + `scripts/eval_supermimic.py` + `for_eval()` / `requested_frames` | 完成（不含 render / sim2sim / watch 轮询） |
| render.py / video 导出 | — | 未迁移（复用 `mxlab play` 渲染） |
| sim2sim.py 校验 | — | 未迁移（按要求跳过） |
| reset_to_frames 逐帧评估入口 | command 的 `requested_frames` 缓冲（宿主置帧 → `env.reset`） | 完成 |
| mjwarp 并行后端 / 4096 env GPU 采集规模 | MotrixSim NumPy 后端（task yaml 4096 env 配置保留） | 适配（后端差异） |
| FastSAC model-free 基线 | `motrix.fastsac` 任务配置 | 完成（MotrixLab 自有） |

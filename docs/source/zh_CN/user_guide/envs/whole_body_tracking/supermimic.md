# g1-supermimic：LAFAN1 语料目标条件跟踪

`g1-supermimic` 在 MotrixSim 上复现 [SuperMimic](https://arxiv.org/abs/2610.28480)（参考实现
[UniSuperMimic](https://github.com/Motphys/UniSuperMimic)）的 G1 全身跟踪任务：策略从 LAFAN1 重定向语料中逐帧跟踪
参考动作，观察中带有未来参考帧目标（goal），episode 起点按 SuperMimic 的基于回报的一秒分桶混合分布采样，并在所属
clip 的最后一帧结束。机器人模型、关节限位与增益由 `UnitreeG129Dof` 资产提供；任务本身拥有命令、观察、奖励、终止与
episode 行为。

与内置 `g1-wbt-*` 任务的区别：策略观察是目标条件 proprioception（不含当前帧参考关节），episode 结束于 clip 末尾而非
中途重采样，起点采样由回报统计驱动而非失败率课程。

任务提供两套训练入口：`motrix.fastsac`（model-free）与 `motrix.supermimic`（可微世界模型，见下文
「世界模型训练」一节）。

## 语料准备

任务默认读取 `~/.cache/motrixlab/lafan/g1_npz`（每个 LAFAN1 clip 一个 MotrixLab motion NPZ v1，50 Hz）。语料不随包发布，
首次使用前构建一次：

```bash
uv run scripts/motion/download_lafan.py --all
```

该命令从 HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset` 下载 40 个 G1 重定向 clip（30 fps CSV），并经正向运动学转换
为 50 Hz NPZ（转换细节见 [Motion 文件格式](motion_format.md)）。设置环境变量 `SUPERMIMIC_MOTION_DIR` 可指向其他语料目录，
多个目录用 `:` 连接。

## 运行

```bash
# 训练（FastSAC，model-free）
mxlab train task=g1-supermimic/motrix.fastsac

# 训练（SuperMimic 可微世界模型）
mxlab train task=g1-supermimic/motrix.supermimic

# 小规模 smoke test
mxlab train task=g1-supermimic/motrix.fastsac num_envs=64 algo.trainer.num_learning_iterations=100

# 回放：从语料第 0 帧起顺序播放，跨 clip 边界连续跟踪
mxlab play env=g1-supermimic num_envs=16
```

## 动作空间

`Box([-1, 1]^{29})`。第 $j$ 维映射到关节位置目标

$$
q_j^{target} = q_j^{default} + \alpha_j a_j,\quad \alpha_j = 0.25\,\frac{effort_j}{kp_j}
$$

其中 $q^{default}$ 是机器人 `default` 关键姿态（与上游 stand 姿态一致：髋 −0.312、膝 0.669、踝 −0.363、肘 0.6、肩 roll
±0.2），$\alpha_j$ 逐关节来自模型执行器的力矩上限与位置增益，目标再被关节限位截断。

## 观察空间

| 观察项 | 策略 | 价值 | 含义 |
| --- | ---: | ---: | --- |
| 关节位置相对默认姿态 | 29 | 29 | rad，噪声均匀分布于 [−0.01, 0.01) |
| 关节速度 | 29 | 29 | rad/s，噪声 [−0.5, 0.5) |
| 根系局部角速度 | 3 | 3 | rad/s（pelvis 系），噪声 [−0.2, 0.2) |
| 重力方向（根系系） | 3 | 3 | 噪声 [−0.05, 0.05) |
| 上一步动作 | 29 | 29 | 无噪声 |
| 相对参考旋转 6D | 6 | 6 | 当前帧参考根系旋转相对机器人根系，噪声 [−0.05, 0.05) |
| 目标（未来参考帧） | 730 | 730 | 10 帧 × 73 维，见下 |
| 当前帧参考关节位置/速度 | — | 58 | 无噪声 |
| 参考根位置误差（机器人系） | — | 3 | 无噪声 |
| 锚定跟踪误差 | — | 45 | 14 个跟踪 body 的位置误差 + 根位置误差（世界系） |

策略组共 829 维，价值组共 935 维；价值组与策略组重叠项不加噪声（privileged）。

每帧目标特征（73 维）在当前帧参考根系的 yaw 系下表达：根位置增量 (3)、相对根系旋转 6D (6)、根系线速度 (3)、根系角速度
(3)、参考关节位置 (29)、参考关节速度 ×0.1 (29)。第 $i$ 个未来帧取 $\min(step+i,\ \text{clip 末帧})$，即在所属 clip 的
末帧截断，不跨 clip。

## 奖励设计

奖励为 SuperMimic 的六项核跟踪与两项正则，总回报按控制步长 `ctrl_dt = 0.02` 加权（与上游 `* dt` 一致），理想跟踪时每步
奖励上限 $5 \times 0.02 = 0.1$：

| 奖励项 | 计算 | 设计目的 |
| --- | --- | --- |
| `motion_relative_body_position_error_exp` | 逐帧锚定的 14 个跟踪 body 位置误差均值的 $\exp(-\bar{e^2}/0.3^2)$，权重 1.0 | 驱动整体身体构型跟踪 |
| `motion_relative_body_orientation_error_exp` | 同上旋转距离核，$\sigma=0.4$，权重 1.0 | 驱动姿态跟踪 |
| `motion_global_body_lin_vel` | 全系线速度误差核，$\sigma=1.0$，权重 1.0 | 匹配身体速度 |
| `motion_global_body_ang_vel` | 全系角速度误差核，$\sigma=3.14$，权重 1.0 | 匹配角动量（翻转、跳跃） |
| `motion_global_ref_position_error_exp` | 根位置误差核，$\sigma=0.3$，权重 0.5 | 弱约束全局根位置 |
| `motion_global_ref_orientation_error_exp` | 根旋转误差核，$\sigma=0.4$，权重 0.5 | 弱约束全局根朝向 |
| `action_rate_l2` | $\lVert a_t - a_{t-1}\rVert_2^2 \times (-0.1)$ | 抑制动作抖动 |
| `limits_dof_pos` | 0.9 软限位越界量 × (−2.0)，无上限截断 | 保持关节可行域 |

锚定语义：每个控制步把参考 body 目标重锚到机器人当前根系位置（平面）与参考根系高度（yaw 对齐），与逐项奖励的跟踪
目标一致，因此平面漂移不会瞬间放大误差。

## 终止条件

| 类型 | 状态 | 条件 | 含义 |
| --- | --- | --- | --- |
| 参考高度偏差 | `terminated` | 根 z 与参考差 > 0.25 m | 跟踪失败（跌倒、悬空） |
| 参考倾角偏差 | `terminated` | 根姿态 $R_{zz}$ 与参考差 > 0.8 | 跟踪失败（倾覆） |
| clip 结束 | `terminated` | 步进到所属 clip 最后一帧 | 成功结束；起点采样按截断处理 |
| 关节越界 / NaN | `terminated` | 越界 > 0.5 rad 或发散 | 数值安全网，正常 rollout 不触发 |
| 时限 | `truncated` | 10 s（500 控制步） | 时间截断 |

clip 结束在 RL 缓冲中按 `terminated` 处理（详见下文与上游的差异）；回报采样器将其视同截断做 bootstrap。

## 重置逻辑与随机化

确定性部分：episode 起点帧决定全部状态 —— 传送该帧的根位姿、根速度、关节位置/速度到仿真。

| 随机量 | 采样 | 时机 | 目的 |
| --- | --- | --- | --- |
| 起点帧 | 基于回报的一秒分桶混合：50% 均匀 + 40% softmax(−25 G_b/G_max) + 10% 历史 softmax 留存 | 每次 episode 重置 | 课程：难桶多练，学会后回落 |
| 地面摩擦系数 | 均匀缩放因子 ∈ [0.3, 1.2)（作用于名义摩擦矢量） | 每次 episode 重置 | 物理域随机化；采样值写入仿真上下文 |
| 躯干质心偏移 | 均匀 [−0.025, 0.025) / [−0.05, 0.05) / [−0.05, 0.05) m（torso_link） | 每次 episode 重置 | 物理域随机化；采样值写入仿真上下文 |
| 根位置噪声 | 均匀 [−0.05, 0.05) m（x/y），[−0.01, 0.01) m（z） | 重置 | 初始状态随机化 |
| 根朝向噪声 | 欧拉角均匀 [−0.1, 0.1) / [−0.1, 0.1) / [−0.2, 0.2) rad | 重置 | 初始状态随机化 |
| 根线速度噪声 | 均匀 [−0.5, 0.5) / [−0.5, 0.5) / [−0.2, 0.2) m/s | 重置 | 初始状态随机化 |
| 根角速度噪声 | 均匀 [−0.52, 0.52) / [−0.52, 0.52) / [−0.78, 0.78) rad/s | 重置 | 初始状态随机化 |
| 关节位置噪声 | 均匀 [−0.1, 0.1) rad，截断到限位 | 重置 | 初始状态随机化 |
| 观察噪声 | 见观察空间表 | 每次观察 | 观察随机化 |

分桶回报统计：每个桶维护 EMA 估计 $G_b \leftarrow 0.8\,G_b + 0.2\,G_{episode}$；episode 回报为折扣和（γ = 0.99），失败
（高度/倾角）终止不计剩余价值，clip 结束与时限按 $\gamma^L \bar{r}_L/(1-\gamma)$ bootstrap；$G_{max} = 5 \times 0.02 /
(1-\gamma) = 10$。采样统计写入 `supermimic_*` metrics。

物理域随机化采样（摩擦缩放因子 + 躯干质心偏移，4 维）同时写入 motion command 的仿真上下文缓冲，供世界模型与
critic 条件化（上游 xi 的子集；play 模式退化为名义值）。关节质量、惯量、执行器增益与延迟保持模型默认。

本任务的观察噪声见表格；play 模式关闭起点采样与全部重置噪声，物理域随机化退化为名义值（摩擦因子 1、质心偏移 0），
并从语料第 0 帧起跨 clip 顺序播放。

## 配置

`SuperMimicMotionCommandCfg` 的常用字段（`mxlab train task=g1-supermimic/motrix.fastsac commands.motion.<field>=...`）：

- `num_goal_frames`（默认 10）：目标观察的未来帧数；
- `supermimic_gamma`（默认 0.99）：分桶回报折扣，需与 `algo.agent.gamma` 一致；
- `supermimic_sampling`（默认 `true`）：关闭后退化为均匀桶采样（play 模式自动关闭）；
- `sim_reset.dynamics.friction_range`（默认 (0.3, 1.2)）与 `sim_reset.dynamics.com_offset_noise`（默认
  (0.025, 0.05, 0.05)）：物理域随机化范围，退化为单点即关闭。

## 世界模型训练（motrix.supermimic）

`motrix.supermimic` 移植 SuperMimic 的可微世界模型训练器：动力学为学习的潜状态增量模型（预测根系系下的局部状态
增量并积分），奖励、软终止、观察与目标头均为解析可微函数（`motrix_rl_builtin.supermimic.mdp`，其前向运动学与
MotrixSim FK 对齐、特征与 manager kernel 逐项对齐）；目标条件策略在想象 rollout 上训练（λ-return critic
bootstrap，软终止折扣）；采集与更新按 `utd` 交替，前 `warmup_steps` 步只采集，前 `wm_only_iters` 轮只训世界模型。
环境侧保持 NumPy 栈：motion command 每步发布展平状态、帧号与仿真上下文缓冲，torch MDP 由训练器从这些 NumPy
表面（语料 clip、机器人模型与默认姿态、控制周期）构建，插件不依赖任务包。

```bash
mxlab train task=g1-supermimic/motrix.supermimic
mxlab train task=g1-supermimic/motrix.supermimic algo.agent.goal_frames=10
```

`algo.agent.goal_frames` 需与 `commands.motion.num_goal_frames` 一致（默认均为 10）；超参数默认值对齐上游，学习率按
余弦衰减到 3%。世界模型条件化在归一化的仿真上下文（摩擦因子 + 躯干质心偏移）上，与采集时的物理域随机化对应。

## 逐 clip 评估

`scripts/eval_supermimic.py` 对训练 checkpoint 做确定性整段评估（移植上游 evaluate，不含 render / sim2sim）：每个
LAFAN1 clip 占一条环境 lane，episode 固定从该 clip 第 0 帧开始，关闭观察、探索、重置噪声与物理域随机化
（`for_eval()` 变体），逐 lane 播放到 clip 末尾（成功）或高度/倾角失败为止。报告：

- `norm_return`：各 clip 回报 /(clip 长度 × 每步奖励上限) 的均值；
- `norm_return_framew`：按帧数加权的全局归一回报；
- `n_failed` / `failed`：失败 clip 列表（clip 末结束不计失败）；
- `per_clip`：逐 clip 的归一回报、存活步数与是否失败。

```bash
# 评估某个 checkpoint
python scripts/eval_supermimic.py --ckpt runs/g1-supermimic/<run>/checkpoints/latest.pt

# 或解析 run 目录的 manifest 策略
python scripts/eval_supermimic.py --run_dir runs/g1-supermimic/<run>
```

结果写入 `<checkpoint>_eval.json`，并向 checkpoint 目录的 `eval.csv` 追加一行。策略超参数（goal 帧数、hist、网络
宽度）随 checkpoint 保存，评估时自动重建网络。

## 与上游 UniSuperMimic 的差异

- 可微世界模型训练器已移植（`motrix.supermimic`，同步单进程）；`motrix.fastsac` 仍可作为 model-free 基线；
- 物理域随机化实现为 MotrixLab 重置项：地面几何摩擦按因子缩放（上游直接设机器人碰撞几何绝对摩擦）+ 躯干质心偏移，
  采样值写入仿真上下文供世界模型/critic 条件化；上游 xi 的重置关节偏移及其在动作目标与观察中的耦合未实现；
  关节质量、惯量、执行器增益与延迟保持模型默认；
- clip 结束以终止项实现：FastSAC 缓冲中不做 bootstrap，仅起点采样器按截断处理；世界模型的想象 rollout 使用软终止
  概率（与上游一致）；
- 相对旋转 6D 取旋转矩阵前两行（行主序），与上游的前两列等价（同一旋转的连续 6D 表示）；
- 奖励对比的参考帧为当前帧（evaluate-then-advance），上游为下一帧。

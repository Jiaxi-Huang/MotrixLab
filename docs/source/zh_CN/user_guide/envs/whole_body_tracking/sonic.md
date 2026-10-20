# SONIC G1 动作跟踪

SONIC 是面向 Unitree G1 的 Manager 环境，并通过 FastSAC 的 `sonic` PolicyVariant 构建专用
actor。策略将滑窗本体感知历史、G1 未来参考、SMPL 未来参考和 hybrid（teleop）参考流组合
起来，经由上游列序 `[g1, teleop, smpl]` 的三值 multi-hot encoder mask 路由：smpl 原生行
同时激活 g1 encoder，并以 0.5 概率激活 teleop encoder，与上游 legacy 采样器一致。hybrid
流对齐上游 teleop encoder 输入：多未来帧腿部块（G1 步长网格上的 12 个腿部关节位置与
速度）加上当前帧尾部（VR 风格 3 点目标——双腕 + 躯干偏移点，位置与四元数均取
reference-anchor 系——以及 anchor 姿态）。该时序深度是 `g1-sonic` 的公共契约；训练、
LAFAN 数据和小规模验证都使用同一个环境。

任务配置把 `algo.variant` 直接内联在 `configs/task/g1-sonic/motrix.fastsac.yaml` 中。
其中 `model.num_future_frames` 同时决定环境 policy observation 宽度与 actor 输入宽度；
`auxiliary` 是五个上游命名辅助损失（`g1_recon`、`g1_smpl_latent`、`g1_teleop_latent`、
`teleop_smpl_latent`、`reencoded_smpl_g1_latent`）的权重，由 FastSAC 加权求和并记录
`aux_*` 指标。`g1_control_decoder_hidden_dims` 不出现在训练配置中：训练 actor 不构造该
decoder，并用 FastSAC policy head 替代它。

## 观察与动作

`g1-sonic` 的 policy observation 宽度为 1355（`num_future_frames=5` 时；G1/SMPL 块随该值
线性缩放，teleop 块另加 27 维常量当前帧尾部），privileged value observation 宽度为 1220。
value observation 合并了 gear_sonic 与 MimicLite 的 critic 契约：逐未来帧的参考关节位置与
速度加上机器人根系下的参考根位姿（对应 MimicLite `ref_root_*_future_b`）、MimicLite 式逐
身体"参考−实际"差分（位置、6D 朝向与世界系线/角速度，覆盖全部 tracked 身体，diff 步为
`[0, 1]`，两侧各在自身 pelvis 锚的投影-yaw 系下规范化），以及滑窗本体感知历史（每帧
`lin_vel | ang_vel | joint_pos | joint_vel | action`）。动作是 29 维归一化关节位置目标，动作
缩放和偏置在环境 action term 内完成；FastSAC actor 保持 identity action affine。

## 数据与运行

训练读取 NPZ 动作语料目录——每个文件是一个 MotrixLab motion schema v1 clip，携带
`ext_smpl_joints` / `ext_smpl_root_quat` 参考通道。默认语料位置即 converter 的缓存输出，
因此下面三步无需额外参数即可串起；自定义位置用 `SONIC_MOTION_DIR`（支持 `:` 分隔多个语料根目录）：

```bash
source .venv/bin/activate
SONIC_MOTION_DIR=$PWD/data/bones_seed_npz \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

## 语料轮换

完整语料（10 万+ clip）无法全部常驻内存，因此 `g1-sonic` 通过固定的双窗口帧轴流式消化语料
（`SonicMotionCommandCfg.rotation_period`，默认 4096 控制步、16 个解码线程；设
`rotation_period: null` 关闭）。
每个窗口持有从 seeded 无放回置换中取出的 `num_motions` 个 clip，遍历一遍完整语料需要
`ceil(语料规模 / num_motions)` 次换窗。窗口解码在后台线程中写入私有 staging 缓冲；发布只是对
预分配 clip 数组中退役半区的一次原地拷贝——每次换窗一次有界 memcpy，步进路径上没有任何其他开销。

换窗从不强制重置 env：重置采样与 clip 末帧重锚立刻使用新窗口，仍在跟踪旧窗口的 lane 会跑到
其自然的 episode 结束，且退役半区仅在没有 lane 再引用它之后才会被覆写。对 off-policy replay 而言，
这使每次换窗都成为平缓的混合比变化——一个好的默认值是每 1–2 个 `buffer_size` 控制步换窗一次：
取 `buffer_size` 时 buffer 永远混合两个窗口，取 `2 × buffer_size`（`g1-sonic` 默认）时每个窗口的
数据在下次换窗前完整离开 buffer，每窗都能得到完整的消化。根据 publish 日志校准：窗口解码耗时应
明显小于 period，排空等待应低于 episode 长度上界。adaptive sampling 默认开启（gear_sonic release
`enable: true`；play/eval 配置经 `for_play()` 关闭，`clip_round_robin` 则取代它）。开启后，采样遵循
gear_sonic 的 per-clip 失败率契约，常驻语料与轮换窗口共用同一实现：每条源 clip 按
`adaptive_bin_size` 帧（50 Hz 下默认 1 秒）切 bin，每个 bin 维护累计的失败数与 episode 数
（同初始化为 `adaptive_init_count`）；每个 lane-step 给当前所在 bin 记 `1 / bin_length`
次 episode，每次终止给结束位置所在的 bin 记一次失败。采样概率由逐 bin 失败率（按均值 ×
`failure_rate_max_over_mean` 截断）与 `uniform_ratio` 均匀下限混合，再经 sequence-length-agnostic
的 per-clip 权重重塑，使每条 clip 携带相同的总质量。episode 起点最多回退
`pre_failure_sample_window` 帧，让策略练习进入困难片段的过程而非直接传送进去。统计属于源
clip 并跨窗口保留：排空中的 lane 仍向其退役 clip 上报，clip 回归时学到的 bin 分布随之恢复。
活跃窗口的 CDF 每 `adaptive_update_period` 控制步重发布一次，窗口发布时立即重发布。
`adaptive_failure_rate` 步级指标报告策略在 adaptive sampling 当前分发内容上的真实失败率
（失败数 / episode 曝光量，指数衰减，半衰期约 7k 控制步）：将它与
`adaptive_sampling_entropy` / `adaptive_sampling_top1_prob` 对照阅读，可以区分策略进步与
分布偏移——采样器收敛的同时该比率下降，说明策略正在掌握当前被分发的困难内容。
rotation 还把学到的难度用在 clip 级，对应上游的 motion resample：每个窗口有
`rotation_difficulty_ratio`（默认 0.5）比例的 clip 槽位按采样器的逐 clip 权重有放回抽取
（困难 clip 跨窗口史频繁地重现），其余槽位仍来自无放回的覆盖排列，因此每条 clip 依旧随
pass 出现，buffer 多样性保有最坏情况下的下限。两种情况下窗口发布都保持 drain-gated：与
上游换批时的 `reset_all` 不同，episode 不会被强制重置（off-policy replay 不应看到人为截断）。
显式关闭 adaptive sampling 时，rotation 回退到原有的均匀帧采样，既不改变随机数序列，也不
分配任何 adaptive 状态。小于一个窗口的语料根会回退到普通的常驻加载，冒烟语料不受影响。

## 构建语料

从 HuggingFace 下载配对的 BONES-SEED 子集（G1 重定向 CSV + SMPL PKL），再转换成语料目录：

```bash
python scripts/motion/download_bones_seed.py   # 原始子集 -> ~/.cache/motrixlab/bones_seed/g1
python scripts/motion/convert_bones_seed.py --workers 8  # 语料 -> ~/.cache/motrixlab/bones_seed_npz/g1
```

下载器按字节预算流式读取两个归档（默认保留约 6000 对、落盘 3-4 GB）；转换器对每个 clip
用 G1 模型做正向运动学并写出一个 schema v1 npz，名称命中默认关键词列表（床/椅/台阶等
道具场景与倒立/侧翻等极限动作，列表来自 gear_sonic）的 clip 会被剔除
（`--filter-keywords=` 关闭）。`--workers N` 起 N 个转换进程，各持一份模型与
FK buffer；已存在的输出自动跳过，中断后可直接续跑。仓库不捆绑任何动作数据。

也可以用 LAFAN1 动捕数据构建同构语料（retarget CSV 通路 + 原始 BVH 的 bvh→smpl 通路，
约 40 对公开数据）：

```bash
python scripts/motion/download_lafan4sonic.py   # 原始对 -> ~/.cache/motrixlab/lafan4sonic
python scripts/motion/convert_lafan4sonic.py    # 语料 -> ~/.cache/motrixlab/lafan4sonic_npz
```

随后以 `SONIC_MOTION_DIR=~/.cache/motrixlab/lafan4sonic_npz` 训练；`:` 拼接多个目录可与
BONES-SEED 语料混用。该路径独立于 WBT 所用的 LaFan 单 clip 烘焙，互不影响。

## 小规模验证

先构建一个小语料（小预算几分钟即可拿到少量配对），再用同一个 `g1-sonic` 配置，
只覆盖并行数和训练长度：

```bash
python scripts/train.py task=g1-sonic/motrix.fastsac \
  num_envs=32 play_num_envs=4 checkpoint.interval=100 \
  algo.trainer.num_learning_iterations=1000
```

MotrixLab checkpoint 的回放方式见
[训练产物](../../tutorial/training/runs_and_checkpoints.md)。

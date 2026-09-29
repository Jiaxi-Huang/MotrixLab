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
线性缩放，teleop 块另加 27 维常量当前帧尾部），privileged value observation 宽度为 890。
动作是 29 维归一化关节位置目标，动作缩放和偏置在环境 action term 内完成；FastSAC actor
保持 identity action affine。

## 数据与运行

训练读取 NPZ 动作语料目录——每个文件是一个 MotrixLab motion schema v1 clip，携带
`ext_smpl_joints` / `ext_smpl_root_quat` 参考通道。默认语料位置即 converter 的缓存输出，
因此下面三步无需额外参数即可串起；自定义位置用 `SONIC_MOTION_DIR`（支持 `:` 分隔多个语料根目录）：

```bash
source .venv/bin/activate
SONIC_MOTION_DIR=$PWD/data/bones_seed_npz \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

## 构建语料

从 HuggingFace 下载配对的 BONES-SEED 子集（G1 重定向 CSV + SMPL PKL），再转换成语料目录：

```bash
python scripts/motion/download_bone_seed.py   # 原始子集 -> ~/.cache/motrixlab/bones_seed/g1
python scripts/motion/convert_bones_seed.py --workers 8  # 语料 -> ~/.cache/motrixlab/bones_seed_npz/g1
```

下载器按字节预算流式读取两个归档（默认保留约 6000 对、落盘 3-4 GB）；转换器对每个 clip
用 G1 模型做正向运动学并写出一个 schema v1 npz，名称命中默认关键词列表（床/椅/台阶等
道具场景与倒立/侧翻等极限动作，列表来自 gear_sonic）的 clip 会被剔除
（`--filter-keywords=` 关闭）。`--workers N` 起 N 个转换进程，各持一份模型与
FK buffer；已存在的输出自动跳过，中断后可直接续跑。仓库不捆绑任何动作数据。

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

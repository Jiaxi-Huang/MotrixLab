# SONIC BONES-SEED NPZ 语料设计

## 摘要

让 SONIC G1 任务直接从一个 NPZ 语料目录训练：每个 `.npz` 是一个 clip（MotrixLab motion
schema v1 + `ext_smpl_joints` / `ext_smpl_root_quat` 参考通道），由移植自 multi-clip infra 的
`MotionLibrary` 拼接为单一全局帧轴。数据来自 BONES-SEED 发布（G1 retargeted CSV + SMPL PKL），
新增 `download_bone_seed.py` 流式子集下载与 `bones_seed_converter` 转换，不再经过 packed
mmap store。

## 背景与动机

现有 SONIC 数据路径是 `motrixlab_sonic_packed_v1` mmap store：全量数据需要先转换成 robot/smpl
成对 npz，再经 `pack_sonic_data.py` 打包成逐数组 `.npy` + manifest。multi-clip infra
（`feat/motion-multi-clip-infra`，见 [motion-multi-clip-infra](./motion-multi-clip-infra.md)）
已经证明"一个文件一个 clip + `ext_` 通道"的数据平面足够支撑多 clip 拼接训练，其设计文档明确
预留了 SONIC 作为第一个新消费者：提供多文件 library、声明 SMPL 通道，无需 command 复制、
无需 clip 数据子类、无需打包脚本。本设计兑现该预留。

## 数据平面

`SonicMotionClip.from_corpus(paths, ...)` 构造 `MotionLibrary(paths, extension_channels=
("smpl_joints", "smpl_root_quat"), fps=50)` 并 assemble：跨文件校验（fps/joint/body 名单/通道
shape/有限性）→ 按任务序重排 → 拼接出逐帧 `frame_clip_end`（clip 边界的唯一运行时载体，per-clip 元数据可由它差分恢复）。
SMPL 数组从 extensions Map 复制进 `SonicMotionClip` 的专用字段（观测 kernel 直接索引
`clip.smpl_joints` / `clip.smpl_root_quat`，不经过 Map 查询）。语料路径是 SONIC 的唯一
数据源——单 clip 就是它的单文件特例。

## 时间线与播放

`SonicMotionCommand.advance` 的到尾换段语义（`steps > frame_clip_end[step]` → 就地重采 +
sim-only reset）天然工作在拼接帧轴上；失败计数 EMA + 全局 bin + CDF 的课程平面在多 clip 语料
上沿用 v1 冻结口径。`for_play` 中 `start_at_timestep_zero_prob` 按 motion 源区分：单 clip /
单 clip 语料回到头帧，多文件语料无头帧概念，起始与换段重采全语料均匀（与 wbt multi-clip
预设同一决定）。

## BONES-SEED 管线

发布源：

- G1 retargeted clips：HuggingFace `bones-studio/seed`（license-gated）的 `g1.tar.gz`，
  36 列 CSV（cm / deg / 120 fps），14.2 万个；
- SMPL reference clips：HuggingFace `nvidia/GEAR-SONIC`（公开）的 `bones_seed_smpl.tar`
  七分卷（~31 GB），joblib PKL（`fps=50, pose_aa, smpl_joints`），13.1 万个。

两个归档都不可随机访问，`scripts/motion/download_bone_seed.py` 以"流式 + 提前终止"取子集：
先按序流 SMPL 分卷提取整 PKL 到字节预算，再流 `g1.tar.gz` 只保留 stem 已有 PKL 的 CSV，
直到配对数或扫描预算达标，最后剪掉未配对 PKL 并写 `manifest.json` 记录出处。默认预算
（SMPL 4.5 GiB + g1 扫描 14 GiB + 6000 对）传输约 10-12 GB、落盘约 3-4 GB；预算即上限，
成员乱序导致配对率是两个前缀的交集，调大预算可线性扩大语料。

`bones_seed_converter` 逐对转换：CSV 每 4 行取 1（120→30 Hz）→ lerp/slerp 重采样到 50 Hz →
名称重排 → MotrixSim 批量 FK 出 body 世界系位姿/速度 → 连同 SMPL 通道写成 schema v1 npz。
SMPL 根朝向 = y-up→z-up 旋转 ∘ `pose_aa[:, 0]` 轴角四元数 ∘ 固定标准化旋转（xyzw）。
`scripts/motion/convert_bones_seed.py` 提供目录级 CLI（跳过已存在、可 `--overwrite` /
`--max-clips` / `--workers N` 并行——每个 worker 各持一份模型与 FK buffer、内存随 worker 数
线性增长；`--filter-keywords` 默认启用 40 词过滤——道具场景与极限动作两类，列表移植自
gear_sonic，`--filter-keywords=` 关闭）。

## 使用

```bash
# 1. 拉子集（默认 ~/.cache/motrixlab/bones_seed/g1，~6000 对）
python scripts/motion/download_bone_seed.py

# 2. 转成 NPZ 语料（默认输出 ~/.cache/motrixlab/bones_seed_npz/g1）
python scripts/motion/convert_bones_seed.py

# 3. 训练（默认读同一缓存路径；自定义语料用 SONIC_MOTION_DIR，多路径用 : 分隔）
python scripts/train.py task=g1-sonic/motrix.fastsac
```

不设 `SONIC_MOTION_DIR` 时默认指向 converter 的缓存输出
`~/.cache/motrixlab/bones_seed_npz/g1`，与 `convert_bones_seed.py` 的默认 `--output`
一致——三步零额外参数即可串起。仓库不捆绑任何 SONIC 动作数据，集成测试使用合成语料
fixture。

## 发布约束

BONES-SEED / GEAR-SONIC 数据的授权状态与 smoke store 相同（见
[sonic-g1-task](./sonic-g1-task.md) 的发布约束）：子集仅供内部训练，公开再分发前需补齐
授权凭据。

## packed store 移除

语料路径稳定后，历史数据源整体删除：packed 层（`SonicPackedMotion` / `pack_sonic_store` /
`from_packed`、`SonicMotionCommandCfg.packed_store` / `packed_clip_limit` 字段、
`scripts/motion/pack_sonic_data.py` 与 `motrixlab_sonic_packer.py`、内置 packed smoke
数据）与单文件路径（`SonicMotionClip.from_motion`、`SonicMotionCommandCfg.motion_file`
及 `'???'` 互斥防护）。SONIC 数据面只剩 schema v1 npz 语料（converter 产出 +
`MotionLibrary` 消费），与 WBT/LAFAN 同构。

## 非目标

- 不做 per-clip 课程统计（沿用 v1 冻结口径）；
- 下载器不做归档随机访问 / 分片索引（tar 物理上不可行，前缀交集即子集语义）。

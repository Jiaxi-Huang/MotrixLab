# SONIC LAFAN4SONIC NPZ 语料设计

## 摘要

`motrix_envs.motion.lafan4sonic` 把 LAFAN1 动捕数据转成 SONIC 训练语料：每个 `.npz` 是一个
clip（MotrixLab motion schema v1 + `ext_smpl_joints` / `ext_smpl_root_quat` 参考通道），
与 BONES-SEED 语料同构、可在同一 `SONIC_MOTION_DIR` 下混用。数据面由两条独立通路汇成：
retarget 通路消费 LAFAN1_Retargeting_Dataset 的 G1 CSV，bvh→smpl 通路消费原始 LAFAN1 BVH。
本设计与仓库原有 LaFan 路径（`lafan_converter` + `download_lafan.py`，面向 WBT 单 clip 烘焙）
互不依赖、互不修改。

## 动机

SONIC 语料需要每个 clip 同时具备机器人参考（joint/body 数组）与 SMPL 人形参考
（`ext_smpl_*`）。BONES-SEED 两边都由上游发布；LAFAN 只有 BVH 与 retarget CSV，
SMPL 参考需要自行从 BVH 推导。UniLab 已验证过一条 bvh→smpl 流程
（移植自 jaraujo98/lafan_to_smplx，MIT），本设计按仓库四元数契约（xyzw、纯 numpy）
移植该流程，机器人侧则复用 BONES-SEED converter 的重采样与速度契约，保证两个语料可互换。

## 数据源

- G1 retarget CSV：HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset`（`g1/` 子目录，公开）；
  无表头 36 列（root pos 米制 + root quat **xyzw** + 29 关节弧度，`G1_CSV_JOINT_ORDER` 序），30 fps。
  四元数顺序经数据核验：站立帧第 4 分量 ≈1、其余 ≈0。
- 原始 LAFAN1 BVH：HuggingFace `johnny095212/lafan1`（Ubisoft LaForge 发布的镜像，公开）；
  Z/Y/X 欧拉通道，root 另有 3 平移通道，30 fps。与 G1 CSV 按文件 stem 一一配对、帧数一致。
- `human_joints_info.pkl`：gear_sonic 的 SMPL 55 关节元数据（`J` 休息位 + `parents_list`），
  从 GR00T-WholeBodyControl 公开仓库下载；加载走 cold-path 的 torch 懒导入。

`scripts/motion/download_lafan4sonic.py` 拉齐三者到 `~/.cache/motrixlab/lafan4sonic/`
（`g1/`、`bvh/`、`human_joints_info.pkl` + `manifest.json`）：先列 G1 集合、只下载有 CSV 对应的
BVH，逐对校验 30 Hz 与帧数一致，manifest 记录出处。LAFAN1 为 CC BY-NC-ND 4.0，
仅供内部训练，公开再分发前需补齐授权凭据。

## 两条通路

### retarget（`retarget.py`）

CSV 按名重排到模型关节序 → 30→50 Hz 重采样（pos/dof lerp、root quat slerp，半开目标网格）
→ MotrixSim 批量 FK 出 body 世界系位姿 → 速度按 BONES-SEED 契约构造：
`joint_vel` 前向差分（末帧复制），body 线/角速度由 FK 位姿差分
（中心差分 + 最短弧四元数差分）+ σ=2 帧高斯平滑。与旧 `lafan_converter` 在
joint_pos / body_pos_w 上逐位一致（已用同一 CSV 对拍，最大差 0）；差异只在速度契约
（旧路径用 `np.gradient` + 引擎速度输出，BONES-SEED 契约改为前向差分 + 位姿差分平滑）。

### bvh→smpl（`bvh.py` + `smpl.py`）

BVH 解析出骨架与局部四元数（Z-Y-X 欧拉 → xyzw）→ 局部四元 slerp 到与机器人侧相同的
50 Hz 目标网格 → 链式求全局四元数 → 逐骨 rest-orientation 修正后提取 SMPL pose 参数
（root + 21 body 旋转矢量，lafan_to_smplx 映射）→ 以 `human_joints_info.pkl` 休息位做
55 关节 FK → 取 24 关节（0–21 + 双手 39/54）→ 套 SONIC SMPL 帧契约：
y-up→z-up 旋转、LAFAN 特有的 +90° z 轴航向修正（BVH 世界系与 retarget G1 世界系相差
90°，修正必须同时作用于 joints 与 root quat，否则破坏 `root⁻¹·joints` 特征）、
`canonical` 固定旋转、根关节位置扣除（joints 为 root-local）。BVH root 平移被丢弃
（SMPL 通道不携带世界位置，机器人侧拥有世界轨迹）。

移植已与 UniLab 参考实现逐值对拍：rotvec 差 ~3e-7、joints 差 ~2.8e-7 m、
root quat 差 ~1e-7（1−dot）。零旋转 BVH 的根四元数恰为 canonical 旋转
`[-0.5,-0.5,-0.5,0.5]`（航向修正 ∘ y-up→z-up ∘ Hips 偏置相消），由测试锁定。

## 汇合与使用

`converter.py` 按 stem 配对、fail-fast 校验帧率/帧数，两通路重采样到同一目标网格后写单 clip
schema v1 npz（输出键集与 BONES-SEED converter 完全一致，`reference_body_name` 为 `pelvis`）。
`scripts/motion/convert_lafan4sonic.py` 提供目录级 CLI（`--overwrite` / `--max-clips` /
`--model-file`，默认读下载缓存、写 `~/.cache/motrixlab/lafan4sonic_npz`；40 clip 全量转换
为单进程顺序执行，无 `--workers`）。训练时设
`SONIC_MOTION_DIR=~/.cache/motrixlab/lafan4sonic_npz`，或以 `:` 拼接与 BONES-SEED 语料混用。

```bash
# 1. 拉原始对（约 40 对 / 26 万源帧，~300 MB）
python scripts/motion/download_lafan4sonic.py

# 2. 转成语料
python scripts/motion/convert_lafan4sonic.py

# 3. 训练
SONIC_MOTION_DIR=~/.cache/motrixlab/lafan4sonic_npz \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

## 非目标

- 不改原有 LaFan 路径（`lafan_converter` / `download_lafan.py`）的任何行为；
- 不做 clip 裁剪、课程统计或关键词过滤（BONES-SEED 的过滤列表针对其道具/极限动作场景，
  LAFAN 全集即训练集——与 UniLab lafan 实验同口径）；
- 不引入 scipy / smplx 依赖：移植全部用 numpy + `motrix_env_core.math.quaternion`，
  四元数 xyzw 端到端统一。

# 计划：多 Motion Clip 训练基础设施

基于 [多 Motion Clip 训练基础设施设计](../design/motion-multi-clip-infra.md) 的分阶段实施计划。
约束：不改 `motrix_env_core`；v1 冻结现有 sampling 设定；遵循现有 motion npz schema（一文件一 clip）。

## P0 数据平面：`MotionLibrary`

- [x] `motrix_envs/motion/` 新增 `MotionLibrary`：多文件有序加载、任务序重排、`ext_` 通道按需装载、
      拼接为全局帧轴、产出 `frame_clip_end`/`clip_lengths`/`clip_offsets`
- [x] 跨文件契约校验：fps 匹配 ctrl、joint/body 名称集合一致、通道 shape 一致、finite、单位四元数
- [x] `WbtMotionClip` 增加 `frame_clip_end` 字段；单文件构建路径产出常量末帧（行为不变）
- [x] 单测：多文件 fixture（clip 边界、重排、通道装载、负向校验：缺通道/fps 不符/名称集合不一致）
- 验收：单 clip 路径与现行为等价；`g1-wbt-dance` 构建 + 数步正常

## P1 时间线平面：`WbtMotionCommand` 原地泛化

- [x] `advance()` 边界测试改为逐帧 `frame_clip_end`（末帧跟踪一步后越界）；hold/wrap 语义不变
- [x] wrap 重采目标为全语料 CDF；起帧抽样 helper 保持模块级 `@njit`
- [x] 多 clip fixture 下的换段/钳制行为测试（到尾 sim-only 重置、不结束回合、边界 bin 不越界）
- 验收：**同 seed 抽样序列与 main 逐位一致**（WBT 回归门）

## P2 通道声明接线

- [x] `WbtMotionCommandCfg` 增加多文件 motion 源与 `ext_` 通道声明字段；单文件路径保持默认
- [x] 文档示例：声明 SMPL 通道的多文件配置写法（见 design 文档"通道声明"）
- 验收：默认配置行为不变；声明通道后 library 正确装载并随命令进入 kernel 输入

## P3 收尾

- [x] 测试补齐与 ruff/prek 通过
- [x] 更新 `wiki/design/motion-multi-clip-infra.md` 与本文档状态
- [ ] 实现完成后按目录规范删除本 plan 并更新 `wiki/plan/index.md`（issue #78 要求设计与计划随首个 PR
      入库，故保留至功能完全落地后的后续 PR 再删除）

## 后续独立项（不在本计划）

- sampler 组件化与统计口径演进（rate/访问归一、clip-local bin、per-clip 等权、集中度上限、可观测性）
- SONIC 类任务作为第一个新消费者接入

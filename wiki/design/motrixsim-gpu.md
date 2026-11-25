# MotrixSim GPU 仿真设计

在 `e98888a5` 基线上，为已有 MotrixSim 后端增加可选的 NVIDIA CUDA 仿真能力。目标是让同一个已注册任务使用原有训练配方在 CPU 或 CUDA 上运行。以下是待实现的行为契约。 [需求 #183](https://github.com/Motphys/MotrixLab/issues/183)

## 设备选择与安装

基线的 `registry.make(..., sim=...)` 选择仿真后端。设计中保留这个入口，并允许在构造环境时单独选择仿真设备。显式指定的设备优先于环境配置，内置任务继续默认使用 CPU。训练、回放和环境可视化应采用一致的仿真设备选项；推理与 learner 的设备仍由 provider 决定。回放优先采用用户覆盖值，其次采用运行记录中的实际仿真设备。旧运行记录缺少该信息时，使用环境默认值。 [环境注册入口](../../motrix_env_core/src/motrix_env_core/registry.py)

GPU SDK 作为 MotrixSim 插件的可选依赖提供。仅安装 CPU 依赖时，应能导入插件并运行 CPU 任务；显式请求 CUDA 时，应能区分缺少依赖、设备不可用和场景能力不足。版本、平台标记和安装选项在验证安装组合后写入用户指南。 [插件依赖声明](../../motrix_env_motrixsim/pyproject.toml)

## 仿真与环境生命周期

- Direct 和 Manager 环境共用后端边界。动作、观测、索引和运行时状态在所选环境设备上可直接使用。查询的通道顺序、单位、形状以及每行数值异常的报告方式遵循现有契约。插件负责让 Torch 写入的控制量对下一次仿真可见，并让后续 Torch 读取看到对应物理步的结果。 [SimBackend](../../motrix_env_core/src/motrix_env_core/sim/backend.py)
- 局部 reset 只影响指定环境行。目标行的物理状态、接触与传感器读数应对应复位后的场景；其他行保持原状态，直到下一次物理步。回合计数与命令状态按回合级或仅仿真级 reset 分别处理。自动 reset 仍向训练端提供结束前的终态观测。 [Manager 运行时](manager/runtime.md) · [终态观测](array-env-final-observation.md)
- 预览覆盖全部内置 Direct 和 Manager 任务。设备适配应保留任务名、观测、奖励和随机化语义。接触、参数写入和 Raycast Camera 的支持程度需要逐项验证。 [能力调查](../research/motrixsim-gpu-backend-review.md)

## 训练与分片

单进程 SKRL、RSLRL 和同步 FastSAC 使用单一仿真设备。异步 FastSAC 中，每个 collector 创建自己的环境分片；普通 CUDA 请求与显式卡号具有不同的放置语义。

| 仿真设备请求 | 预期放置 |
| --- | --- |
| CPU | 沿用现有 CPU affinity、NUMA 内存放置和样本传输规则。 |
| CUDA，未指定卡号 | 每个 collector 在所属 learner 对应的可见 GPU 上独立推进其分片。 |
| CUDA，显式指定卡号 | 所有相关分片固定到指定的可见卡，运行记录保存实际放置。 |

预览允许通过现有主机通道交接 CUDA 样本与终态观测，传输成本单独测量。跨卡传输优化、IPC 重构、分布式 PPO 和部署后端留给独立议题。 [FastSAC 异步训练设计](fastsac-async-heterogeneous-trainer.md) · [实施与验收](../plan/motrixsim-gpu.md)

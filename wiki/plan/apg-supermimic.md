# APG 与 SuperMimic 实施计划

## 摘要

按 [APG 设计](../design/wm-bptt-trainer.md) 实现 GenCoord motion 的同步参考路径。
同步路径作为质量参考；单 collector 异步采样已接入，GPU 加速效果与训练质量需单独验收。多 GPU 不在本轮范围。

## main 基线结构适配 TODO

- [x] 从命名 observation layout 统一 collector、learner、play 输入与加载校验。
- [x] 复用 main 的可微 history 操作，保持 checkpoint 和 ONNX 输入兼容。
- [x] 验证布局换序、采集/回放、梯度、旧 checkpoint 与 ONNX parity。
- [x] 校正设计与评估文档中的基线、动作量程和运行时边界。
- [ ] 独立完成 Torch 支持基线决策、依赖升级及各 provider 回归（不在此次结构适配范围）。


### 结构适配验收

所有 pytest 命令使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。

- `python -m pytest motrix_rl_builtin/tests/test_apg.py motrix_rl_builtin/tests/test_apg_observations.py motrix_env_core/tests/test_manager_modeling.py motrix_rl/tests/test_replay.py -m 'not integration and not slow' -q`：
  临时 Torch 2.11 环境 76 passed、16 deselected；锁定 Torch 2.7 环境
  74 passed、2 failed、16 deselected。两项失败仍为缺少 `torch.optim.Muon`
  与 compiled modeling 的 dataclass 构造限制，未通过 skip 隐藏。
- `python -m pytest motrix_rl_builtin/tests/test_apg.py -k 'async_collector_order or onnx_history' -q`：
  锁定环境 7 passed，覆盖两种 term 顺序、正常/失败/提前停止采集及 ONNX round-trip。
- 一次性 CPU smoke：MotrixSim 1.0、SuperMimic 四个环境、五次采集，
  Torch 2.11 下 Muon world-model/actor 与 critic 完整更新的指标有限，
  training-state checkpoint 加载成功。脚本与日志位于
  `/tmp/apg-main-adaptation-smoke.py`、`/tmp/apg-main-adaptation-smoke.log`。
- 本次 Python 文件通过项目固定版本 Ruff check 与 format check，
  `git diff --check` 通过。未进行 GPU 训练或吞吐验收。


## TODO

- [x] 北京 5090 隔离环境复现当前配置长训基线及定期 40-clip 评估。
- [x] 对照真实/想象观测噪声，验证独立 RNG、梯度和 checkpoint 恢复。
- [x] 记录预算内实测结果与失败片段：未达到 40/40，不提供推测的达标时间。

本轮服务器实验使用独立目录
`huangjiaxi@js5.blockelite.cn:10600:/mnt/gfs/home/huangjiaxi/apg-lafan1-20261116`，
产物在 `/mnt/gfs/home/huangjiaxi/data/apg-lafan1-20261116`。
固定版本的 `baseline-fused`、`noise-fused`、`wm10-fused` 均从零开始，预算各 100000
步，每 8192 步及最终 checkpoint 做同协议 4 进程评估；首次 40/40 后停止对应实验。
后两组分别仅开启想象噪声、仅将 WM 更新次数改为 10。网络、batch、horizon 保持当前
task 值。输入 40 个 NPZ 的 SHA256 与本机一致。

同时在本机运行独立新训 `local-wm10-fused`（同样 seed=1、WM updates=10），
checkpoint 发布到服务器，由同一 4 进程评估队列处理。初步端到端吞吐本机约
13000 env-steps/s，北京约 5500–7000；本机 i5-14600KF 与服务器 Xeon 8380
的 CPU 仿真/调度性能不同，不能仅按 GPU 型号推断训练耗时。

固定里程碑评估（本轮已结束，未达到 40/40）：

| 实验 | 步数 | 启动至 checkpoint（分钟） | 完整片段成功数 |
| --- | ---: | ---: | ---: |
| 本机 WM10 | 8192 | 21.40 | 26/40 |
| 北京 imagination noise | 8192 | 47.91 | 17/40 |
| 北京 WM4 | 8192 | 48.90 | 18/40 |
| 北京 WM10 | 8192 | 54.06 | 24/40 |
| 本机 WM10 | 16384 | 42.63 | 26/40 |
| 本机 WM10 | 24576 | 63.85 | 31/40 |
| 北京 imagination noise | 16384 | 90.10 | 28/40 |
| 北京 WM4 | 16384 | 92.46 | 25/40 |
| 北京 WM10 | 16384 | 102.64 | 29/40 |
| 本机 WM10 | 32768 | 85.00 | 35/40 |
| 本机 WM10 | 40960 | 106.17 | 28/40 |
| 北京 imagination noise | 24576 | 132.39 | 29/40 |
| 北京 WM4 | 24576 | 135.95 | 28/40 |
| 本机 WM10 | 49152 | 127.31 | 34/40 |
| 本机 WM10 | 57344 | 148.42 | 36/40 |
| 本机 WM10 | 65536 | 169.52 | 34/40 |
| 本机 WM10 | 73728 | 190.62 | 35/40 |
| 本机 WM10 | 81920 | 211.70 | 35/40 |
| 本机 WM10 | 90112 | 232.82 | 34/40 |
| 本机 WM10 | 98304 | 253.93 | 35/40 |
| 本机 WM10 | 100000 | 258.32 | 36/40 |
| 北京 WM10 | 24576 | 151.28 | 33/40 |
| 北京 WM10 | 32768 | 198.75 | 33/40 |
| 北京 WM10 | 40960 | 246.10 | 31/40 |

北京 WM4 baseline/noise 在三轮固定评估后主动停止，保留其训练日志与模型。
WM10 三轮均优于同机两组，后续训练预算集中到本机与北京 WM10；停止的两组
属于提前终止的对照，不据此声称它们在完整 100000 步预算内无法达到目标。
任务默认 `world_model.num_updates` 已设为 10，与本轮主候选的有效配置一致；
运行中实验仍使用启动时冻结的配置，不加载后续文件改动。想象噪声保持关闭。
北京 WM10 在第五轮评估后主动停止：最佳 33/40、最新 31/40，耗时已超过本机
候选；不推断其在完整 100000 步预算中的最终能力。本机已正常完成预算，退出码
为 0；服务器评估队列也已完成，所有本任务训练/评估容器均已退出。

本机最终预算为 100000 个向量环境 step（204800000 env-steps），checkpoint
记录 24985 次 learner 更新轮次，每轮 10 次 WM 更新。步数口径遵循 FastSAC，
不能直接与 Uni-SuperMimic 的 learner iteration 数字比较。13 个固定 checkpoint
均做完整 40-clip 评估；最佳与最终均为 36/40，最早的最佳点为 57344 步、148.42
分钟。各片段都曾在某个 checkpoint 上通过，但没有一个 checkpoint 同时 40/40。

最终模型失败片段为 `fallAndGetUp1_subject4`（1273/8410 帧）、
`fallAndGetUp2_subject2`（2153/8195）、`fight1_subject2`（7586/12244）、
`fightAndSports1_subject1`（2244/12257）。最终模型 SHA256：
`a278be74b15c3a09a33a2c326b2d15ad857e8e75d14cb849b075740de0fae5e3`。

本机 run：`/tmp/apg-local-wm10-fused/runs/g1-wbt-supermimic/motrix/torch/apg/26-11-16_15-40-23-675265`。
最早最佳模型为 `checkpoints/model_0057344.pt`，SHA256：
`d0a4efed0cf2c9cf69ee8ff947bdf291bb8fb3d9853e10a621ea6967eaa034aa`。
完整报告、曲线 CSV、所有逐片段原始结果、数据与源码 manifest 位于
`/tmp/apg-5090-results/`，同一证据 archive 保留在服务器产物目录。

本机 16384 步相较 8192 步，`fallAndGetUp1_subject4/5` 已完整成功，但部分跑跳
片段退化，总数未提升；不据此声称收敛或推算达到 40/40 的时间。

16384 步模型另做只读积分诊断：64 lanes、2048 步、131072 个真实 transition，
使用 FinalState 保留终止前状态。将真实 successor 速度增量输入解析积分器后，
root position norm 误差均值 0.068 mm、p99 0.451 mm；关节位置 RMS 误差均值
0.00136 rad、p99 0.00681 rad。当前 learned model 对应均值为 0.160 mm 和
0.00213 rad。这是采集分布上的一步诊断，不是 BPTT 多步误差上界，也不足以
证明积分器是困难动作失败的主因；未据此改变模型结构或重启实验。

计时包括各自训练启动及编译，不含此前服务器环境安装；服务器使用独立冷编译
缓存，本机可复用编译缓存。评估耗时单列在服务器 `curve-fused.jsonl`，不计入
checkpoint 训练耗时。8192 步采样间隔只能确定首次观测达标的时间上界。

性能预验收中发现服务器主机调度开销较大，原实现约 4100 env-steps/s，并非换 5090
就自然更快。已实现固定 Manager 数值 program 编译、整段 WM loss fullgraph 编译，
以及消除动作环形缓存的 Python 整数读取/GPU 同步。独立数值/梯度对照：batch 1024
Manager 前反向约 10.8→2.6 ms；batch 128 WM 前反向相对逐步编译约 34.5→6.4 ms，
梯度相对 L2 误差 3.5e-6。这些是局部 benchmark，不替代端到端训练耗时。
完整 32 步 BPTT 新训通过，编译 RNG 推进与 17×33 次观测调用精确一致；快速回归
773 passed、1 skipped。预验收 `baseline/noise` 与 `baseline-fast/noise-fast` 已在首个
评估点之前主动停止，其权重未用于后续训练；日志保留作性能诊断证据。

- [x] 冻结参考来源、输入输出、任务配方与有意差异。
- [x] 实现 G1 WBT 族 SuperMimic 任务（`g1/supermimic.py`）与可微数学。
- [x] 补齐 modeling 的 replay 起点恢复、可变 batch/device 与梯度隔离。
- [x] 实现 APG 配置、provider、网络、转换装配、多步 WM 与 BPTT/critic。
- [x] 接入同步训练、checkpoint/resume、play 与 ONNX。
- [x] CPU 契约/数值/梯度与现有 WBT 回归、lint；记录硬件限制。


## 后续验收

- [ ] 固定参考实现的完整前向/梯度专项 parity，补齐 soft survival、随机化和课程差异。
- [ ] 在具备 GPU 的环境执行固定 motion、seed 和预算的训练质量/性能验收。

当前完成的是可运行同步参考实现，不将小规模 smoke 或有限 loss 视为训练收敛证明。


## 已执行验收（2026-11-12）

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest motrix_env_core/tests motrix_rl/tests
  motrix_rl_builtin/tests tests -m "not slow and not integration" -q`：765 passed，1 skipped。
- APG、modeling、SuperMimic term、既有 WBT 数学/command 和 task 配置定向集合：184 passed，
  包含 ONNX Runtime parity 集成测试。
- CPU 真实 MotrixSim：2 lanes、小网络、WM horizon=2、policy horizon=3、history=3，
  同步训练 4 步、恢复至 6 步并导出 ONNX 成功。另通过实际 `mxlab train` 入口运行同样的小配置；
  使用 `hydra.job.chdir=true hydra.run.dir=/tmp/apg-cli-smoke` 将输出放到可写目录。
- ONNX parity 32 samples，最大绝对误差低于 `1e-6`；只证明已训练小模型的输入输出数值一致。
- 本机 NVIDIA driver 不可用，未执行 GPU、长训收敛、吞吐或参考实现的完整训练质量对比。


## Uni-SuperMimic 对照优化

- [x] 输出增量、critic 输入/目标归一化与小策略初始化。
- [x] 任务声明可微终止，真实硬终止与 modeling 默认路径不变。
- [x] 网络 bf16、编译、fused Adam 与性能实测。
- [x] 修复 MotrixSim post-step link 查询 freshness，并通过真实 FK 一致性回归。
- [x] Muon/Adam 与采样/更新比例的独立新训对照。
- [x] 全 LAFAN1 从头训练，记录配置、质量、吞吐与剩余差异。

### 本机验收（2026-11-16）

来源为相邻 `Uni-SuperMimic` 的 `e8da48f`，主要对照 `learner.py`、`networks.py`、
`optim.py`。保留既定 Manager 单源、FinalState、硬终止/截断、console/live metrics/
TensorBoard 和 checkpoint 契约；具体实现差异见设计中的单 GPU 对照章节。

所有候选训练均从随机初始化开始，未加载旧模型。使用 RTX 5060 Ti 16 GiB、全 40
个 LAFAN1 clip、2048 env、batch 1024，actor 1024/512/256、WM 1024/1024/1024；
history=10、WM horizon=8、policy horizon=32 未改。最终 recipe 已写入 task YAML。
混合精度限于网络；没有降低任务奖励要求、放宽真实终止、过滤困难 clip 或跳过坏梯度。

发现并修复原生 post-step link 查询缓存陈旧：刷新前 body position 误差约 0.025 m、
角速度约 9.2 rad/s；刷新后与相同 root/q/qd 的可微 FK 比较，各项最大误差小于
1e-5。使用正式 `forward_kinematic` 接口，不推进仿真时间。

独立评估：seed=2026、128 env、500 步、均匀抽取 motion frame、关闭观测/reset 噪声，
同一修复后的真实环境及硬终止；旧 checkpoint 仅用于评估。各组均覆盖 40 个 clip。
首 episode 截断均值为每个环境首次 episode 长度与 500 的较小值，避免仅统计已经
结束的短 episode 导致偏差；不是完整长序列或多 seed 收敛验收。

| 指标 | 原 trainer 4096 步 | 新 Muon 2048 步 | 新 Muon 4096 步 |
| --- | ---: | ---: | ---: |
| 从启动至 checkpoint（分钟） | 9.41 | 7.32 | 15.03 |
| 平均逐步 reward | 0.03565 | 0.04336 | 0.05651 |
| 首 episode 截断均值（上限 500） | 49.68 | 286.98 | 456.57 |
| 存活满 500 步的环境数 | 1/128 | 59/128 | 112/128 |
| root height 平均绝对误差（m） | 0.07315 | 0.03117 | 0.02485 |
| 任务 root tilt error 均值 | 0.09916 | 0.02499 | 0.02636 |

Muon 新训的 1000–4000 步区间吞吐中位数为 9438 env-steps/s，显存约 7.2 GiB。
旧采集/更新比例的编译版约 24k env-steps/s，但独立质量提升有限；最终配方增加更新
密度，优先考虑达到有效策略所需时间。上表为实际 wall time，包含启动；运行期间
有评估共享 GPU，不能当作严格隔离的性能 benchmark。Muon、更新密度和学习率组合
实验未逐项消融，不将全部收益归因于某一个因素。

验收 run 位于
`/tmp/apg-uni-muon-fresh/runs/g1-wbt-supermimic/motrix/torch/apg/26-11-16_13-05-35-205074`，
本次预算 16384 步；默认 recipe 为 100000 步。运行入口为 `mxlab train
task=g1-wbt-supermimic/motrix.apg`，本次加 `hydra.job.chdir=true
hydra.run.dir=/tmp/apg-uni-muon-fresh algo.device=cuda
algo.trainer.num_learning_iterations=16384`，CPU 线程为 4。
独立评估数据为 `/tmp/apg-muon-2048-evaluation.json` 和
`/tmp/apg-muon-4096-evaluation.json`，脚本 `/tmp/apg_eval_fresh.py`。
该长训启动于最后一次 critic batch 合并前；后者另外通过目标/梯度/参数更新数值
对照及 GPU 短训验收，尚未单独计入上述长期质量和吞吐。

验证结果：快速回归 857 passed、4 skipped；真实 post-step FK 集成回归通过；
Muon 状态恢复后的下一次更新一致；编译/eager actor 输出和输入梯度通过数值对照；
最终代码 GPU 128 步新训完成且指标有限、最终 checkpoint 事件不重复。
4096 步实际模型 ONNX 导出通过 32 samples parity，最大绝对误差 1.43e-6，产物
`/tmp/apg-uni-policy.onnx`。尚未验收全数据长序列收敛、多 seed 稳定性或完整参考 parity。

### 最终 checkpoint 的完整 LAFAN1 评估

上述新训已正常完成 16384 步（33,554,432 env-steps），耗时 60.50 分钟。
使用最终 `model_0016384.pt`，4 个独立评估进程各负责 10 个 clip，105.6 秒完成全量
评估，结果 **27/40（67.5%）**。不选择中间 checkpoint，不重复失败 episode。

评估 seed=2026，标准 play 配置关闭观测和 reset 噪声；显式将每个 lane 初始化至
各自 clip 首帧，沿用任务原有高度/倾斜硬终止。只有完整到达 clip 末帧、触发
clip truncation 且无 hard termination 才成功；中途失败后的自动 reset 不计入结果。
这是完整片段存活成功率，不额外保证某一 tracking error 阈值。

13 个失败 clip 为 `dance1_subject1`、`fallAndGetUp1_subject1/4/5`、
`fallAndGetUp2_subject2/3`、`fallAndGetUp3_subject1`、`fight1_subject2/5`、
`jumps1_subject2`、`run1_subject2`、`run2_subject1`、`walk2_subject1`。
尚不能视为完整 LAFAN1 已收敛，尤其起身动作仍有明显失败。

复现脚本和产物位于 `/tmp/apg-lafan1-full-eval/`：`evaluate.py`、`results.json`、
`results.csv`、`report.md`、`evaluation.log` 及包含全部输入文件 SHA256 的
`corpus_manifest.json`。checkpoint SHA256 为
`475b152296009233185a509fb19120385609c45c63e28aa9e9897c040041233c`。

## 配置与调度对齐 TODO

- [x] 统一 agent/trainer 配置归属和 AMP/replay 字段命名。
- [x] 实现精确 UTD、窗口 warmup、学习率衰减和 EMA checkpoint。
- [x] 验证调度、恢复、推理 artifact 与 CPU 真实短训。

### 有界异步采样与文件职责审查

- [x] 提交已验证的 APG/SuperMimic 基线。
- [x] 共用同步采样实现，接入公共 host ring 和原子 inference bundle，增加有界 spawn collector。
- [x] 保持序列/UTD/normalizer/checkpoint 语义，验证异常退出与 resume。
- [x] 审查 APG 新增文件职责，记录 rand.py 的真实使用方与保留理由。
- [x] 实测同步/异步相同配置的短跑速度及 GPU checkpoint/退出行为。
- [ ] 完成同步/异步从零长训的成功率和达标时间对照，不以短跑代替收敛验收。

异步采样验收记录（2026-11-17）：

- 基线提交：`45c8495`；保持原任务超参数、learner 配额和目标窗口。
- 快速契约测试：462 passed、1 skipped；APG 专项（含 ONNX 与 spawn）21 passed。
- 真实 CPU 短跑：20 步为 6 个 learner rounds、余数 4；恢复到 30 步为 9 rounds、余数 3。
  独立 collector RNG 的 checkpoint/resume 也已执行；日志在 `/tmp/apg-async-rng-{smoke,resume}.log`。
- 代码与文件职责见设计文档；`schedule.py` 合并到 collector，`rand.py` 经使用方审查保留。
- 北京隔离速度对照：`/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/{sync,async}`，
  GPU 2、CPU 8–15、2048 envs、512 步；同一容器顺序执行，首次编译单独计入启动成本。
  短跑不提供同等成功率或收敛时间结论；完整质量比较 TODO 保留。

同卡短跑结果（512 步，取 step 128–512 的全部 13 个等长窗口，以总样本/总耗时汇总）：

| 路径 | env-steps/s | collect ms/步 | learn ms/步 |
| --- | ---: | ---: | ---: |
| 同步 | 4090.1 | 115.2 | 375.7 |
| 单 collector 异步 | 4778.8 | 44.9 | 331.4 |

稳态吞吐提升 **16.8%**。这里的异步 collect 是主进程等待/接收/归一化时间；
两列阶段耗时不包含旧短跑版本的权重发布，不能将它们相加作为完整 wall time。
修正版把 publish 计入 learner 总时间，并单列 publish/load/collector 实际采样时间。
首轮同步冷编译、后续异步暖缓存，所以不能以两次启动到结束总时长计算加速比。
一个中间异步窗口曾降至 2390 env-steps/s，结果保留该窗口，未挑选最佳片段。

速度门槛通过后已启动 `8ee51d5` 的独立质量对照：
`quality-sync`（GPU 2、CPU 8–15）与 `quality-async`（GPU 3、CPU 16–23），
均 seed=1、从零训练、8192 步、每 2048 步保存，结束后运行原有 4 进程 40-clip
首帧至末帧评估。产物根目录同上；状态见 `quality-*.train.exit`、`quality-*.eval.exit`，
结果见 `eval-{sync,async}/results.json`。这两组长训尚未完成，**没有同等成功率更快达标的结论**。

连续参数绑定补充优化使用 `src-batched` 独立快照在 GPU 4 做 256 步验证，
不覆写质量对照的 `src-final`。该优化复用公共 `bind_flat_params` 与 receiver 的
`flat_params` 接口，将逐参数 H2D 加载合并为一次连续拷贝，不改变 actor/normalizer 的版本契约。


补充验证：连续参数加载版本在 GPU 4 完成 256 步，step 128–256 汇总为
5241.3 env-steps/s；publish 平均 0.93 ms/步、collector weight load 0.72 ms/步。
该卡/CPU 配额与上面的同卡对照不同，且长训同时运行，因此不把它作为额外提速百分比。
GPU 退出时观察到 Queue semaphore 清理告警后，元数据通道改为单向 Pipe；最终版本
在 GPU 4 完成 32 步、checkpoint 和正常退出（退出码 0），日志无 traceback/semaphore 告警。
日志为 `async-pipe.log`。Pipe 版本 21 项 APG 测试通过，pre-commit 全部通过；
额外 CPU 验证 UTD=2.5、publish interval=3 得到预期的 12 rounds/余数 1，模型张量全部有限。

### 全量服务器配置与 learner 调度分析

- [x] 增加显式服务器 experiment，核对 Uni-SuperMimic 实际配置；本机默认缩小网络配置保留。
- [x] 使用 CPU/CUDA profiler 捕获当前实现的更新窗口，记录 kernel/同步热点、峰值显存。
- [x] 根据证据融合 learner FK 小算子，验证数值/梯度；GPU/CPU 使用同 NUMA cpuset。
- [x] 对本机规模和服务器全量规模执行 GPU 验收，记录吞吐、资源利用率及限制。

CUDA baseline（缩小版、一次完整 learner round）：38372 个 kernel，中位 kernel 1.184 μs，
kernel duration 总和 88.75 ms；36 次 cudaStreamSynchronize 合计 0.212 ms。
profiled WM CPU span 435 ms、policy span 1078 ms，profiler 总窗口 2.34 s，峰值 allocated 4.92 GiB。
profiler 有开销，最终速度必须用不带 profiler 的同规模运行比较。


2026-11-17 CUDA 对照（每项均捕获第 9 次 learner round，单 GPU；Profiler 时间不作吞吐）：

| 规模 / 路径 | kernel 数 | kernel duration 总和 ms | allocated 峰值 GiB | policy CPU span ms |
| --- | ---: | ---: | ---: | ---: |
| 缩小 / baseline | 38372 | 88.75 | 4.92 | 1078 |
| 缩小 / FK 融合 | 31664 | 71.18 | 4.87 | 899 |
| 全量 / baseline | 40879 | 288.34 | 21.63 | 1092 |
| 全量 / FK 融合 | 32286 | 247.35 | 21.42 | 932 |

baseline 是 `29d811b`，融合版只替换共享 `math/kinematics.py`；完整配置维持
history=10、WM window=8、policy horizon=32、UTD=0.4、WM updates=10。
两种规模各完成 256 步 GPU 训练（退出码 0）。全量 batch/replay/网络无需缩减即可运行。
真实 CUDA Inductor 的 float64 FK 位姿/速度和全部输入梯度对照通过；CPU aot_eager 同契约测试保留。
原始 trace/summary：服务器 `/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/{profile-local,profile-full,fk-local,fk-full}/`。
Profiler 退出仍可能出现第三方 resource_tracker semaphore 清理告警，不影响 trace/退出码。

资源分析：既有长训低频采样中，async learner 约 0.85 个 CPU 核、collector 约 2 个核，
collector 主线程约 46% 样本在等待 ring；无 CPU quota/throttling，iowait≈0。
本次 CUDA trace 的 36 次 stream synchronize 仅约 0.2 ms，主要问题是数万次微秒级 kernel
与 autograd host 调度。增加 collector/ring 容量不能解决 learner 关键路径，反而增加策略滞后。
单卡单 learner + 单 collector 保留；不同实验使用独立 GPU 与不重叠的同 NUMA cpuset。
本机 GPU 需要提权执行环境才能访问；已确认 RTX 5060 Ti 16 GiB，并另行执行本机对照。
服务器缩小规模对照不能当成本机成绩。


同卡不带 profiler 对照使用 256 collector steps，取 128–256 的全部 9 个等长窗口，
按 step 去重 console 结束时的重复输出，SPS 用调和平均（总样本 / 总时间）。
服务器缩小版 GPU 4 / CPU 40–47：baseline 4907.8、融合 5001.5 env-steps/s（+1.9%）；
learn 336.6→306.7 ms/步（-8.9%），collect 45.0→46.3 ms/步。
CPU 约 2.7 核 / 8 核，console GPU 窗口值均值约 13.6%→12.8%。
这些短跑只证明执行性能，不证明达到同等成功率所需时间。


已提交执行优化与配置：`909344b`。全量从零长训使用等价 runtime 快照
（`29d811b` 源码 + `909344b` 的 FK/experiment 只读覆盖），容器
`apg-full-909344b-20261117`，GPU 5 / CPU 48–55，Torch threads=4，seed=1。
预算 30010 collector steps（12000 learner rounds），checkpoint=5000，logging=125。
产物根 `/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/`：
`full-909344b.log`、`full-909344b-run/runs/`、`full-909344b.exit`；
结束后自动对最终 `model_0030010.pt` 做原有四进程、首帧至末帧、无重试的全部 40 clip 评估，
结果 `eval-full-909344b/results.json`。这不是恢复既有 checkpoint 的续训。

额外单线程试验：融合全量 GPU 5 / CPU 48–55，256 steps，稳定 9 窗口
8650.1 env-steps/s，collect=70.2 ms、learn=342.5 ms、CPU≈3.7 核。
它与主同卡对照的 CPU/GPU 配额不同，只作为调度探索，不能当严格的线程加速比；
未见采用单线程的依据，正式配置保留 4 线程。

验收：core 快速测试 258 passed；builtin + FK 快速测试 125 passed、1 skipped；
Manager modeling/task/CLI 配置测试 156 passed。CUDA Inductor 输出/梯度对照通过，
Hydra 全量 experiment 组合检查通过，ruff/pre-commit 通过。


本机实际对照（i5-14600KF + RTX 5060 Ti 16 GiB，默认缩小网络，async，256 steps）：
baseline 10828.9 → 融合 12242.3 env-steps/s（+13.1%），collect 21.0→20.6 ms/步，
learn 167.9→146.4 ms/步（-12.8%）；console GPU 窗口值均值 47.7%→52.4%。
本机为桌面共享 GPU，GPU/系统 CPU 数值包含其他桌面进程，不能当作训练进程专属利用率。
两个运行退出码均为 0；`/tmp/apg-local-{base,fused}.log`，产物同名目录，
每个 run 含最终 checkpoint 与 TensorBoard。baseline 临时加载 HEAD=29d811b 的 FK，
其余 runtime/配置相同；实际工作区保持融合版，没有切换或覆盖运行中的源码。
本机首个日志窗口包含编译，baseline 150 s、融合 85 s，缓存状态不同，不据此计算启动加速比。

服务器是双路 Xeon 8380（标称最大 3.4 GHz），本机 i5-14600KF（最大 5.3 GHz）。
同规模下本机 collect/learn 均明显更短；结合大量微秒级 kernel 与 host autograd 的 trace，
证据支持 host 调度/单线程能力是服务器慢的重要因素，但频率规格不是隔离变量的因果实验。
全量 batch 让服务器承担更多 GPU 工作，不能把全量与缩小模型的 SPS 当作算法质量的公平对照。


服务器全量同卡对照最终结果：GPU 6 / CPU 64–79，4 Torch threads，全部 9 个稳定窗口，
baseline 9288.3 → 融合 9555.4 env-steps/s（+2.9%），collect 55.7→56.7 ms/步，
learn 359.3→320.5 ms/步（-10.8%），CPU 约 4.72→4.81 核 / 16 核，
console GPU 窗口值均值 26.3%→28.2%，整卡占用约 24.3 GiB。退出码均为 0。
原始日志 `steady-{local,full}-{base,fused}.log`；一秒 GPU 采样 `steady-resources.csv`
包含启动/编译阶段，不能直接拿整份平均值当稳态训练利用率。

长训在 2026-11-17 05:42 UTC 已达 500/30010，最近 125-step 窗口 9852 env-steps/s；
尚无完整 40-clip 成功率，稳态速度不保证达标时间。融合减少了 17.5%（缩小）/21.0%（全量）
kernel，但剩余数万次小 kernel 和 host 调度仍限制 5090，未达到 GPU 饱和。
本次不引入多 learner/DDP、不减少更新次数来掩盖这个限制；后续收敛验收继续保留。

### 编译边界与 worker 预算实施（2026-11-17）

- [x] 固定 909344b runtime 基线，改造 WM/critic backward 编译边界并验证更新一致性。
- [x] 预计算 Manager 工作状态结构，合并 imagination 单步，验证跨步梯度/RNG/隔离。
- [x] 提取公共 worker CPU 放置，迁移 APG torch_threads，检查 FastSAC 行为与 spawn affinity。
- [x] 本机缩小与服务器全量三次 512-step 对照，按同一区间统计并检查 GPU trace。
- [x] 完成从零训练及 40-clip 评估（35/40）；不将吞吐提升等同于达到 40/40。
- [ ] 在相同硬件/recipe 的多种子长训中确认 time-to-quality；优化实现的全量 40/40 尚未验证。

不修改正在运行的训练。保持 UTD、网络、奖励和终态语义；未完成的性能/质量验收
明确保留，不将短跑吞吐当作成功率证明。


实施中间验收：WM/critic CUDA 连续三轮更新的参数、梯度及跨代指标存储对照通过；
Manager functional transition 的 eager/aot_eager 多步梯度、噪声与 RNG 对照通过。
266 项快速测试通过，1 skipped；ONNX/真实 spawn 4 passed；真实父进程绑核后
collector 使用另一 CPU 的 spawn 测试通过。实际 CPU G1 小规模新训完成 12 步及 checkpoint。

本机同卡 3 × 512-step 更新边界实验（128–512 共 13 个等长窗口）：
baseline SPS=12486.7/12511.1/12741.1；更新编译 SPS=13537.0/13484.8/13534.2，
中位提升 8.2%。原始日志 `/tmp/apg-plan-local-{baseline,stage-a}-{1,2,3}.log`。
服务器全量及完整 imagination 候选尚在验收，不能以本机结果替代。

线程配置迁移：删除 `algo.trainer.torch_threads` 覆写；需要固定进程级预算时在启动前设置
`OMP_NUM_THREADS`/`MKL_NUM_THREADS`，需要 worker 隔离时使用与 FastSAC 同名的
`algo.trainer.async_options.{learner_cpu_cores,collector_cpu_cores}`。
没有新增运行时文件；公共 CPU helper 移至现有 numa.py，测试和职责说明在现有文件扩展。


#### 实现与可复现验收

实现分为三个提交：`956996a`（WM/critic 更新编译）、`8be31c3`（Manager 显式工作状态与
imagination 单步编译）、`7b10c05`（公共 worker CPU 调度及归档配置迁移）。没有新增运行时文件。
`rand.py` 继续负责可携带、可恢复的张量 RNG；functional transition 传入并返回 RNG 状态，
不能以全局随机数或删除噪声替代它。文件职责沿用设计文档中的模块边界。

性能基线固定为 909344b runtime（0f0d123 只增加文档）。每种实现运行三次 512 collector steps，
统计 128–512 的 13 个等长日志窗口，以窗口 SPS 的调和平均得到单次吞吐，再报告三次中位数。
每次都从头启动；保持 seed、更新比例、模型、batch、horizon、日志频率相同。编译冷启动不计入
稳态指标，也不计作吞吐收益。服务器采用同卡交替对照，并行实验使用不同 GPU 和 CPU 配额，
但共享主机和文件系统，因此不是独占整机实验。

本机 RTX 5060 Ti / i5-14600KF，缩小默认网络、async、OMP/MKL=4：

| 实现 | 三次 env-steps/s | 中位数相对基线 |
| --- | --- | --- |
| 基线 | 12486.7 / 12511.1 / 12741.1 | — |
| 仅更新编译 | 13537.0 / 13484.8 / 13534.2 | +8.2% |
| 完整实现 | 14744.6 / 15295.6 / 15374.8 | +22.3% |

完整实现 learn 约 113–119 ms/collector step，基线 141–143 ms；collect 均约 20–21 ms。
完整实现本机第一次编译的首个 32-step 窗口耗时约 175 秒；缓存热启动结果不能用来推导冷启动收益。
原始日志 `/tmp/apg-plan-local-{baseline,stage-a,candidate}-{1,2,3}.log`。

CUDA profiler 的同规模单轮证据（带 profiler 的时间不用于计算训练 SPS）：

| 项目 | 基线 | 仅更新编译 | 完整实现 |
| --- | --- | --- | --- |
| GPU kernels | 31664 | 31336 | 29619 |
| kernel 总时间 | 71.18 ms | 69.37 ms | 67.26 ms |
| cudaGraphLaunch | 0 | 60 | 60 |
| 峰值 allocated | 4.868 GiB | 4.802 GiB | 4.884 GiB |

完整实现仍有约 2.3 万次 host kernel launch，policy CPU span 从约 899 ms 降至 732 ms。
因此已验证 CUDA Graph 真正重放，但并未消除 host 调度瓶颈，也不能宣称 GPU 已饱和。
服务器证据在 `data/apg-async-20261117/{fk-local,update-profile,final-profile}/` 的 summary/trace JSON。

本机线程默认值验收：不设置 OMP/MKL 时实际为 14 threads，async 15285.3 SPS；
sync 13322.4 SPS，而 sync 4 threads 为 13068.0 SPS（各一次 256-step，统计 128–256）。
未见默认线程数引起灾难性退化，无需 APG 专属固定线程字段。CPU affinity 不变时不调用
sched_setaffinity，避免默认启动无谓要求绑核权限。显式 worker 配额仍按 FastSAC 既有规则
把 intra-op threads 限制到所分配 CPU 数。

验证：487 项快速测试通过、1 skipped、42 deselected；最后补充默认 spawn 无需 affinity
权限的回归后，NUMA 全部 10 项通过（含真实子进程）。WM/critic 真实 CUDA 连续更新测试通过；
实际 G1 CUDA 三步轨迹的输出、所有输入梯度和 RNG 对照连续两轮通过。
CPU G1 新训 12 步后恢复到 20 步通过，实际 learner/collector 使用不同 CPU；旧 run 配置
恢复和真实 ONNX 导出通过。完整 pre-commit 通过。CUDA 在 sandbox 中 skip 不作为硬件验收，
上述 CUDA 结论来自专用 GPU 实际执行。


服务器单独更新编译对照（GPU 6 / CPU 64–71，全量配置、4 threads）：
基线 9415.9 / 9398.2 / 9305.2 SPS，更新编译 9274.7 / 9723.6 / 8989.1 SPS，
中位数约 -1.3%。learn 从约 347–350 ms 降至 331–341 ms，但端到端吞吐没有稳定收益。
因此不能声称更新编译单项在服务器提速；保留依据是本机重复收益及完整组合验收，
最终服务器结论以同卡完整组合的三轮数据为准。日志 `update-{baseline,stage-a}-{1,2,3}.log`。


从头训练质量验收：本机缩小 recipe，seed=1、async、8192 collector steps，使用最终
`model_0008192.pt` 的 EMA policy。训练计时约 19m08s（包含该次启动后的编译窗口，缓存已有），
四进程评估 53.4 秒，**35/40**。每条从首帧到末帧，仅一次 rollout，无重试、无 checkpoint
挑选；使用标准 play 配置，无 observation/reset noise，成功要求完整 clip truncation 且无
hard termination。失败动作：dance1_subject1、fallAndGetUp1_subject1、fallAndGetUp1_subject4、
fallAndGetUp2_subject2、fallAndGetUp2_subject3。尚未达到 40/40。

权重 SHA-256：`30bbab63c35b7aa2a38d62b6ec334592de86f157233bd9df3d8a86f9e50ed91b`。
归档根目录 `runs/apg-implementation-validation-20261117/`：
`evaluation/results.json` 含逐动作判定与 hash；`apg-plan-evaluate.py` 保留本次评估入口；
`benchmarks/` 保留本机原始性能日志和统计脚本。TensorBoard 与权重位于其
`runs/g1-wbt-supermimic/motrix/torch/apg/26-11-17_14-45-49-314830/`。
评估 JSON 中 checkpoint 原始路径保留为 `/tmp/apg-plan-quality-candidate/...`，归档副本内容相同。

历史 8ee51d5 小模型 sync/async 8192-step 单种子结果分别为 32/40、29/40，但硬件和代码基线
与本次不同，不能据此宣称优化提高了算法成功率或已证明 time-to-40/40 改善。
原有 `apg-full-909344b-20261117` 全量长训未修改，继续独立运行；它不代表本次实现的全量质量验收。


服务器完整组合最终对照（GPU 7 / CPU 72–79，全量网络/4096 envs，OMP/MKL=4）：

| 实现 | 三次 env-steps/s | 中位数 |
| --- | --- | --- |
| 基线 | 8818.6 / 8744.1 / 9432.7 | 8818.6 |
| 完整实现 | 10447.7 / 10253.3 / 10843.3 | 10447.7 |

中位吞吐 **+18.5%**，三组交替配对均提升（约 +18.5% / +17.3% / +15.0%）；
learn 从 346–352 ms 降至 280–293 ms，collect 从 68–70 ms 变为 69–72 ms。
三轮均正常退出。日志位于服务器 `data/apg-async-20261117/final-{baseline,candidate}-{1,2,3}.log`。
本机/服务器均通过实际重复验收，接受完整实现；仍需后续优化剩余 host launch 与 BPTT 开销。

服务器线程诊断使用 GPU 1 / CPU 4–11，全量模型、固定 affinity，独立调整 learner/collector
线程数，每种一次 256-step，统计 128–256 的五个窗口。诊断 wrapper 只用于实验，不进入产品配置：

| learner / collector threads | SPS | collect ms | learn ms |
| --- | --- | --- | --- |
| 1 / 4 | 9102.0 | 68.0 | 287.5 |
| 2 / 4 | 11164.1 | 68.2 | 276.6 |
| 4 / 4 | 11457.1 | 69.3 | 272.3 |
| 4 / 1 | 11383.6 | 74.2 | 269.3 |
| 4 / 2 | 11353.7 | 70.6 | 271.0 |
| 继承环境（learner=8） | 10971.0 | 73.4 | 277.0 |
| sync，4 threads | 8961.6 | 193.8 | 235.9 |
| sync，继承环境 | 9051.2 | 194.3 | 240.1 |

这些单轮结果不用于选定新的算法默认值。显式 4 threads 可作为当前硬件可复现启动预算；
默认继承环境可运行且接近它，固定单线程没有优势。async 与 sync 的 collect 计时口径包含不同
等待区间，不能将二者 collect 差值解释成仿真本体加速。日志为服务器 `thread-*.log`。

上述短跑实验正常结束并释放 GPU；原有全量长训随后按下文单 GPU 约束停止。


### 最新代码单 GPU 全量资源评估（2026-11-17）

按单 GPU 约束，停止额外 GPU 6 的诊断及 GPU 5 上旧 909344b 长训；旧运行保留
15000-step checkpoint、TensorBoard 和日志。此次重新从头运行 HEAD `0c83442` 的完整源码
快照，GPU 5，CPU 均位于 NUMA 1，不恢复旧权重。远端 learner.py SHA-256 与本地一致：
`ab5cbaaa7d2f18dc43f322bbe798e0077e591bde68695603a46c2b2b5f3395d1`。

全量配置保持 4096 envs、actor [2048,1024,512,256]、dynamics [2048,2048,2048]、
policy batch 4096、WM batch 2048、replay 1000、history 10、WM horizon 8、policy horizon 32，
UTD=0.4、每 learner round 10 次 WM 与 20 次 critic 更新。CPU 配额不进入算法 recipe。

吞吐统一统计 128–256 的五个 32-step 窗口，取调和平均；共享配额运行 512 steps，另外两个
运行 256 steps，均退出正常。资源从 128 步后以约 1 Hz 采样，仅取前三十个样本，排除短跑
结束释放 CUDA context/checkpoint 阶段。CPU 为训练进程及其线程的 /proc CPU 时间增量，
单位为实际消耗的逻辑核数，不是整台 160-thread 机器的百分比；GPU 为 NVML 忙碌时间比例，
不是 SM occupancy 或峰值 FLOPS 百分比。

| CPU 方案 | SPS | collect ms/step | learn ms/step | CPU 实耗核数 | GPU busy | 显存 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 共享 CPU 40–79，Torch 各 4 threads | 11149.4 | 49.70 | 297.08 | 9.34 | 30.0% | 24.85 |
| learner 48–51 / collector 40–47，4 / 8 threads | 12068.6 | 50.88 | 258.26 | 4.71 | 36.8% | 24.84 |
| learner 48–51 / collector 64–79，4 / 16 threads | 13064.4 | 38.80 | 259.18 | 5.65 | 34.4% | 24.84 |

4+16 配额单轮比共享配额快 17.2%，比 4+8 快 8.3%。其中 learner 实耗约 0.96 核，collector
约 4.69 核；共享配额分别为 1.00 / 8.34 核。扩大可用 CPU 后 collector 可以消耗更多核，
但更多 CPU 时间不等于更高吞吐。独立配额减少竞争并提升有效流水线吞吐；这组试验同时改变
affinity 与线程预算，不能把收益单独归因于绑核或某一个线程数，也不能声称已找到全局最优。

4+16 的 TensorBoard 细分：端到端约 313.5 ms/collector step；learn=259.18 ms（其中
publish=2.15 ms），主进程 collect=38.78 ms，剩余约 15.6 ms 尚未单独归因，含日志、同步与
编排等区间。learn 占已计时 collect+learn 的约 87%。collector 自身每步 108.57 ms，
接收权重约 1.46 ms，二者与 learner 并行，不能再次加入端到端总时间。ring_fill 平均 1/2、
weight_version_lag 平均 1.2；当前没有证据支持通过扩大 ring 或继续增加 collector 数量解决
主瓶颈。collect 包括接收、replay 入库、归一化和 episode 统计，不能直接解释为纯仿真时间。

全量 learner 单轮 profiler（共享配额运行的第 9 个更新，profiler 开销另计）：

| 区域 | CPU wall span | 备注 |
| --- | --- | --- |
| WM 更新整体 | 178.23 ms | 约占 WM+policy 的 20.9% |
| policy 更新整体 | 676.34 ms | 约 79.1%，包含以下各子区间 |
| 32 个 imagination step | 225.73 ms | policy 子区间 |
| actor BPTT backward | 370.32 ms | policy 最大子区间 |
| 20 次 critic backward callable | 21.98 ms | 不含全部 critic 编排/optimizer |
| actor clip + optimizer | 5.83 ms | policy 子区间 |

该轮 30301 个 GPU kernel，89.6% 小于 10 μs，中位 kernel 1.248 μs，kernel 累计执行时间
238.50 ms。实际 cudaGraphLaunch=60；host kernel launch 约 2.38 万次；36 次 stream
synchronize 累计 39.76 ms。峰值 allocated=21.50 GiB、reserved=23.13 GiB。
profiler 总耗时约 1.29 秒，不能当作无 profiler 的正常 learner round 时长，也不能将嵌套
区间相加。当前证据支持 policy/BPTT 的串行 host 调度和大量小 kernel 是主瓶颈，而非整机
CPU 核数或 GPU 容量不足；显存占用高不代表 GPU 计算饱和。

优化顺序：

1. 使用现有 worker 配额将 learner 与 collector 分开，保持单 GPU、同 NUMA；通过重复试验
   确认当前较快的 4+16 配额，不新增 APG 私有线程字段。
2. 优先把 actor rollout/loss/backward 扩为可复用编译区域，或先比较较小的固定步块；采样与
   有效性判断在外部，保持跨步梯度/RNG/工作状态所有权，不逐步重置仍被 BPTT 使用的 graph 存储。
3. 拆分主进程 collect 内部的 recv、replay.extend、normalization、episode 统计；当前
   normalization 逐字段 .to(device)，与 replay 已有 pinned staging 存在重复数据迁移路径。
   优先复用现有 staging 与 transport 契约，避免复制一套 APG 专属数据通路。
4. 再定位 WM 更新中 backward callable 外部的采样、裁剪及 Muon/Adam 开销，评估合法的
   批量/融合 optimizer 执行。保持更新次数、优化器数学、LR/EMA 和终态语义。
5. 为日志/同步/编排余量增加短期诊断计时，验证后再改；现有数据不足以断言 GFS 或日志是主瓶颈。

本次原始资料位于服务器 `data/apg-async-20261117/`：`latest-{shared4,split8,split16}.log`、
对应 `*-resources.json`、`latest-full-profile/{summary,trace}.json` 与各 `latest-*-run/`。
启动脚本与统计脚本位于 `apg-async-20261117/`，源码快照为 `src-0c83442`。


4+16 独立重复一次：12189.4 SPS、collect=39.02 ms、learn=260.84 ms；前三十个资源样本
CPU=5.27 核、GPU busy=32.2%、显存=24.84 GiB、功率=213.45 W。首次为 13064.4 SPS，
两次阶段耗时接近，但端到端吞吐相差约 7%。第二次端到端约 336 ms，未归因余量约 36 ms，
高于首次约 16 ms，说明日志/同步/编排区间需要继续细分，不把最高单次 SPS 当作保证值。
两次相对共享配额单轮提升约 9.3%–17.2%；此范围是观察值，不是统计置信区间。
全部最新源码诊断正常结束，GPU 已释放，未自动启动新的长训。

建议后续长训以单 GPU 5、learner CPU 48–51 / collector CPU 64–79 起步（同 NUMA 1）；
若重视 CPU 配额效率，4+8 也是有效候选。配置只通过已有字段传递：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 mxlab train \
  task=g1-wbt-supermimic/motrix.apg +experiment=g1-wbt-supermimic/apg-server \
  algo.trainer.async_options.learner_cpu_cores=48:51 \
  algo.trainer.async_options.collector_cpu_cores=64:79
```

此命令在单卡 Docker 的 CPU 40–79 配额内运行；GPU 通过 `--gpus device=5` 选择。
collector 的实际 Torch 预算由公共 pin_worker_cpus 按 16 核配额覆盖为 16，learner 保持 4。
按 1.22–1.31 万 SPS 估算 30010 collector steps 约 2h37m–2h48m，另加启动/编译与 checkpoint
开销；这是预算耗时外推，不是达到 40/40 的时间承诺。


### Policy/BPTT host 调度优化

- [x] 提取完整 actor 数值轨迹与 backward，保护输出所有权与 eager 等价。
- [x] 验证完整轨迹编译；全量首次编译未通过时限，不接受候选。
- [x] 验证固定 8 步普通编译块；数值通过，全量首次编译同样超时，不接受候选。
- [ ] 同卡同 CPU 配额三组 512-step 配对；SPS 中位提升至少 10%、policy span 降低至少 20%。
- [ ] 本机缩小配置、真实 CUDA、checkpoint/RNG、ONNX 与从头 8192-step 配对评估。

服务器只使用 GPU 5，learner CPU 48–51、collector CPU 64–79；基线固定 4450488。
不改变网络、horizon、UTD 或训练数学；未通过正确性/性能/质量门槛不启用新默认执行路径。


完整 32 步 actor 编译候选在全量配置下未通过首次编译门槛：17 分钟总运行时限
（含环境启动与 WM warmup）退出 124，尚无首个 32-step 日志窗口；采样调用栈停留于
AOTAutograd graph capture，GPU 约 14 GiB且未进入训练。该候选不进入默认路径。
轻量真实 G1 的 3 步完整编译及三轮 WM/actor/critic/RNG/恢复对照通过，但不能替代
全量可用性。转入计划中的 8 步普通编译块；3、9、32 步独立轨迹梯度 oracle 已通过。


8 步块候选在相同全量配置与 CPU 配额下也退出 124（2026-11-17 08:39:31 UTC），
17 分钟总时限内未产出首个 32-step 日志窗口。中途显存约 16.1 GiB，不是 OOM；
它在本机小 batch、9 步真实 G1 上通过三轮更新/RNG/恢复对照，但不能替代全量门槛。
完整图和步块的不可变源码分别保留于服务器 src-actorwhole、src-actorblock，日志为
latest-{actorwhole,actorblock}.log 与对应 .exit。

按计划撤回两种新编译策略，默认恢复原有单步普通编译。只保留数值轨迹/backward 的职责
拆分与已完成轨迹的存储所有权。没有达到 SPS +10% / policy span -20% 的性能验收，
不启动三轮长基准或 8192-step 质量配对来掩盖编译门槛失败；这些项目仍未完成。
后续若继续降低 host launch，应优先研究在已有单步编译结果之上捕获完整 forward/backward
执行，避免再次展开巨大的 AOT 图；需要独立设计静态输入 staging 与 CUDA Graph 生命周期，
本次没有引入未验证的自动 capture 或静默 fallback。


最终保留实现验收：491 passed、1 skipped、44 deselected 的快速回归通过；真实 CUDA actor
测试 2 passed；真实 G1 9-step 三轮 WM/actor/critic 参数/指标/RNG 与 4450488 基线对照通过，
包含中途恢复后的下一轮更新。CPU 4-env 新训 12 步、checkpoint 恢复到 20 步、真实 ONNX
导出（32 个验证样本）均通过。服务器两次有时限诊断均已退出，GPU 已释放。
此次未新增运行时模块、配置字段或编译自动降级分支；未获得新的训练成功率或稳态加速结论。


### 复用单步编译的整轨迹 capture

- [x] 固定输入与 Manager 工作状态缓冲，捕获完整 actor forward/backward 的执行。
- [x] 覆盖改变输入/噪声、连续 optimizer 更新、warmup 不提交状态、恢复与输出所有权。
- [x] 单卡 GPU 5 全量编译/显存门槛、三轮配对吞吐与 profiler 验收。
- [ ] 本机回归、ONNX/恢复与通过性能门槛后的从头训练质量配对。

基线为 b58afd2；同 GPU、CPU 4+16、全量网络/UTD/horizon 不变，继续采用之前性能与质量门槛。

初版捕获暴露 cuBLAS 工作区生命周期缺陷：第一轮数值正确，WM/critic 自动图录制清理
预热工作区后，第二轮 actor replay 非法访存。隔离关闭 WM/critic 自动图后三轮通过；
最终保留其加速路径，仅让 actor 工作区归属自己的图池。真实 G1（小网络、horizon=9、
想象噪声开启）连续三轮参数/指标/RNG 对照通过，包含原位 checkpoint 恢复。
新增交替自动图录制的真实 CUDA actor 测试 2 passed；控制面、builtin 与 Manager modeling
快速测试 353 passed、1 skipped、44 deselected。全量性能与质量仍需独立验收。

本机默认缩小 recipe（2048 env、async）128-step 新训、恢复到 160 steps 与 ONNX 32-sample
校验通过；core 快速测试另执行 261 passed、4 deselected，prek 全仓检查通过。
服务器修复快照 `src-capture2` 的 64-step 全量短测正常退出，启动至退出约 102 秒；
`capture2-source.sha256` 固定两个运行时文件内容。基线 `src-capturebase` 为 b58afd2。
三组 512-step 交错顺序为 candidate/base、base/candidate、candidate/base；使用 128–512
的 13 个窗口，各 run 的 SPS 用窗口调和均值汇总，再比较三轮中位数。profiler 独立运行，
不混入吞吐窗口；资源采样取 iter 128 后前 30 秒，整卡显存每秒采样覆盖整个 run。
证据保存在服务器 `data/apg-async-20261117/capture-{pair,profile,quality}*` 与
`capture-report.json`。辅助 shell/Python 仅负责本次验收启动、采样、汇总，不加入长期运行时。
质量队列仅在吞吐 +10%、policy CPU span -20%、整卡峰值 ≤29 GiB 全部通过后启动：
全量配置、同 GPU 5、seed=1、基线与候选各从头 8192 steps，最终 checkpoint 四进程评估。
候选成功数更低时才追加 seed=2 的完整配对；没有中途择优、续训或重试片段。

AMP 开启、原单步编译基线与候选整轨迹捕获的真实 G1 三轮参数/指标/RNG 对照同样通过，
包含第二轮后的原位 checkpoint 恢复；并非仅验证 FP32 eager 参考。

三组单 GPU 512-step 配对均正常完成，未改变网络、batch、horizon、UTD 或 CPU 配额：

| 指标 | 单步编译基线 | 整轨迹 capture |
| --- | ---: | ---: |
| 各轮稳定 env-steps/s | 13195 / 13112 / 13269 | 22961 / 21002 / 23292 |
| SPS 中位数 | 13195 | 22961（+74.0%） |
| learner 时间中位数 | 258.7 ms | 124.6 ms |
| collect 时间中位数 | 38.3 ms | 35.3 ms |
| CPU 使用核数中位数 | 5.72 | 9.15 |
| GPU 利用率中位数 | 35.1% | 62.7% |
| 每秒采样整卡显存峰值 | 25.47 GiB | 26.28 GiB |

上述资源数据来自独立每秒采样；计时为无 profiler 的稳定窗口，不包括首次编译/初始化。
完整启动至 512-step 退出：基线 253 / 253 / 257 秒，候选 187 / 194 / 184 秒。
包含启动后这一短任务的加速小于稳定窗口加速；不把稳态 SPS 当作端到端到达成功率所需时间。

独立 profiler 在第 13 个 learner round 采样，同一配置与 CPU 配额：policy CPU span
737.4→184.4 ms（-75.0%）；actor rollout+backward 入口 CPU span 665.7→40.5 ms。
整轮 `cudaLaunchKernel + cuLaunchKernel` 次数 23591→4949（-79.0%），graph launch
60→61：新增的一个 actor graph 替代了轨迹内的串行 host launch。WM CPU span
226.6→207.1 ms，成为候选较大的单阶段开销。上述 span 包含 profiler 开销、嵌套区间
不能相加，不能直接替代无 profiler 的 124.6 ms/collector-step learner 时间。
CPU span 从 trace 的 CPU user_annotation 核对；key_averages 中同名 GPU annotation
不能覆盖 CPU 项，GPU 子项时间也不能作为整段 graph 的真实执行时间。

2026-11-17 17:37（北京时间）性能 gate 通过后，GPU 5 已启动 seed=1 基线从头训练，
队列随后训练候选并逐一进行四进程 40-clip 评估。状态文件 `capture-quality.status`，
TensorBoard 在 `capture-quality-seed1-{capturebase,capture2}-run/runs/g1-wbt-supermimic/`
之下。质量验收尚未完成，没有新的 x/40 结果，也没有据此宣称达到 40/40 或质量不退化。

### 2026-11-17 细分诊断与优化优先级

质量状态更新：用户要求停止基线后，只完成 capture2 seed=1 从头 8192-step 训练。
训练控制台耗时 23m27s，四进程评估 167.9s，启动至评估结束 26m51s，结果 37/40。
失败动作为 fallAndGetUp1_subject5、fallAndGetUp2_subject2、walk3_subject3。
没有完整配对基线质量结果，不能据此证明质量不退化。

诊断证据目录为服务器 `/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/capture-profile-detail/`，
包含 `trace.json`、`summary.json`；任务状态为同级 `detail.status`。
采用独立 src-detail 快照，4096 env、全量网络、GPU 5、learner 4 核、collector 16 核，
在第 13 个 learner round 采样。新增 record_function 标记只用于解释现有计算范围，
不改变优化器或采样语义；本次没有新增运行时文件。

| CPU 范围 | 总毫秒/learner round | 说明 |
| --- | ---: | --- |
| WM | 213.46 | 包含下面三个子范围及编码等工作 |
| WM sample | 39.99 | 10 次采样 |
| WM backward | 63.08 | host 提交范围，不等于完整 GPU 执行时间 |
| WM optimizer（含裁剪） | 64.62 | 同步等待占 32.14 ms |
| Policy（含 critic） | 185.89 | 与 WM 为顺序顶层范围 |
| Actor graph 入口 | 41.83 | GPU 完成等待部分落在后续裁剪 |
| Critic backward | 34.29 | 20 次更新 |
| Critic optimizer（含裁剪） | 13.97 | 20 次更新 |

31 次梯度裁剪范围累计 120.59 ms；最长一次 71.67 ms 中 cudaStreamSynchronize
占 71.04 ms。不能将等待之前 GPU 工作完成的时间作为可直接删除的 CPU 浪费。
WM sample 内有 400 次 aten::index（累计嵌套 CPU span 19.65 ms）、310 次
contiguous/clone、1160 次普通 kernel launch。子算子包含关系不能相加。

将 learner trace 的 kernel/memcpy/memset 区间取并集并裁切到 CPU 顶层范围：
WM 107.96/213.46 ms（50.6%），policy 131.00/185.89 ms（70.5%）。
这不是整卡 NVML 利用率：没有 collector CUDA 活动，跨范围的异步尾部被裁切，
且 profiler 有 Buffer Flush 15.71 ms、Command Buffer Full 14.95 ms 等扰动。
此前无 profiler 配对短跑 GPU 利用率中位数 62.7% 仍是更合适的稳态参考。

优化按以下顺序独立验收，不降低 batch/horizon/更新次数：

1. **WM 按需采样。** 当前 sample_windows gather 全部状态、command/obs、整段当前
   状态及 rewards/done/previous_action；WM 只消费六类根/关节状态的首帧、对应 next
   窗口、actions 和 valid。为 SequenceReplay 增加明确的字段投影/首帧读取能力，保留
   默认完整输出契约；复用相同 slot 抽样和 mask 构造，不能额外消耗随机数或改变分布。
   先去掉未消费 gather，再评估固定形状索引/编码融合；避免为任务复制另一套 replay。
2. **状态编码与 WM 外围融合。** 将根/关节状态 encode 和固定形状准备合并编译，减少
   Python 调度及小 kernel。保留已有 backward graph，先验证是否重编译及数值一致。
3. **消除重复 H2D。** SequenceReplay 已有 pinned staging；归一化目前从 CPU transition
   再搬运一次。复用已上传数据需要明确借用有效期和 event 依赖，不可直接泄漏私有 staging
   或提前复用 ring 槽位。先补 ingest/normalization 分项计时，再决定是否值得扩展接口。
4. **优化器和 critic 后处理。** WM Muon 的矩阵计算是算法必要成本；优先编译其 host
   小算子序列和 critic 固定形状准备，保留非有限梯度在更新前报错的语义。不能简单删除
   error_if_nonfinite 或改变检查频率。动态目标索引可用等价 masked moments 替代，需测试
   部分/全部有效样本、零有效样本以及累计统计数值。
5. **采样端与资源配额。** learner 加速后重新测 producer ring-full、consumer empty、
   collector env step 和推理时间；再调整 CPU 配额。当前证据不支持盲目加线程或 ring。

验收：每项候选与原版交替三组无 profiler 短跑，比较同一稳定窗口的 SPS、阶段耗时、
显存、CPU 核数和权重滞后；开启标记的额外开销也需单独验证。最终用至少五分钟
NVML 1Hz 样本验证平均 GPU 利用率目标 90%，同时要求吞吐改善。若达不到，报告
剩余瓶颈而不增加无效工作。最终 fresh 8192-step + 四进程 40-clip 评估独立验收质量。
截至本次分析，仅完成诊断和 CPU 单元测试（23 passed），没有新的优化吞吐或质量结果。


#### 按需采样候选实际运行

实现 SequenceReplay.sample_windows 的字段投影、首帧当前状态和可选 metadata；
默认调用保持完整输出，APG WM 只请求 root_keys + joint_keys、actions、valid。
共享同一抽样和 mask 路径，不添加任务专用 replay。测试对比 episode 边界及回绕后的
完整/投影结果和抽样 RNG 状态：replay + APG CPU 测试 39 passed，ruff 通过。

GPU 5 全量配置 64-step 真实训练正常结束，证据 `capture-profile-projected/` 与
`projected.status`。单次同阶段 profiler 对比：WM sample 39.99→21.96 ms（-45.1%），
WM 总范围 213.46→179.72 ms（-15.8%），普通 cudaLaunchKernel 4529→3889（-640）；
policy 185.89→175.02 ms，未修改 policy，其变化不可归因为本次优化。
这证明减少了采样工作，但单次 profiler 不证明稳态端到端加速率；尚需交替重复的
无 profiler 吞吐对照及长期质量评估，也未验证 90% GPU 利用率目标。

#### 单卡利用率优化第二轮（进行中）

用户要求最终候选 fresh 约 5000 collector steps 后进行 40-clip 多进程评估，
替代本轮先前计划的 8192 steps。保留全量网络、horizon、batch 和更新比例。

职责保持在现有文件内：`optim.py` 管理 Muon/Adam 分工、编译更新及恢复后的学习率
张量；`learner.py` 原地更新学习率并保留更新前的非有限梯度检查；`test_apg.py`
保护衰减和旧 checkpoint 恢复后的连续更新语义。不新增 runtime 文件。
Muon 使用普通 torch.compile，不启用第二套手动 CUDA graph，避免影响 actor graph
生命周期。直接编译在 BF16 Newton–Schulz 中间舍入上未通过一致性测试；显式开启
`emulate_precision_casts` 后本机 CUDA 连续六步及中途恢复验收通过。

独立源码快照：src-detail、src-projected、src-muoncompile；实验全部串行使用 GPU 5。
首个 512-step 基线（128–512 稳定日志窗口）22868.5 SPS，collect 35.08 ms、
learn 126.29 ms；资源采样前 30 个稳态样本平均 GPU 63.23%、CPU 9.43 核。
其余候选仍待完成；这些短窗口不代表五分钟持续利用率，也不证明达到 90%。

Muon 编译候选在服务器也通过数值验收，但无 profiler 512-step 实测 24172.2 SPS、
GPU 61.3%，相比仅投影采样 24399.7 SPS、GPU 60.8% 没有明确收益。因此撤回该
候选的 runtime 与专用测试，不为未验证收益保留精度选项、张量学习率和恢复分支。
服务器独立快照与日志保留用于审计。继续验证状态编码融合和重复 H2D 消除。

重复上传消除：`SequenceReplay.extend` 返回只读借用的 device batch，同一 stream
立即消费且不得跨下一次 extend 保留；CPU 返回输入别名，CUDA 返回 staging 别名，
replay 自身始终独立复制。公共生命周期契约记录在 `wiki/design/rl-sequence-data-plane.md`。
`train.py` 负责将该借用交给归一化，`learner.py` 复用已上传的 obs 与状态，同时编译
状态编码的小算子序列。两种设备的消费顺序测试通过，不新增 runtime 文件。

单线程 learner（collector 仍为 16 核）首组 512-step：25337.2 SPS，collect 35.91 ms、
learn 113.10 ms、GPU 60.37%、CPU 合计 10.08 核；比 projected 四线程吞吐高约 3.8%，
但没有提高 GPU 占用。GPU 5 与 CPU 40–79 均属 NUMA 1，没有跨 socket 绑定问题。

状态编码融合候选：25086.5 SPS、learn 116.99 ms（直接基线为 Muon 候选 123.29 ms），
GPU 60.57%；首次编译触发新 backward 图，冷启动成本明显，需与稳态分开。
复用上传数据候选：25647.4 SPS、collect 32.04 ms、learn 117.85 ms、GPU 64.67%。
以上均为单次 512-step 实测，未做重复置信区间。保留有收益的三项改动，去掉 Muon
编译后，采用 learner 1 核、collector 16 核启动 `optimized-5000-seed1-streamed`。
2026-11-17 11:25:41 UTC 开始，从头训练，无 resume；5000 step 后自动四进程评估，
中途不选 checkpoint。外部脚本采集 300 个约 1 秒资源样本以及分阶段 host 耗时。

验证补充：公共 RL 快速测试 213 passed；APG/Replay 快速测试 40 passed；CUDA/CPU
backward 图回归 8 passed；borrowed batch CPU/CUDA 2 passed；额外首帧/窗口状态
编码 CUDA 数值对照通过。端到端训练和成功率仍以实际完成的结果为准。

#### 5000-step 正式训练与继续调度优化

`optimized-5000-seed1-streamed` 于 11:42:56 UTC 完成训练与评估（总 wall 17m15s）。
最后 4992-step panel 的训练时钟为 13m46s，近期 27–28k SPS；四进程评估 172.8s，
**34/40**。checkpoint 位于：
`optimized-5000-seed1-streamed-run/runs/g1-wbt-supermimic/motrix/torch/apg/26-11-17_11-25-47-535632/checkpoints/model_0005000.pt`。
5000 是 collector steps；此配置累计 1996 learner rounds。先前 8192-step 的 37/40
不是同训练预算对照，不能据此单独判断质量回归。

失败动作（执行帧/总帧）：dance1_subject1 5348/6574；fallAndGetUp1_subject1 3146/8410；
fallAndGetUp2_subject2 7697/8195；fallAndGetUp2_subject3 7125/8195；
fallAndGetUp3_subject1 3446/5109；walk3_subject3 12205/12330。

300 个连续约 1 秒样本的 GPU 平均为 **65.62%**，CPU 合计 **10.85 核**；
不能用 console 的瞬时 95% 代替该平均值。用户明确不需要 MPS；本轮没有启动任何
MPS 服务，未运行的准备脚本已删除。

20 秒、49Hz Python 栈采样（979 个主线程样本/进程）显示 collector 约 43.7% 等待
ring-full、50.4% 位于 env.step；后者 79.1% 落在 native `SceneModel.step_n`。
collector 推理 actions.cpu 仅约 0.1%；learner 约 8.3% 等待批次，另约 4% 克隆
ring 内容。两端交替等待支持检查 ring 容量与 0.4 UTD（每 2–3 步更新）的匹配。
这些百分比是采样墙钟分布，并非 CPU cycles 或 GPU kernel 时间。

新增候选仍只使用现有文件：`replay.py` 将纯 device 窗口 gather/mask 放入编译区域，
RNG 与变化的 host 游标保持 eager，避免改变随机流或随每步重编译；`train.py` 将已有
agent.compile 传给 replay。CUDA 完整/投影窗口在 episode 边界及 ring 复用下逐项、
逐随机状态精确对照通过（2 passed）。

| 单次 512-step 对照 | SPS | collect ms/step | learn ms/step | GPU 30 样本平均 |
| --- | ---: | ---: | ---: | ---: |
| windowfusion，ring=2，collector 16 核 | 25666.8 | 34.23 | 109.20 | 70.13% |
| 同代码，ring=4，collector 16 核 | 30487.3 | 16.89 | 110.52 | 75.33% |

ring=4 首组吞吐提升约 18.8%，尚需长窗口和质量验收。接着测试 collector 31 核，
以及 `learner.py` 中状态编码/五组归一化统计合并的纯 device region；后者保持 CPU
fallback 和累计均值/方差/count，CUDA 连续四批对照通过（1 passed）。尚未声称达到90%。

31 核 collector 未带来收益：29578.3 SPS、collect 17.07 ms、CPU 18.04 核、GPU 67.27%，
相较 ring=4/16 核的 30487.3 SPS 更慢；保留 16 核，不能以更多 CPU 消耗代替有效吞吐。
独立优化器 graph 微基准在 5090 上 512/1024 方阵有收益，2048 方阵仅 3.84→3.68 ms；
还存在编译 BF16 的累积舍入差异，因此不把复杂的 optimizer graph 状态管理并入运行时。

继续验证的最小候选在 `finish_update` 使用 CUDA 顺序断言检查 finite norm，再在同一
stream 排入 optimizer；CPU 仍用原 `error_if_nonfinite=True`。CUDA 不再为每次 finite
检查同步 host，数学裁剪与阈值不变。**错误语义：非有限梯度会触发设备断言，CUDA
context 不可继续使用，训练进程必须结束；host 可能在之后的 CUDA API 才收到异常。**
这不是跳过检查或容忍坏梯度，参数更新排在断言之后；也不提供错误后继续训练的路径。
隔离子进程的 NaN 注入验收通过，8 项既有 CPU/CUDA backward 回归通过。实际 GPU
短跑、长期质量仍待该候选完成；profile 中等待可能移动到 publish，必须比较端到端 SPS。

同一候选将 lambda-return 递推合并编译；首帧/终止/截断计算公式不变，本机 32×4096
随机输入含分数 failure probability 的数值对照通过，微基准 1.09→0.053 ms，不能直接
把该微基准比例当成整轮训练加速率。

第二次正式训练 `devicecheck-5000-seed1-devicecheck` 从 12:02:49 到 12:17:51 UTC
完成训练和评估，总 wall 15m02s；4992-step 训练时钟 11m33s（第一版 13m46s）。
5000-step 四进程评估仍为 **34/40**，172.2s，无重试；300 个连续资源样本 GPU
平均 **78.31%**、CPU **12.58 核**，未达到 90%。不同短跑 GPU 均值波动较大，
不能用单次短窗口的较高结果替代这个长窗口。短跑时 SM 时钟均约 2.81GHz。

失败动作：dance1_subject1 5363/6574；fallAndGetUp1_subject1 903/8410；
fallAndGetUp1_subject4 3026/8410；fallAndGetUp2_subject2 6370/8195；
fallAndGetUp3_subject1 4193/5109；jumps1_subject2 11736/12222。
相同成功条数不表示轨迹完全相同；异步队列调度改变权重可见时机，本轮还没有多 seed
统计，不能把两次 34/40 解释为严格质量等价证明。

进一步移除 host clone：`collector.py` 的 `borrow` 上下文持有 ring 槽，`train.py`
在上下文内完成 replay staging、归一化与 episode/logging 标量读取，退出才 commit。
CUDA H2D 的源仍是 replay 自己的 pinned staging，不从已释放 ring 异步读取。
原 `collect()` 继续返回 owned tensors；sync Collector 使用相同消费上下文。长期保护
lease、异常释放和 owned copy 行为的测试，以及既有进程顺序/checkpoint/failure 测试
全部通过（4 passed）。候选短跑 31941.0 SPS，collect 8.73 ms、learn 107.57 ms；
直接基线 devicecheck 的 collect 为 15.9 ms。没有新增 runtime 文件。

服务器 preset 使用已测得更合适的 ring_capacity=4；机器 CPU 编号继续由启动命令提供，
不固化到通用算法默认值。Hydra 解析验证得到 num_envs=4096、ring_capacity=4。
BPTT 2-step 编译块仅作为独立候选，完整 horizon=32 数值/梯度/RNG/输出所有权验收
通过；在服务器设 17 分钟冷启动时限，不因候选存在就替换默认单步 capture 路径。

截至 2026-11-17 12:33 UTC，最新快速测试为 337 passed、1 skipped、49 deselected；
修改的六个 Python 文件通过 Ruff check/format，git diff --check 通过。
2-step 候选仍在首次 actor backward 编译，Python 栈位于 Inductor/Triton precompile，
不是稳定训练阶段，不能把此时 GPU idle 当成稳态瓶颈数据。候选保持 1020s 总时限。
已串行排队 `borrowed-5000-seed1-borrowed`：候选容器退出后使用已验证的 borrowed
快照从头训练，随后四进程评估全 40 动作；独立 300 样本资源采集，host-stage probe
适配 `borrow()` 获取阶段。状态文件为服务器 artifacts 下 `borrowed-5000.status`。
此队列不使用 MPS，未改变 batch、horizon、更新次数或网络规模；排队不算完成验收。

### 2026-11-18 参数对照准备（待用户确认，未启动）

目标改为尽量在 3000–5000 learner rounds 达到固定协议的 40/40；需与用户确认
这里的 update 是否确指 learner rounds，而非 console collector steps。现有 schedule
warmup=10、UTD=0.4，因此 5000 collector steps 仅有 1996 learner rounds。

borrowed 最终验收已完成：5000-step 34/40，eval 168.1s；4992-step panel 11m01s，
300 个连续资源样本 GPU 平均 81.27%、CPU 13.03 核。启动至训练和评估结束
12:39:09–12:53:35 UTC，共 14m26s。2-step BPTT 编译候选超时，退出码 124，未采用。

参数对照仅改变 actor_learning_rate：4e-4（对照）、6e-4、8e-4；WM 与 critic
学习率仍为 4e-4。网络、batch、horizon、每轮 WM/critic 更新数、UTD、EMA、训练与
评估环境、seed=1 保持一致。尚无证据证明这组学习率能达到目标，不修改默认配置。

预备命令使用 server preset + num_learning_iterations=12510 + checkpoint.interval=2500；
评估 collector step 7500、10000、12510，对应 learner round 2996、3996、5000。
每组从头训练，训练后在同一 GPU 上串行执行三个 checkpoint 的四进程全 40 动作评估，
分别统计训练墙钟与评估墙钟。全部保存的 checkpoint 不等于全部已评估；报告只引用
实际评估点。若取得 40/40，应再独立 seed 验证，不能只凭单 seed 声称稳定达到。

- [x] 对照上游配置与当前 learner 的计数、优化器和学习率。
- [x] 准备一次性启动脚本 `/tmp/apg-tune-20261118.sh`，shell 语法检查通过。
- [ ] 用户确认 update 口径与本轮参数范围（按用户启动前确认的要求）。
- [ ] 单 GPU 串行参数对照，记录三个预算点的成功数、失败片段、训练耗时。
- [ ] 根据对照结果决定后续候选与独立 seed 验证。

### 2026-11-18 XMimic 可直接覆盖参数实验

用户授权先只使用现有 CLI/YAML 字段，暂不实现失败探针、镜像增强、接触奖励。
因此未引入新的 Actor LR 调度代码，也没有把 XMimic 的前期 4 倍 LR 误用为全程常数。
本轮保留 actor/WM/critic 基础 LR=4e-4 和既有 cosine schedule；这是部分参数对照，
不是完整 XMimic 复现。之前准备的固定 LR 三组扫描未启动。

在 `+experiment=g1-wbt-supermimic/apg-server` 上仅覆写训练超参数：

```text
algo.agent.dynamics_hidden_dims=[1024,1024,1024]
algo.agent.world_model.num_updates=5
algo.agent.batch_size=2048
algo.agent.learning_starts=64
algo.agent.actor_learning_rate=0.0004
algo.trainer.num_learning_iterations=12564
checkpoint.interval=2500
logging.interval=128
seed=1
```

单 GPU 5，无 MPS，异步、learner CPU48 / collector CPU64:79、ring=4；4096 env，
WM batch=2048、policy H32、WM H8、history10、UTD0.4、EMA0.995 保持。
从头训练 12564 collector steps，即扣除 64 warmup 后 5000 learner rounds。
训练结束后同卡四进程评估 step7500/10000/12564（learner2974/3974/5000），
使用完整 40 动作、从首帧到尾帧、无重试的既有协议。

运行名 `xmcli-20261118-seed1`，服务器 artifacts 根目录
`/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/`；状态文件 `xmcli-20261118.status`，
训练日志 `xmcli-20261118-seed1.log`。代码快照 `src-xmcli-20261118` 复用已验证 borrowed
基线并同步当前四个 runtime 文件和任务/server YAML。覆盖归档 SHA256：
`2455756bcefc87dd1663696b74e499b5e83818aba51da6fd28f68282c5ba3571`。
本次没有新增或修改 runtime 代码、没有修改任务默认 YAML；一次性启动脚本位于
服务器 workspace 的 `apg-xmcli-20261118.sh`，负责单卡训练、评估与状态记录。

- [x] Hydra 配置解析与启动脚本 shell 语法检查通过。
- [x] 启动单 GPU 从头训练。
- [x] 核实训练正常完成并汇总三个预算点的成功率、训练耗时与资源利用率。


本轮于 2026-11-18 03:46:57 UTC 正常结束，启动至全部评估结束 47m47s。
三组固定协议评估：

| Collector step | Learner round | 成功 | 评估耗时 |
| --- | ---: | ---: | ---: |
| 7500 | 2974 | 34/40 | 157.1s |
| 10000 | 3974 | 34/40 | 164.5s |
| 12564 | 5000 | 34/40 | 161.4s |

12544-step panel 训练计时 38m55s；首个 128-step panel 为 17m35s，首次初始化/编译
窗口成本很高，不能将其全部精确归因为编译时间。之后稳态约 39–41k env-steps/s，
约 21 分钟用于余下训练；300 样本 GPU 平均 41.25%、CPU 14.07 核，显存约20.3GB。
吞吐增加而 GPU 平均利用率降低，host-stage 数据显示 learner 缩短后更多等待 collector。

5000 learner round 失败动作（执行帧/总帧）：dance2_subject1 6057/11284；
fallAndGetUp1_subject4 3438/8410；fallAndGetUp2_subject2 6368/8195；fight1_subject2
2994/12244；fightAndSports1_subject1 1224/12257；fightAndSports1_subject4 971/12257。
三组成功总数相同但失败集合变化；只能说这些已评估点没有达到40/40，不能据此声称
所有中间 checkpoint 都相同，或仅凭单 seed 证明某个参数导致退化。该部分参数对照没有
支持把小网络/少更新配方直接提升为默认，也没有验证 XMimic 缺失机制的收益。
原始结果在 artifacts 下 `xmcli-20261118-seed1-eval-{0007500,0010000,0012564}/results.json`。

### Return history 修复与四组对照（2026-11-18）

用户已批准最多四组加一个独立 seed 复验；范围仅为修复历史更新、现有采样权重与训练参数。
采样修复归 `motion/sampler.py`，行为回归归既有 WBT motion command 测试，课程契约见
[多动作基础设施设计](../design/motion-multi-clip-infra.md)。没有新增 runtime 模块或公共配置字段。

共同配方沿用 xmcli：4096 env、WM1024×3、policy/WM batch2048、warmup64、UTD0.4、
actor H32/WM H8/history10、EMA0.995、全部LR4e-4。A/B/C/D 分别为
(default mix,5 WM)、(0.8/0.1/0.1 mix,5 WM)、(default mix,10 WM)、(0.8/0.1/0.1 mix,10 WM)。
每组从头 seed1，到12564 collector steps/5000 learner rounds。checkpoint每1250 steps，
评估7500、8750、10000、11250、12564；不更改评估协议。采样权重通过隔离源码快照中的
已有 MotionCommandCfg 字段装配，不向全局CLI添加环境参数接口，不覆盖用户任务配置。

一次性工具职责（位于服务器独立 workspace，非运行时 package）：
- `apg-history-check.py`：真实 Manager reset 与 CUDA 编译重建/采样一致性验收。
- `apg-history-run.sh`：单组从头训练、资源采样、五次串行评估、开始/结束状态。
- `apg-history-summary.py`：逐 checkpoint 成功率/完成比例/墙钟、采样分布与训练标量汇总、排序。
- `apg-history-queue.py`：最多四组串行控制；连续两个评估点40/40则停止搜索；选优后seed2复验。
选优依次比较成功数、首次达到该成功数的启动到checkpoint完成墙钟、learner rounds、
平均完成比例；组名仅用于最终确定性破同分。报告所有结果，不宣称单seed稳定成功。

- [x] 补齐历史更新，替换复制旧实现的测试；27项motion单元测试通过。
- [x] 完成真实Manager编译、CPU/CUDA eager一致性验收与GPU短程训练冒烟。
- [x] 完成最多四组实验与最佳组seed2复验。
- [x] 汇总39/40、40/40首次达标时间及未达标时的瓶颈证据。

真实 Manager 默认编译 reset 的五次历史更新验收已通过。额外尝试把整个 sampler
`rebuild` 在 CUDA 上 fullgraph 编译遇到 Inductor `TritonSplitDimKernel reductions`
不支持 argmax 与 scan 的组合；这是训练未使用的路径（实际 sampler 驻留CPU Manager），
未因此修改运行时或给训练加入兼容分支。保留 `historyfix-*-fullgraph-failed` 日志，
验收调整为实际CPU Manager编译路径 + CPU/CUDA eager 状态/采样一致性；GPU learner
编译由后续真实短程训练验收。此结果不能宣称完整sampler CUDA fullgraph受支持。
本地WBT motion与RL快速测试合计364 passed、1 skipped、49 deselected；Ruff通过。

CPU Manager编译与CPU/CUDA eager数值/采样一致性均已通过。第二次一次性验收在
通过断言后的清理阶段误调不存在的 `ManagerEnv.close()`，后续误调
`MotrixSimBackend.close()` 同样导致非零退出（close接口属于renderer）。已删除这两项
调用，沿用该一次性进程退出释放模拟器资源的方式；保留cleanup-failed日志。队列重新
完整验收后才放行冒烟训练。该修正仅影响诊断脚本，不涉及训练运行时。

2026-11-18 04:32:05 UTC：真实 Manager 编译 reset 与 CPU/CUDA eager 一致性验收
正常退出（两个 PASS 标记）；已启动4096环境、128 collector-step冒烟训练。
自动队列状态文件：`/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/historyfix-queue.status`；
配置/代码校验清单：服务器workspace的 `historyfix-manifest.json`。所有正式实验命名
`historyfix-{A,B,C,D}-seed{1,2}`；每组输出 `*-summary.json`，最终汇总为
`historyfix-search-summary.json`。实验尚在执行，不把队列启动记为质量达标。

2026-11-18 04:34:22 UTC：128-step GPU冒烟正常结束，A seed1正式训练启动。
本机一次性 `/tmp/apg-history-trigger/watch.py` 负责每60秒轮询上述队列状态；仅在
`COMPLETE bounded search and seed2 confirmation` 或 `FAILED; no further jobs launched`
出现时，通过 `codex queue --resume` 向当前会话提交分析任务。SSH错误只重试。
状态、锁与日志在 `/tmp/apg-history-trigger/`，锁与提交状态防止重复通知；提交异常需
核查日志和会话后重试。该工具不属于训练runtime，不启动额外实验。自动唤醒依赖本机
持续运行及Codex daemon可用；queue成功不等价于分析已经执行，应检查其输出中的resume警告。


### Historyfix 队列终态与分析（2026-11-18）

**结论：C seed1在5000 learner updates首次观测到40/40，启动至checkpoint为1392.1s
（23m12.1s）；尚未实现连续40/40或独立seed的40/40复现。C seed2在4474、5000 updates
均为39/40，但两次失败动作不同。四组加一次seed2的预算已经用完，本次分析没有追加训练。**

#### 终态、证据和计时口径

服务器 `historyfix-queue.status` 在 `2026-11-18 07:43:04.775477 UTC`
（北京时间15:43:04）写入 `COMPLETE bounded search and seed2 confirmation`。
五个正式容器 `apg-historyfix-{A,B,C,D}-seed1`、`apg-historyfix-C-seed2` 均为
`exited, ExitCode=0`，最后一个于07:42:59 UTC退出。不是仅凭DONE字符串判断成功。
最终search-summary与五个独立summary逐项相等；25份eval结果均有40个不重复动作，
成功数、checkpoint SHA256与summary一致，成功动作均完成全部帧且truncated、未terminated。
早期fullgraph/cleanup诊断失败日志属于前面的验收尝试，不是这五个正式实验的终态。

评估沿用相同脚本、4进程、每动作从首帧到尾帧一次rollout、无重试、play无观察/重置噪声。
训练seed为1或2，评估固定seed=2026；成功判据是 `truncated and not terminated`，并断言
没有提前truncation。results的“no hard termination”描述是成功要求，脚本未关闭height/tilt条件。
本报告中的“首次”均指**五个已评估checkpoint中首次观测到**，不能定位两个评估点之间
真正首次达标的update，也不使用随机中间起点的训练episode代替完整动作评估。

时间由每组docker启动前的 `*-started.txt` 到checkpoint文件mtime计算，包含启动与初始化，
不包含后面的离线评估。每组容器DONE还包含五轮评估，不能拿它当达标训练耗时。
xmcli没有started.txt，使用其status中的02:59:10 UTC启动时间和相同checkpoint mtime。
TensorBoard标量取 `(checkpoint_step-640, checkpoint_step]` 内日志点的算术平均；
采样CDF/history是checkpoint瞬时状态，两者不能混作同一时刻的数值。

原始证据仍在服务器 `/mnt/gfs/home/huangjiaxi/data/apg-async-20261117/`；
源码配置清单在 `/mnt/gfs/home/huangjiaxi/apg-async-20261117/historyfix-manifest.json`。
本地只读证据归档在 `outputs/historyfix-20261118-analysis/`，包含25份eval、训练日志、
资源JSON、summary、基线checkpoint/TensorBoard补充统计、评估源码与容器退出码。
未上传本机Instinct源码或数据，未启动额外服务器作业。

#### 四组、基线与seed2

mix顺序为uniform/return/history。A/C为0.5/0.4/0.1，B/D为0.8/0.1/0.1。
A/B每round 5次WM梯度更新，C/D为10次；均为5000 learner updates上限。
源码manifest确认A/C使用同一任务源码、B/D使用同一任务源码，其余所列runtime文件相同。

| 实验 | 2974 updates | 3474 | 3974 | 4474 | 5000 | 首次≥39：updates /启动墙钟 | 首次40：updates /启动墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| xmcli seed1 | 34 | 未评估 | 34 | 未评估 | 34 | 未达到 | 未达到 |
| A seed1 | 35 | 33 | 37 | 38 | 39 | 5000 /22m45.4s | 未达到 |
| B seed1 | 33 | 34 | 36 | 36 | 36 | 未达到 | 未达到 |
| C seed1 | 37 | 38 | 39 | 38 | **40** | **3974 /18m50.7s** | **5000 /23m12.1s** |
| D seed1 | 34 | 37 | 37 | 39 | 38 | 4474 /21m48.9s | 未达到 |
| C seed2 | 36 | 38 | 38 | 39 | 39 | 4474 /22m30.3s | 未达到 |

所有组 `held_two=false`。C seed1的后段为39→38→40，40/40仅出现在最后一个点；
C seed2有连续两个39/40，但4474失败 `fallAndGetUp1_subject4`，5000失败
`fallAndGetUp1_subject1`。因此选优C是本次有限搜索的结果，不是稳定40/40的证明。

最终checkpoint耗时：xmcli2378.0s（39m38.0s）、A1365.4s、B1393.1s、C1392.1s、
D1454.2s、C seed2 1493.3s。C比xmcli表面上短约41.5%，但初始128-step panel时间
分别为1m16s和17m35s，差额约979s；总时间差986s。相同7500→12564 collector-step
后段分别为517.1s和519.7s。**稳态几乎一样；不能将启动窗口差异称为history修复的吞吐收益。**
共享编译缓存和基线stage-probe入口是混杂因素；现有证据只定位到初始化窗口，不能精确把
该窗口的每一秒归因为编译。C相对A增加WM训练，最终checkpoint仅多26.7s（约2%）。

#### 失败动作与稳定性

以下为5000-update完整动作失败位置，格式为执行帧/总帧：

| 实验 | 失败动作 |
| --- | --- |
| xmcli | dance2_subject1 6057/11284；fallAndGetUp1_subject4 3438/8410；fallAndGetUp2_subject2 6368/8195；fight1_subject2 2994/12244；fightAndSports1_subject1 1224/12257；fightAndSports1_subject4 971/12257 |
| A | run1_subject5 5639/11890 |
| B | fallAndGetUp1_subject5 5577/8410；fallAndGetUp2_subject3 7155/8195；fightAndSports1_subject1 1214/12257；jumps1_subject2 10544/12222 |
| C seed1 | 无，40条均完成 |
| D | fallAndGetUp1_subject1 1830/8410；walk3_subject3 9145/12330 |
| C seed2 | fallAndGetUp1_subject1 1812/8410 |

C seed1在3974仅失败fallAndGetUp2_subject3；4474又新增fallAndGetUp1_subject4失败。
最终平均动作完成比例为A98.69%、B96.24%、C100%、D97.40%、C seed2 98.04%。
seed2虽连续39，平均完成比例从4474的98.69%降到98.04%，说明“成功数相同”不等于
剩余动作更接近完成。D与C seed2在同一fallAndGetUp1_subject1附近（1830/1812帧）失败，
提示应优先检查该接触/起身阶段；目前没有逐接触轨迹证据，不能直接归因于接触模型。

#### Return/history sampling

xmcli三个checkpoint中8842个bin的history全部相等，min=max=0.00011309658，
证实其history保持初始化均匀值。修复后的五组history均非均匀，min约1e-9、max为
0.182–0.303；实现每次rebuild执行 `history = .999*history + .001*soft` 后归一化并混合。
这验证了历史项更新缺口已修复，不等价于已经找到最优混合权重。

| 5000-update瞬时状态 | 归一化entropy | ESS=1/Σp²（bins） | 最大bin概率 | 最大bin |
| --- | ---: | ---: | ---: | --- |
| A | 0.892 | 233.1 | 5.53% | fallAndGetUp1_subject5，frame1500 |
| B | 0.916 | 185.8 | 5.54% | dance1_subject1，frame5300 |
| C seed1 | 0.689 | 9.1 | **32.69%** | dance2_subject4，frame2950 |
| D | 0.938 | 480.1 | 2.27% | fallAndGetUp1_subject1，frame6450 |
| C seed2 | 0.832 | 43.1 | 14.70% | dance1_subject1，frame5300 |

C seed1在5000时给dance2_subject4整条动作34.11%概率，该动作完整评估已通过；
seed2给最终失败的fallAndGetUp1_subject1整条动作仅3.05%。最大采样bin并不直接等于
当前完整评估的失败动作。return来自不同起始帧和episode长度，历史项还会滞后，必须与
完整动作失败位置联合看待。C末段窗口平均top1为35.9%（seed2为37.3%），A为29.2%，
B/D约5.9%/6.8%；A最后瞬时top1却只有5.53%，说明分布随训练明显变化。
较高均匀占比确实缓和集中，但本次B/D最终成功数低于对应A/C，不能凭高entropy推断质量更好。

**推断：** 修复后的困难区间采样有助于减少动作失败；过度集中和历史滞后可能参与后段
回退。但只有四组seed1和一个C seed2，尚不能把每次失败的出现/消失因果归于某个bin权重。

#### train/* 与世界模型

下表均为5000-update附近最后640 collector steps的日志窗口均值；loss是当前采样分布的
训练指标，不是固定held-out数据上的泛化误差。

| 实验 | WM loss / final loss | actor loss | actor grad norm | critic loss | imagined length | imagined reward | rollout return / ep_len |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| xmcli | .08236 / .13708 | -8.005 | .901 | .00166 | 31.855 | .07760 | 33.07 /451.2 |
| A | .08397 / .14434 | -7.862 | .917 | .00168 | 31.872 | .07676 | 34.69 /474.8 |
| B | .07177 / .12321 | -8.065 | .785 | .00166 | 31.866 | .07933 | 33.01 /445.4 |
| C seed1 | .05309 / .08900 | -7.939 | .898 | .00152 | 31.908 | .07695 | 33.85 /474.4 |
| D | .04236 / .07163 | -8.120 | .902 | .00156 | 31.892 | .08002 | 35.78 /470.2 |
| C seed2 | .04306 / .07206 | -7.880 | 1.019 | .00151 | 31.912 | .07669 | 32.83 /453.5 |

10次WM更新相对5次降低了本次窗口的模型loss，C也改善完整评估。但D的WM loss更低、
imagined reward和rollout return更高，却只有38/40；C seed2的模型loss也低于seed1，
仍未到40/40。各组imagined length都接近32、critic loss都约0.0015–0.0017，
这些指标不能作为长动作稳定性的充分证据。最终平均根高度误差仅约1.86–2.13cm，
也掩盖了稀有阶段的提前失败。这里未提供动作阶段分层的held-out WM误差，不能宣称
“世界模型精度已足够”或“只剩policy问题”。

#### 资源利用率与瓶颈

资源文件各含300个样本，采样从日志出现首个128-step panel后开始、间隔约1秒；
它们是训练早期局部窗口，不是全程平均。显存约20156MiB（基线20300MiB）。
后两列来自最终checkpoint前640-step TensorBoard窗口，与资源采样不是同一时间段。

| 实验 | GPU利用率均值 | 合计CPU核 | collector CPU核 | 末段collector自身ms/step | learner侧collect段ms/step | learn段ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xmcli | 41.25% | 14.07 | 13.48 | 96.51 | 48.49 | 51.42 |
| A | 37.13% | 14.35 | 13.76 | 95.01 | 47.43 | 50.65 |
| B | 38.82% | 14.32 | 13.73 | 96.14 | 48.67 | 50.75 |
| C seed1 | 40.94% | 14.04 | 13.39 | 96.50 | 41.06 | 59.53 |
| D | 41.69% | 14.08 | 13.43 | 104.00 | 47.20 | 61.30 |
| C seed2 | 41.42% | 13.89 | 13.26 | 102.23 | 44.83 | 61.36 |

learner本身约0.59–0.65 CPU核，collector使用分配16线程中的约13.3–13.8核。
发布约2.9ms/step、collector权重加载约1.4ms/step，末段weight-version-lag均值1.2、
ring-fill均值0.4。`collect`段包含borrow等待、transport、replay写入及normalization等，
不能把41–49ms全说成纯等待，也不能把异步collector自身时间与learner时间直接相加。

**推断：** 这些数据更支持collector/host供给限制吞吐，而不是GPU算力耗尽；WM次数翻倍
只把learn段增加约9–11ms，GPU仍约41%，符合部分新增学习工作被采集间隙吸收的情形。
现有证据不足以将剩余host开销精确分摊到仿真、reset/sampler与数据搬运，后续若获批
应先做这些分段的profiling，再决定增加CPU并行度或改调度；本次未启动新profiling训练。

#### 当前可采用的结论

1. 保留history更新修复；A相对xmcli在同一5000-update预算由34提升至39，支持继续使用修复，
   但尚无多seed对照可量化独立因果收益。
2. C（0.5/0.4/0.1 mix、10次WM更新）是本次最佳候选：seed1单次40、seed2连续39。
   不将其宣传为稳定40/40，也不把更均匀的B/D提升为默认。
3. 后续研究优先级是fallAndGetUp的失败阶段、采样集中/滞后与固定验证分布上的WM误差；
   这些是待检验方向，不是已确认根因。本次预算已结束，不追加seed、checkpoint或训练。

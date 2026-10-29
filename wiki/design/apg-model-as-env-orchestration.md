# APG Model-as-Env 编排设计

## 摘要

把 `motrix.apg` sync 训练从一条硬编码串行链

```
collect(真实 env) → update(){ wm_update×N → policy_update(){ 手写想象循环 → 解析 reward/termination → policy+critic 梯度 } }
```

重构为四个各自独立、可复用的阶段：**collect（真实环境收集）/ fit（世界模型监督拟合）/ imagine（想象环境上的标准 rollout）/ update（policy + critic 更新）**。核心手段是 **Model-as-Env**：`WorldModel` 与任务侧解析头（`ApgTask`）封装成一个可微「想象环境」`ImaginedEnv`，其 `step()` 字段形状对齐 `ArrayEnvState` 契约，从而让「想象」从埋在 `policy_update()` 里的特殊分支变成一个普通的 rollout 循环。

## 背景与动机

### 前提澄清：WM 只预测 state

一个常见误解是世界模型会预测 `state / observation / reward / termination`。实际不是：`WorldModel.step()`
（`motrix_rl_builtin/src/motrix_rl_builtin/apg/networks.py:141`）只预测 next floating-base state 的
frame-local 增量（`d_position / d_rotation / d_lin_vel / d_ang_vel / d_joint_pos / d_joint_vel`）并积分；
`obs / reward / soft_termination / goal / critic_features` 全部由任务侧**可微解析头**
（`ApgTask` 协议，`motrix_rl_builtin/src/motrix_rl_builtin/apg/contracts.py:53`）在想象出的 state 上计算。
WM 损失只有一个 masked 的 state-increment MSE（`apg/learner.py:52-77`）。

这反而是 Model-as-Env 的最好论据：reward / termination / obs 本来就是 state 的纯可微函数，
把 WM 与解析头拼起来就得到一个**完整、可微的「想象环境」**——它天然能以 env 的姿态被 rollout。

### 现状痛点

| # | 痛点 | 位置 |
|---|------|------|
| 1 | 想象循环硬编码在 `policy_update()` 内部，不可复用 | `apg/learner.py:332-469`（`_imag_step` `:314-330`） |
| 2 | WM 拟合与策略更新耦合在 `update()` 单方法里，顺序与 `wm_only_iters` 门控写死 | `apg/learner.py:475-507` |
| 3 | 想象 rollout 前向只算一次，但为 actor 目标与 critic 目标**各自重复组装**（flatten/stack、`critic_features` 拼接两遍） | `learner.py:360-368` vs `:407-428`，`critic_features` 在 `:393-394` 与 `:422-428` 两处调用 |
| 4 | action 采样 + obs 历史平移 + 噪声 + clamp 逻辑在 4 处重复（sync collect、async `collect_blocks`、两个 play 循环） | `learner.py:221-229`、`async_train.py:172-180`、`sync/train.py:146-153`、`trainer.py::PlayRunner.run` |
| 5 | 编排可读性差：训练循环调用者看不出「env 计算 → WM 更新 → WM 预测 → RL 更新」的真实结构 | `apg/sync/train.py:159-208` 只有 `collect()` + `update()` 两个黑盒 |

## 方向对比

### 方向 A：只抽公共助手，`update()` 保持整体

抽出重复的采样/历史/噪声逻辑为 helper，其余不动。

- 优点：diff 最小、风险最低。
- 缺点：只解决痛点 4；想象仍埋在 `policy_update()`，WM 拟合与策略更新仍耦合，编排结构不变。

### 方向 B：Model-as-Env + learner 阶段拆分（选定）

新增 `ImaginedEnv` + `ImaginedTrajectory` + `imagine_rollout`；`policy_update()` 改为消费**单份缓存的想象轨迹**；`ApgLearner` 拆出 `collect / fit_wm / imagine_and_update_policy`；sync 循环显式三阶段编排；`update()` 保留为薄兼容壳。

- 优点：痛点 1-5 全部解决；爆炸半径可控（仅 sync 路径，async 不动）；「同一 step 契约」落在字段形状层。
- 缺点：必须精细保留 autograd 边界（想象环境返回的 tensor 后续要被微分）；`soft_termination` 是连续 hazard 而非 bool `done`，契约需显式声明差异。

### 方向 C：字面统一 env 协议（否决）

让 `ImaginedEnv` 实现与 `ArrayEnv` 完全一致的 `step()/reset()`，返回 detached numpy 式 obs 与 hard bool
`done`，真实与想象跑同一个 `rollout(env, ...)`。

- 否决理由：想象需要**可微**的 `reward` 与 `soft_termination`；强行套 `ArrayEnvState` 语义（detach obs、
  bool done）会切断 `obs_t → obs_hist → pi → action_{t+1} → reward_{t+1}` 的梯度链，改变训练行为；
  要保留梯度就得再加一层「梯度注入 wrapper」，等于把刚解开的耦合重新加回去。真实收集
  （numpy 进出、`no_grad`、env 自带观测噪声）与想象（torch 进出、保图、解析 obs + 显式噪声）的
  语义差异是**本质的**，不追求字面统一。

**结论**：契约对齐做到字段形状层（`obs / reward / terminated / truncated`），并以两处显式声明差异收束：
字段是带 autograd 图的 tensor；`terminated` 是可微 soft hazard 而非 hard bool。

## 目标

- 新增 `apg/imagined_env.py`：`ImaginedEnv`（可微想象环境）、`ImaginedTransition` / `ImaginedTrajectory`
  数据结构、`imagine_rollout()` 标准想象循环。
- 新增 `apg/rollout.py`：`shift_history()` / `select_action()` 共享采样助手，消除 4 处重复。
- `ApgLearner` 拆出 `apply_lr` / `fit_wm` / `imagine_and_update_policy` 三个独立阶段方法；
  `policy_update()` 内部改为「构造 ImaginedEnv → imagine_rollout → `_policy_objective` + `_critic_update`」，
  想象轨迹只组装一次、两段共用；`update()` 保留为薄兼容壳（外部调用者与测试不破）。
- `apg/sync/train.py::_run_loop` 显式三阶段编排，utd 累加器、warmup、日志与 checkpoint 行为不变。

## 非目标

- 不做字面统一 rollout 协议（方向 C 已否决）。
- 不改 async trainer 编排与双 GPU WM worker（后续可免费复用 `ImaginedEnv` 与共享助手，见文末展望）。
- 不改 `ApgTask` 协议、replay buffer、配置面（`ApgCfg`）与 checkpoint 格式。

## 设计

### 重构基准：现 `policy_update()` 的结构

```
_sample_valid_starts(pi_batch)            # 从 replay 采样合法起点 + 历史（learner.py:287）
冻结 wm / critic（:347-348）
t=0: no_grad 下用 head.obs(history) 构造 obs_hist，add_noise 一次（:339-346）
for t in range(horizon):                  # 想象循环（:360-368）
    action = pi(obs_hist, goal)           # 原始输出：无噪声、无 clamp（_imag_step :324）
    state = wm.step(state, action, ξ̂)
    obs = head.add_noise(head.obs(state, action, ξ, f+1))
    obs_hist ← 平移; goal ← head.goal(f+1, ξ)
段一（actor）: 解析 rewards / soft dones → alive/continuation/discount → objective + 末端 critic bootstrap（:378-402）
段二（critic）: 重组 critic_states/critic_frames/critic_goals → critic_features → λ-returns 目标 → minibatch 回归（:407-460）
解冻 wm / critic（finally，:403-405）
```

### ImaginedEnv（`apg/imagined_env.py`，新增）

薄封装，**复用**现有 head 方法（`head.obs / add_noise / goal / reward / soft_termination /
critic_features / normalize_context`）与 `WorldModel.step`，不重写任何任务语义——与现
`policy_update()` 的解析等价性是结构性的，不是碰巧。

```python
@dataclass
class ImaginedTransition:
    """一步想象转移；字段形状对齐 ArrayEnvState，语义差异见下。"""
    state: State                    # post-step 状态元组 (p,R,v,w,q,qd)
    frames: torch.Tensor            # (B,) long
    action: torch.Tensor            # 产生本步的 action（保图）
    previous_action: torch.Tensor   # 本步之前的 action（保图）
    obs: torch.Tensor               # 加噪后的本体感 obs（保图）
    goal: torch.Tensor              # (B, goal_dim)
    reward: torch.Tensor            # 解析 reward（保图）
    terminated: torch.Tensor        # soft termination hazard ∈ [0,1]（保图，非 bool）
    truncated: torch.Tensor         # 全 False，仅字段形状对齐
    critic_input: torch.Tensor      # cat(critic_features, goal)，带图：末端 bootstrap 穿过它求导；critic 回归侧自行 detach

@dataclass
class ImaginedTrajectory:
    """整段想象轨迹：状态轴 H+1（s_0..s_H），转移轴 H。"""
    states: list[State]             # 长度 H+1
    frames: torch.Tensor            # (H+1, B)
    goals: torch.Tensor             # (H+1, B, goal_dim)
    actions: torch.Tensor           # (H, B, N)    actions[t] 引起 s_t → s_{t+1}
    previous_actions: torch.Tensor  # (H, B, N)    previous_actions[0] == start["a_prev"]
    rewards: torch.Tensor           # (H, B)       actor 目标用（保图）
    dones: torch.Tensor             # (H, B)       soft hazard（保图）
    critic_input: torch.Tensor      # (H+1, B, D)  bootstrap / 值目标用（no_grad）

class ImaginedEnv:
    def __init__(self, wm: WorldModel, head: ApgTask, cfg: ApgAgentCfg, *, device): ...

    def reset(self, start: dict, history: dict) -> None:
        # 由 replay 采样起点（_sample_valid_starts 的输出）构造 t=0：
        #   state=unflatten(start["s"]), frames=start["f"], prev_action=start["a_prev"]
        #   context=start["xi"], normalized_context=head.normalize_context(context)
        #   goal=head.goal(frames, context)
        #   obs_hist：no_grad 下 head.obs(history) + add_noise 一次（对齐 :339-346）
        #   初始 critic_input：no_grad 下 cat(critic_features(state,f,ξ̂), goal)（对齐 :408-428 的 t=0 项）

    def step(self, action: torch.Tensor) -> ImaginedTransition:
        # 调用方负责冻结 WM；action 是 policy 的原始输出
        #   next_state = wm.step(state, action, normalized_context)
        #   next_frames = frames + 1
        #   obs = head.add_noise(head.obs(next_state, action, context, next_frames))
        #   next_goal = head.goal(next_frames, context)
        #   reward = head.reward(next_state, next_frames, action, previous_action, context)
        #   terminated = head.soft_termination(next_state, next_frames)
        #   critic_input：cat(critic_features(next_state,...), next_goal)，保留图
        # 推进滚动状态 state/frames/previous_action/obs_hist/goal 后返回转移
```

与 `ArrayEnvState` 契约的对齐与差异：

| 字段 | ArrayEnv | ImaginedEnv | 说明 |
|------|----------|-------------|------|
| `obs` | numpy，env 侧噪声 | torch，**保图**，解析 + 显式噪声 | 梯度链 `obs→action` 必须保留 |
| `reward` | env 计算 | torch，**保图**，解析 head | |
| `terminated` | bool | torch ∈ [0,1]，**可微 hazard** | 消费端做 `cumprod(1-done)` |
| `truncated` | bool | 全 False | 想象 rollout 无时间截断 |
| `final_obs` / auto-reset | 有 | 无 | 想象按定长 horizon 展开 |

### imagine_rollout：标准想象循环

```python
def imagine_rollout(env: ImaginedEnv, pi: Policy, horizon: int) -> ImaginedTrajectory:
    # for _ in range(horizon):
    #     action = pi(env.obs_hist, env.goal)   # 原始输出，无噪声无 clamp，保图
    #     tr = env.step(action)
    #     累积进 trajectory
```

索引语义与现 `policy_update()` 严格一致：`rewards[t] = head.reward(s_{t+1}, f_{t+1},
actions[t], previous_actions[t], context)`，`previous_actions[0] == start["a_prev"]`。

**rollout 不做的事**：不包 `@torch.no_grad()`（整段要建图）；不 detach `obs/reward/terminated/action`；
想象里的 `pi(...)` 不加噪声、不 clamp（与采集侧 `select_action` 的关键差异）。

### 共享采样助手（`apg/rollout.py`，新增）

```python
@torch.no_grad()
def shift_history(obs_hist, obs, fresh=None) -> torch.Tensor:
    # 平移 obs 历史；fresh（新 episode）时用当前 obs 填满整段

@torch.no_grad()
def select_action(pi, obs_hist, goal, action_low, action_high, action_noise) -> torch.Tensor:
    # clamp(pi(obs_hist, goal) + noise * randn_like(...), low, high)
```

消费方与各自保留的差异：

| 消费方 | 用法 | 保留在外的差异 |
|--------|------|----------------|
| `ApgLearner.collect`（sync） | 两个助手都用 | `pi.obs_norm/goal_norm.update` 留在 collect（async 采集有意不更新） |
| `async_train.collect_blocks` | 两个助手都用 | CUDA `.to()/.cpu()` 传输包装在外 |
| `sync/train.py::play`、`trainer.py::PlayRunner.run` | `shift_history` + 裸 `pi(...)` | play 无噪声、无 clamp |

想象循环**不消费**这两个助手——它保图、用解析 obs，与采集语义本质不同（方向 C 的教训）。

### ApgLearner 阶段拆分（`apg/learner.py`）

```python
def apply_lr(self, iteration=None) -> float:          # 三 optimizer 的 cosine LR（原 update 开头）
def fit_wm(self, *, defer_loss_sync=False) -> tuple[Tensor, Tensor]
                                                      # wm_update × wm_batches；返回末批 (loss, error)。
                                                      # async 传 defer_loss_sync=True：除末批外逐批保留
                                                      # loss tensor 供迟到指标回读；默认立即 detach
def imagine_and_update_policy(self) -> dict[str, float]:  # 包 policy_update()，产出指标 dict
def update(self) -> dict[str, float]:                 # 薄兼容壳：apply_lr + fit_wm + (门控) policy + iter+=1
```

`policy_update()` 公开签名不变（`-> tuple[Tensor×5]`，async trainer 按 `policy[0..4]` 索引）。内部：

```python
def policy_update(self):
    _set_requires_grad(self.wm, False); _set_requires_grad(self.critic, False)
    try:
        start, history = self._sample_valid_starts(self.cfg.pi_batch)
        env = ImaginedEnv(self.wm, self.head, self.cfg, device=self.device)
        env.reset(start, history)
        traj = imagine_rollout(env, self.pi, self.cfg.horizon)      # 单份轨迹
        loss, grad_norm, survival, reward_sum = self._policy_objective(traj)
        value_loss = self._critic_update(traj)
    finally:
        _set_requires_grad(self.wm, True); _set_requires_grad(self.critic, True)
    ...
```

- `_policy_objective(traj)` 复刻 `:378-404`：解析 reward/done 已在轨迹里；
  `alive/continuation/discount`、末端 `γ^H · survival · critic(traj.critic_input[-1])` bootstrap、
  opt_pi 的 `clip_grad_norm_(1.0)` + `clip_grad_value_(0.1)` + 条件 step 原样。
- `_critic_update(traj)` 复刻 `:407-460`：直接消费 `traj.critic_input (H+1,B,D)` 与
  `traj.rewards.detach() / traj.dones.detach()`——**不再重新 flatten/stack/拼 critic_features**
  （痛点 3 消除）；`critic.norm.update` 在估值前、`values[-1]` bootstrap、λ-returns 反推、
  `cache_critic_features` 与 minibatch 回归原样。

### sync 训练循环重连（`apg/sync/train.py::_run_loop`）

`learner.update()`（`:178`）展开为显式三阶段，其余（warmup collect `:165`、utd 累加器
`:173-177`、日志与 checkpoint 块）不动：

```python
lr = learner.apply_lr()
update_metrics = {"learning_rate": lr, **learner.fit_wm()}
if learner.iteration >= cfg.wm_only_iters:
    update_metrics.update(learner.imagine_and_update_policy())
learner.iteration += 1
```

### torch.compile

现 `_imag_step` 被 compile（`learner.py:165`）。`ImaginedEnv.step` 是有状态方法，不利 compile；
改为提供纯函数 `imagine_step(wm, head, state, obs_hist, prev_action, frames, context, ξ̂, goal)`，
`imagine_rollout` 按配置选择 eager 或 compiled 版本。`compile` 默认 `false`，正确性优先，
compile parity 作为 CUDA 上的跟进检查。

## 行为不变清单

以下每一条在重构前后必须逐位等价（回归验证的依据）：

1. WM/critic 在想象前冻结、`finally` 恢复，异常路径同样恢复（`:347-348, 403-405`）。
2. t=0 `obs_hist` 与 `initial_goal` 在 no_grad 下构造；`add_noise` 在 reset 时只加一次（`:339-352`）。
3. 想象动作 = 原始 `pi(obs_hist, goal)`，无噪声、无 clamp（`:324`）。
4. `obs` 保留对 `action` 的计算图（`obs_t → obs_hist → pi → action_{t+1}` 链），不得 detach。
5. `reward/terminated` 逐步由 `(next_state, next_frames, action, previous_action, context)` 计算；
   `alive/continuation/discount` 与末端 bootstrap 公式不变。
6. critic 目标用 `detached_rewards / detached_dones`（`:392, 435-437`）。
7. critic：`critic.norm.update` 先于估值、`values[-1]` bootstrap、λ-returns 反推、minibatch 回归、
   `cache_critic_features`、`run_mlp` amp 路径（`:429-460`）。
8. optimizer 步进与 clip 值：opt_pi `clip_grad_norm_(1.0)` + `clip_grad_value_(0.1)` + 有限才 step；
   opt_v `clip_grad_norm_(1.0)`。
9. `imagined_reward` 归一化 `/ horizon / (max_step_reward * ctrl_dt)`（`:462`）。
10. 采集侧全链路：obs 来源（env 产出、含噪声）、历史平移、norm 更新顺序、噪声+clamp、
    `env.step(action.detach().cpu().numpy())`、`buffer.add`、`wm.in_norm/out_norm.update`、
    `sim_steps += num_envs`、stats 追加（`:206-258`）。
11. warmup 量、utd 小数累加器、`sim_steps` 记账、iteration 自增、cosine LR 调度；
    `ApgProvider.total_env_steps` 的 sync 公式（`framework.py:44-46`）保持有效。
12. `_sample_valid_starts`（2× 过采样、`valid_imagination_start` 过滤、8 轮空转 fail-fast）不变。

## 实施顺序与验证

每步独立可合入、行为不变、有测试兜底：

| 步骤 | 内容 | 验证 |
|------|------|------|
| 1 | `apg/rollout.py` + 接入 collect / async collect_blocks / 两个 play | `pytest motrix_rl_builtin/tests/test_apg.py -k "collect or episode_length or terminations"`；`pytest tests/test_apg_integration.py -k quadruped_sync` |
| 2 | `apg/imagined_env.py` + 单测（形状；`rewards/actions/requires_grad is True`、`critic_input.requires_grad is False`；`dones∈[0,1]`；固定 WM+起点下与手写 head 循环逐项 parity） | 新增 `motrix_rl_builtin/tests/test_imagined_env.py` |
| 3 | `policy_update()` 改走 `ImaginedEnv/imagine_rollout`，拆 `_policy_objective/_critic_update` | `pytest ... -k "policy_update or imagined_reward"` + 前后数值等值（同 seed 短跑对比标量） |
| 4 | 拆 `apply_lr/fit_wm/imagine_and_update_policy`，`update()` 留薄壳 | `pytest ... -k update` |
| 5 | `_run_loop` 显式三阶段 | `pytest tests/test_apg_integration.py -k sync_trainer`（两个任务变体断言 `checkpoint["sim_steps"] == 6 * num_envs`） |

**数值前后等值**（最便宜的回归网）：同 seed 同配置短跑两遍（基线 vs 重构），断言前若干 iteration
的标量在 fp32 容差内相等：`wm_loss / wm_last_error / imagined_reward / imagined_survival /
policy_loss / value_loss / policy_grad_norm / sim_steps / completed_episodes / episode_length /
termination_fraction`。

**回归哨兵**（哪个指标守哪类错误）：

- `policy_grad_norm` —— 误 detach `obs/reward/dones`（链断而无不报错）。
- `value_loss` —— critic 输入/目标错位（H+1 vs H 索引、漏 `norm.update`）。
- `wm_last_error` —— replay 窗口 / normalizer 变更。
- `sim_steps` —— utd / warmup / 收集循环漂移。

## 风险与开放问题

1. **autograd 边界（最高风险）**：想象 env 返回带活图的 `obs/reward/terminated/action`；任何一处
   「顺手 detach」都会静默切断 `obs_t → action_{t+1} → reward_{t+1}` 梯度链，只表现为
   `policy_grad_norm/loss` 漂移而无报错。以 `requires_grad` 单测断言兜底（步骤 2）。
   **实现期确认的隐藏路径**：末端 bootstrap 的 `critic_features(s_H, ·)` 在原实现中于**带梯度的**
   最终想象状态上计算（`learner.py` 旧 :393-394），梯度穿过冻结的 critic 流回 WM/policy——因此
   `critic_input` 必须逐 step 带图计算，仅 `_critic_update` 入口整体 detach；统一 no_grad 会静默
   切掉这条路径。
2. **初始 obs_hist 的构造**：需要完整 replay 历史（`history["s"]/a_prev/xi/f`），`add_noise` 只加一次；
   `ImaginedEnv.reset` 在 no_grad 下复刻，不与采集侧噪声路径共享。
3. **soft hazard 不折成 bool**：策略目标在 `continuation/alive` 内消费可微 hazard，critic 消费
   `(1 - detached_dones)`；hard mask 会把终止梯度清零、改变目标函数。
4. **utd 小数记账**：累加器与 `sim_steps` 留在 collect / `_run_loop`，不受阶段拆分影响。
5. **torch.compile**：见上文；compiled parity 在 CUDA 上另行验证。

## async 扩展展望（已落地）

async trainer 的 `policy_update()` 调用的是同一个 `ApgLearner.policy_update()`，步骤 3 落地后自动
获得单份缓存轨迹；`collect_blocks` 复用 `shift_history/select_action`（步骤 1 已接入）。后续里程碑
把 async 循环里手写的 cosine 调度与内联 WM 批循环也收敛到共享阶段方法：`learner.apply_lr(iteration)`
与 `learner.fit_wm(defer_loss_sync=cfg.defer_wm_loss_sync)`——`defer_loss_sync=True` 时除末批外
逐批保留 loss tensor（async 迟到指标回读），默认 False 立即 detach（sync 直接上报浮点），两条路径
数值语义与各自原实现逐位一致。双 GPU WM-snapshot 路径（`wm_client/wm_worker`）保持在
`ImaginedEnv` 之外——它在 `policy_update()` 前换 `learner.wm` 权重，`ImaginedEnv` 经同一引用读到
新权重。

## 实施状态

已于 2026-10-28 在 supermimic worktree 按 5 步全部落地：

- 新增 `apg/rollout.py`（`shift_history` / `select_action`）、`apg/imagined_env.py`
  （`ImaginedTransition` / `ImaginedTrajectory` / `imagine_step` / `ImaginedEnv` / `imagine_rollout`）、
  `motrix_rl_builtin/tests/test_imagined_env.py`（6 项：形状/索引、与重构前手写循环逐项 parity、
  冻结模块下的梯度流、hazard 区间、噪声一次性契约、纯函数性）。
- `learner.py`：`collect` 改用共享助手；`policy_update` 改为「想象环境 + 单份缓存轨迹 +
  `_policy_objective` / `_critic_update`」；`update()` 拆为 `apply_lr` / `fit_wm` /
  `imagine_and_update_policy` 三个阶段方法 + 薄兼容壳；`torch.compile` 改编译纯函数 `imagine_step`。
- `sync/train.py::_run_loop` 显式三阶段编排；async `collect_blocks` 与两个 play 循环接入共享助手。
- 验证：同 seed 新旧实现 `collect → wm_update → policy_update → update ×2` 的 19 项指标
  （含损失、梯度范数、更新后网络权重、normalizer 统计、sim_steps）逐位一致；APG 全部 35 项测试通过。
- 实现期唯一设计修正：`critic_input` 逐 step 带图计算（上文风险 1）。
- 2026-10-28 后续里程碑完成（每步以 go2-walk-flat 与 g1-supermimic 的 `motrix.apg.smoke.yaml`
  冒烟训练为门禁）：
  - **async 阶段共享**：async 循环改用 `apply_lr` / `fit_wm(defer_loss_sync=...)`（见上文展望）；
    28 项 APG 测试全绿；async 冒烟双环境通过——`policy_lag` / `collector_fps` / `update_credit`
    等 async 指标与 2304 `sim_steps` 记账（warmup 16 + utd 0.4 × 36 步 × 64 envs）均正常。
  - **CUDA compile 等值**：RTX 5060 Ti、torch 2.7.0+cu128，同 seed 下 eager vs compiled 的
    13 项训练标量最大相对差 6.9e-06、更新后网络权重最大相对差 2.2e-04（纯 inductor 代码生成
    差异）；`algo.agent.compile=true` 冒烟双环境通过，标量与 eager 一致至 1e-4 量级。

# g1-supermimic: goal-conditioned LAFAN1 corpus tracking

`g1-supermimic` reproduces the G1 whole-body tracking task of [SuperMimic](https://arxiv.org/abs/2610.28480) (reference
implementation [UniSuperMimic](https://github.com/Motphys/UniSuperMimic)) on MotrixSim: the policy tracks the retargeted
LAFAN1 corpus frame by frame, observes future-reference goals, samples episode starts from SuperMimic's return-based
one-second-bin mixture, and ends each episode on the owning clip's final frame. The robot model, joint limits, and gains come
from the `UnitreeG129Dof` asset; the task owns the commands, observations, rewards, terminations, and episode behavior.

Compared with the built-in `g1-wbt-*` tasks: the policy observation is goal-conditioned proprioception (without the
current-frame reference joints), episodes end at the clip boundary instead of resampling mid-episode, and start sampling is
driven by return statistics instead of the failure-rate curriculum.

The task ships two training entries: `motrix.fastsac` (model-free) and `motrix.supermimic` (differentiable world
model; see the World-model training section below).

## Corpus preparation

The task reads `~/.cache/motrixlab/lafan/g1_npz` by default (one MotrixLab motion NPZ v1 per LAFAN1 clip at 50 Hz). The corpus
is not shipped with the package; build it once before first use:

```bash
uv run scripts/motion/download_lafan.py --all
```

This downloads the 40 retargeted G1 clips (30 fps CSV) from the HuggingFace dataset `lvhaidong/LAFAN1_Retargeting_Dataset`
and converts them to 50 Hz NPZ through forward kinematics (see [Motion file format](motion_format.md)). Set the environment
variable `SUPERMIMIC_MOTION_DIR` to point at other corpus directories; multiple entries join with `:`.

## Running

```bash
# Train (FastSAC, model-free)
mxlab train task=g1-supermimic/motrix.fastsac

# Train (SuperMimic differentiable world model)
mxlab train task=g1-supermimic/motrix.supermimic

# Small smoke test
mxlab train task=g1-supermimic/motrix.fastsac num_envs=64 algo.trainer.num_learning_iterations=100

# Play: sequential playback from corpus frame zero, crossing clip boundaries
mxlab play env=g1-supermimic num_envs=16
```

## Action space

`Box([-1, 1]^{29})`. Dimension $j$ maps to the joint-position target

$$
q_j^{target} = q_j^{default} + \alpha_j a_j,\quad \alpha_j = 0.25\,\frac{effort_j}{kp_j}
$$

where $q^{default}$ is the robot's `default` key pose (identical to the upstream stand pose: hips −0.312, knees 0.669, ankles
−0.363, elbows 0.6, shoulder roll ±0.2), $\alpha_j$ is per-joint from the model actuators' effort limits and position gains,
and targets are clamped to the joint limits.

## Observation space

| Observation | Policy | Value | Meaning |
| --- | ---: | ---: | --- |
| Joint positions relative to the default pose | 29 | 29 | rad, uniform noise in [−0.01, 0.01) |
| Joint velocities | 29 | 29 | rad/s, noise [−0.5, 0.5) |
| Root local angular velocity | 3 | 3 | rad/s (pelvis frame), noise [−0.2, 0.2) |
| Gravity direction (root frame) | 3 | 3 | noise [−0.05, 0.05) |
| Previous action | 29 | 29 | noise-free |
| Relative reference rotation 6D | 6 | 6 | current-frame reference root rotation relative to the robot root, noise [−0.05, 0.05) |
| Goal (future reference frames) | 730 | 730 | 10 frames × 73, see below |
| Current-frame reference joint positions/velocities | — | 58 | noise-free |
| Reference root position error (robot frame) | — | 3 | noise-free |
| Anchored tracking errors | — | 45 | 14 tracked-body position errors + root error (world frame) |

The policy group totals 829 dimensions and the value group 935; terms shared with the policy group appear noise-free in the
value group (privileged).

Each goal frame (73 dimensions) is expressed in the yaw frame of the current-frame reference root: root position delta (3),
relative root rotation 6D (6), root linear velocity (3), root angular velocity (3), reference joint positions (29), and
reference joint velocities ×0.1 (29). Future frame $i$ is $\min(step+i,\ \text{clip final frame})$ — clamped at the owning
clip's end, never crossing clips.

## Reward design

The reward is SuperMimic's six tracking kernels plus two regularizers; the total return is weighted by the control timestep
`ctrl_dt = 0.02` (matching upstream `* dt`), so perfect tracking earns at most $5 \times 0.02 = 0.1$ per step:

| Reward term | Computation | Design purpose |
| --- | --- | --- |
| `motion_relative_body_position_error_exp` | $\exp(-\bar{e^2}/0.3^2)$ over the per-frame anchored 14 tracked bodies, weight 1.0 | drive overall body-configuration tracking |
| `motion_relative_body_orientation_error_exp` | same kernel over rotation distance, $\sigma=0.4$, weight 1.0 | drive pose tracking |
| `motion_global_body_lin_vel` | world-frame linear-velocity error kernel, $\sigma=1.0$, weight 1.0 | match body velocities |
| `motion_global_body_ang_vel` | world-frame angular-velocity error kernel, $\sigma=3.14$, weight 1.0 | match angular momentum (flips, jumps) |
| `motion_global_ref_position_error_exp` | root position error kernel, $\sigma=0.3$, weight 0.5 | weakly constrain the global root position |
| `motion_global_ref_orientation_error_exp` | root rotation error kernel, $\sigma=0.4$, weight 0.5 | weakly constrain the global root heading |
| `action_rate_l2` | $\lVert a_t - a_{t-1}\rVert_2^2 \times (-0.1)$ | discourage action churn |
| `limits_dof_pos` | 0.9 soft-limit violation × (−2.0), uncapped | keep joints feasible |

Anchoring semantics: every control step re-anchors the reference body targets onto the robot's current root position
(planar) and the reference root height (yaw aligned), matching the per-step anchoring of the individual rewards, so planar
drift does not instantly inflate the errors.

## Termination conditions

| Type | Status | Condition | Meaning |
| --- | --- | --- | --- |
| Reference height deviation | `terminated` | root z differs from the reference by > 0.25 m | tracking failure (fall, hang) |
| Reference tilt deviation | `terminated` | root attitude $R_{zz}$ differs by > 0.8 | tracking failure (tipped over) |
| Clip end | `terminated` | stepped onto the owning clip's final frame | success; the start sampler treats it like a truncation |
| Joint limit / NaN | `terminated` | violation > 0.5 rad or divergence | numerical safety net, never fires on healthy rollouts |
| Time limit | `truncated` | 10 s (500 control steps) | time-out truncation |

Clip end is handled as `terminated` in the RL buffer (see the upstream differences below); the return sampler bootstraps it
like a truncation.

## Reset logic and randomization

Deterministic part: the sampled episode start frame determines the whole state — its root pose, root velocity, joint
positions, and joint velocities are teleported into the simulation.

| Randomized quantity | Sampling | Timing | Purpose |
| --- | --- | --- | --- |
| Start frame | return-based one-second-bin mixture: 50% uniform + 40% softmax(−25 G_b/G_max) + 10% past-softmax retention | every episode reset | curriculum: practice hard bins more, relax after mastery |
| Ground friction | uniform scale factor in [0.3, 1.2) applied to the nominal friction vector | every episode reset | physical domain randomization; the sample rides the simulation context |
| Torso CoM offset | uniform [−0.025, 0.025) / [−0.05, 0.05) / [−0.05, 0.05) m (torso_link) | every episode reset | physical domain randomization; the sample rides the simulation context |
| Root position noise | uniform [−0.05, 0.05) m (x/y), [−0.01, 0.01) m (z) | reset | initial-state randomization |
| Root orientation noise | Euler uniform [−0.1, 0.1) / [−0.1, 0.1) / [−0.2, 0.2) rad | reset | initial-state randomization |
| Root linear-velocity noise | uniform [−0.5, 0.5) / [−0.5, 0.5) / [−0.2, 0.2) m/s | reset | initial-state randomization |
| Root angular-velocity noise | uniform [−0.52, 0.52) / [−0.52, 0.52) / [−0.78, 0.78) rad/s | reset | initial-state randomization |
| Joint position noise | uniform [−0.1, 0.1) rad, clamped to limits | reset | initial-state randomization |
| Observation noise | see the observation table | every observation | observation randomization |

Bin return statistics: each bin keeps the EMA estimate $G_b \leftarrow 0.8\,G_b + 0.2\,G_{episode}$; the episode return is the
discounted sum (γ = 0.99), failures (height/tilt) receive no remaining value, and clip-end/time-out episodes bootstrap with
$\gamma^L \bar{r}_L/(1-\gamma)$; $G_{max} = 5 \times 0.02 / (1-\gamma) = 10$. Sampling statistics are published as
`supermimic_*` metrics.

The physical-domain samples (friction scale + torso CoM offset, 4 dimensions) are also written into the motion command's
simulation-context buffer, conditioning the world model and critic (a subset of upstream xi; play mode falls back to the
nominal values). Joint masses, inertia, actuator gains, and delays stay at the model defaults.

Observation noise is listed in the observation table. Play mode disables start sampling and all reset noise, falls the
physical-domain randomization back to nominal (friction scale 1, zero CoM offset), and plays the corpus sequentially from
frame zero, crossing clip boundaries.

## Configuration

Frequently used fields of `SuperMimicMotionCommandCfg` (via
`mxlab train task=g1-supermimic/motrix.fastsac commands.motion.<field>=...`):

- `num_goal_frames` (default 10): number of future frames in the goal observation;
- `supermimic_gamma` (default 0.99): discount for the bin returns; keep it equal to `algo.agent.gamma`;
- `supermimic_sampling` (default `true`): disable to fall back to uniform bin sampling (play mode disables it
  automatically);
- `sim_reset.dynamics.friction_range` (default (0.3, 1.2)) and `sim_reset.dynamics.com_offset_noise` (default
  (0.025, 0.05, 0.05)): physical-domain randomization ranges; degenerate single-point ranges disable the item.

## World-model training (motrix.supermimic)

`motrix.supermimic` ports the SuperMimic world-model trainer: the dynamics is a learned latent increment model (predicting
root-frame-local state increments and integrating them), while the reward, soft-termination, observation, and goal heads are
differentiable analytic functions (`motrix_rl_builtin.supermimic.mdp`; its forward kinematics matches MotrixSim FK and its
features match the manager kernels term by term). The goal-conditioned policy trains on imagined rollouts (λ-return critic
bootstrap, soft-termination discounting); collection and updates alternate at the configured `utd`, the first `warmup_steps`
steps only collect, and the first `wm_only_iters` iterations train only the world model. The environment stays on the NumPy
stack: the motion command publishes the flattened state, frame index, and simulation-context buffers every step, and the
trainer builds the torch MDP from those NumPy surfaces (corpus clip, robot model and default pose, control rate); the plugin
never imports task packages.

```bash
mxlab train task=g1-supermimic/motrix.supermimic
mxlab train task=g1-supermimic/motrix.supermimic algo.agent.goal_frames=10
```

Keep `algo.agent.goal_frames` equal to `commands.motion.num_goal_frames` (both default to 10); hyperparameters mirror the
upstream defaults and the learning rate decays on a cosine schedule to 3%. The world model conditions on the normalized
simulation context (friction scale + torso CoM offset), matching the collection-time physical-domain randomization.

## Per-clip evaluation

`scripts/eval_supermimic.py` runs deterministic full-clip evaluation of a trained checkpoint (the upstream evaluate port,
without render / sim2sim): one environment lane per LAFAN1 clip, every episode pinned to its clip's first frame, with
observation, exploration, reset, and physical-domain noise disabled (the `for_eval()` variant), rolling each lane until it
finishes its clip (success) or fails a height/tilt check. It reports:

- `norm_return`: mean over clips of the return divided by `(clip length × per-step reward ceiling)`;
- `norm_return_framew`: the frame-weighted global normalized return;
- `n_failed` / `failed`: the failed-clip list (ending at the clip boundary is a success);
- `per_clip`: per-clip normalized return, survived steps, and failure flag.

```bash
# Evaluate one checkpoint
python scripts/eval_supermimic.py --ckpt runs/g1-supermimic/<run>/checkpoints/latest.pt

# Or resolve the run's recorded policy through the manifest
python scripts/eval_supermimic.py --run_dir runs/g1-supermimic/<run>
```

Results go to `<checkpoint>_eval.json`, and one row is appended to `eval.csv` next to the checkpoint. The policy
hyperparameters (goal frames, history, network widths) ride the checkpoint, so the evaluator rebuilds the network
automatically.

## Differences from upstream UniSuperMimic

- The differentiable world-model trainer is ported (`motrix.supermimic`, synchronous single process); `motrix.fastsac`
  remains available as the model-free baseline;
- physical-domain randomization is implemented as MotrixLab reset terms: the ground geom's friction vector is scaled by a
  factor (upstream sets the robot collision geoms' absolute friction) plus the torso CoM offset, and the samples ride the
  simulation context for world-model/critic conditioning; upstream xi's reset joint offsets and their coupling into action
  targets and observations are not implemented; joint masses, inertia, actuator gains, and delays stay at model defaults;
- clip end is implemented as a termination term: the FastSAC buffer does not bootstrap it (only the start sampler does); the
  world-model imagination uses the soft termination probability, as upstream;
- the relative-rotation 6D uses the first two matrix rows (row-major), equivalent to upstream's first two columns (a
  continuous 6D representation of the same rotation);
- rewards compare against the current frame (evaluate-then-advance); upstream compares against the next frame.

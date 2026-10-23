# SONIC G1 Tracking

SONIC is a manager-based Unitree G1 motion-tracking task whose dedicated actor is
built by the FastSAC `sonic` PolicyVariant. The policy combines a windowed
proprioceptive history with G1, SMPL, and hybrid (teleop) reference streams
routed by a three-value multi-hot encoder mask in the upstream column order
`[g1, teleop, smpl]`: smpl-native rows also activate the g1 encoder and, with
probability 0.5, the teleop encoder, matching the upstream legacy sampler.
The hybrid stream mirrors the upstream teleop encoder input: a multi-future
lower-body block (12 leg joint positions and velocities on the G1 stride
grid) plus a current-frame tail of VR-style 3-point targets (both wrists plus
a torso offset point, positions and quaternions in the reference-anchor
frame) and the anchor orientation. This temporal depth is the public contract
of `g1-sonic`; training, LAFAN data, and small-scale validation all use the same
environment.

The task recipe inlines `algo.variant` in
`configs/task/g1-sonic/motrix.fastsac.yaml`. `model.num_future_frames` defines
both the environment policy-observation width and the actor input width;
`model.encoder_index: false` drops the trailing three-value encoder mask from
the packed observation: every row then routes through the g1 encoder and
observation normalization covers all dimensions uniformly (no binary
passthrough channels remain). `auxiliary` holds weights for the five
upstream-named auxiliary losses
(`g1_recon`, `g1_smpl_latent`, `g1_teleop_latent`, `teleop_smpl_latent`,
`reencoded_smpl_g1_latent`), which FastSAC weights and logs as `aux_*`
metrics. `g1_control_decoder_hidden_dims` does not appear in the training
configuration: the training actor omits that decoder and uses the FastSAC
policy head.

## Observations and actions

The `g1-sonic` policy observation has width 1355 (with `num_future_frames=5`;
the G1/SMPL blocks scale linearly with that value and the teleop block adds a
constant 27-wide current-frame tail) and the privileged value observation has
width 890. The value observation ports the upstream sonic_release
`privileged_mf_hist` critic contract: a future-reference joint block (all
position frames then all velocity frames on the G1 stride grid), current-frame
anchor channels (the reference anchor position in the robot anchor frame — 3
values carrying the z height error and the xy drift — plus the reference
anchor orientation relative to the robot anchor as 6D), the robot body block
(per tracked body position and 6D orientation in the robot anchor frame), and
the windowed proprioception history (`lin_vel | ang_vel | joint_pos | joint_vel |
action` per frame). Actions are 29 normalized joint-position targets; the
environment action term owns action scaling and offsets, so the FastSAC actor
keeps an identity action affine.

## Data and execution

Training reads a NPZ motion corpus — a directory holding one MotrixLab motion
schema v1 clip per file with the `ext_smpl_joints` / `ext_smpl_root_quat`
reference channels. The default corpus location is the converter's cache
output, so the three steps below chain up without extra flags; override with
`SONIC_MOTION_DIR` (`:`-separated paths join multiple corpus roots):

```bash
source .venv/bin/activate
SONIC_MOTION_DIR=$PWD/data/bones_seed_npz \
  python scripts/train.py task=g1-sonic/motrix.fastsac
```

## Corpus rotation

Full corpora (100k+ clips) cannot stay resident, so `g1-sonic` streams the
corpus through a fixed two-window frame axis
(`SonicMotionCommandCfg.rotation_period`, default 4096 control steps on 16
decode threads; disable with `rotation_period: null`). Each window holds `num_motions` clips drawn
from a seeded without-replacement permutation, so one pass over the corpus
takes `ceil(corpus_size / num_motions)` swaps. Window decode runs on
background threads into private staging buffers; publishing is an in-place
copy into the retiring half of the pre-allocated clip arrays, which costs one
bounded memcpy per swap and nothing else on the stepping path.

Swaps never force-reset envs: reset sampling and clip-end re-anchoring
immediately use the new window, while lanes still tracking the old window run
until their natural episode end, and the retired half is only overwritten
once no lane references it. With off-policy replay this keeps each swap a
gradual mixture shift — a good default is one swap per one to two
`buffer_size` control steps: at `buffer_size` the buffer always blends two
windows, while at `2 × buffer_size` (the `g1-sonic` default) each window's
data fully churns out of the buffer before the next swap arrives, giving
every window a complete digestion pass. Tune from the publish log: window
decode should stay well under the period, and the drain wait under the
episode-length bound. Episode start sampling is selected by `sampling_mode`
(default `"adaptive"`, the gear_sonic release recipe; play/eval configs
switch to `"uniform"` via `for_play()`). In adaptive mode, sampling follows gear_sonic's
per-clip failure-rate contract, shared verbatim by resident corpora and
rotated windows: every source clip is split into `adaptive_bin_size`-frame
bins (one second at 50 Hz by default), and each bin keeps cumulative failure
and episode counters (both initialized to `adaptive_init_count`). Every
lane-step adds `1 / bin_length` episodes to the bin it occupies; every
terminated episode adds one failure at the bin where it ended. Sampling
probabilities blend the per-bin failure rate — clipped at
`failure_rate_max_over_mean` times the mean rate — with a `uniform_ratio`
uniform floor, then reshape the result with sequence-length-agnostic
per-clip weights so every clip carries the same total mass. Episode starts
shift back by up to `pre_failure_sample_window` frames so the policy
practices the approach into a hard segment instead of teleporting into it.
Statistics belong to the source clip and survive window swaps: draining
lanes still report to their retired clip, and learned bins reappear when the
clip returns. The active-window CDF republishes every
`adaptive_update_period` control steps and eagerly on each window publication.
The `adaptive_failure_rate` step metric reports the policy's realized failure
rate over what adaptive sampling is currently serving (failures over episode
exposure, exponentially decayed with a ~7k-step half-life): read it against
`adaptive_sampling_entropy` / `adaptive_sampling_top1_prob` to tell policy
progress from distribution shift — a falling rate while the sampler
concentrates means the policy is mastering what it is being served.
Rotation also spends the learned difficulty at the clip level, mirroring
gear_sonic's motion resample: `rotation_difficulty_ratio` (default 0.5) of
each window's clip slots are drawn with replacement from the sampler's
per-clip weights — hard clips reappear across windows more often — while the
remaining slots keep coming from the without-replacement coverage permutation,
so every clip still rides the pass and buffer diversity keeps a worst-case
floor. Window publication stays drain-gated either way: unlike upstream's
resample-time `reset_all`, episodes are never force-reset (off-policy replay
must not see artificial truncations). With `sampling_mode = "uniform"`, rotation
falls back to the original uniform frame sampler: it neither changes the
random draw sequence nor allocates adaptive state. Corpus roots smaller than
one window fall back to the plain resident load, so smoke corpora are
unaffected.

## Success Guided Sampling

Setting `sampling_mode = "sgs"` swaps the probability model for Success
Guided Sampling (arXiv 2610.12465) while keeping the adaptive path's bin
layout, source-clip statistics ownership, and rotation integration (including
`rotation_difficulty_ratio`). Each bin keeps the last `sgs_history` boolean
episode outcomes — success when the episode reaches its time limit without a
failure termination, failure when it is terminated — credited to the bin
where the episode ended. Starts are drawn through the paper's Beta-shaped
kernel over each bin's success rate (mode `sgs_target_success`, concentration
`sgs_concentration`) and a softmax with temperature `sgs_temperature`, so
training concentrates on bins near the frontier of the policy's capability
while `sgs_epsilon` keeps every bin's probability non-zero and the estimates
updating. Defaults (`0.66 / 5 / 1e-8 / 2 / 100`) are the paper's locomotion
recipe. The step metrics `sgs_sampling_entropy`, `sgs_sampling_top1_prob`,
and `sgs_mean_success_rate` report the distribution and the policy's success
rate over recorded outcomes. `sampling_mode = "round_robin"` instead claims
clips in cyclic order so every clip receives the same number of episode
starts (resident corpora only).

## Build a corpus

Download a paired BONES-SEED subset (G1 retargeted CSVs + SMPL PKLs) from
HuggingFace and convert it into a corpus directory:

```bash
python scripts/motion/download_bones_seed.py   # raw subset -> ~/.cache/motrixlab/bones_seed/g1
python scripts/motion/convert_bones_seed.py --workers 8  # corpus -> ~/.cache/motrixlab/bones_seed_npz/g1
```

The downloader streams both archives with byte budgets (defaults keep about
6000 pairs, 3-4 GB on disk); the converter writes one schema v1 npz per clip
with forward kinematics from the G1 model, dropping clips whose name matches
the default keyword list (prop scenes such as beds/chairs/stairs and extreme
acrobatics such as handstands; the list mirrors gear_sonic, and
`--filter-keywords=` disables it).
`--workers N` runs N conversion processes, each holding its own model and FK
buffer; existing outputs are skipped, so an interrupted run resumes. No
motion data is bundled with the repository.

LAFAN1 motion-capture data builds an equivalent corpus (a retarget-CSV pathway
plus a bvh→smpl pathway over the raw BVHs; ~40 public pairs):

```bash
python scripts/motion/download_lafan4sonic.py   # raw pairs -> ~/.cache/motrixlab/lafan4sonic
python scripts/motion/convert_lafan4sonic.py    # corpus -> ~/.cache/motrixlab/lafan4sonic_npz
```

Train with `SONIC_MOTION_DIR=~/.cache/motrixlab/lafan4sonic_npz`; `:`-joined
paths mix it with a BONES-SEED corpus. This pathway is independent of the
single-clip LAFAN baking used by WBT.

## Small-scale validation

Build a tiny corpus first (a couple of pairs land in minutes), then run the
same `g1-sonic` recipe overriding only scale and duration:

```bash
python scripts/train.py task=g1-sonic/motrix.fastsac \
  num_envs=32 play_num_envs=4 checkpoint.interval=100 \
  algo.trainer.num_learning_iterations=1000
```

See [Training Artifacts](../../tutorial/training/runs_and_checkpoints.md) for playback
of MotrixLab checkpoints.

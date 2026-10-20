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
`auxiliary` holds weights for the five upstream-named auxiliary losses
(`g1_recon`, `g1_smpl_latent`, `g1_teleop_latent`, `teleop_smpl_latent`,
`reencoded_smpl_g1_latent`), which FastSAC weights and logs as `aux_*`
metrics. `g1_control_decoder_hidden_dims` does not appear in the training
configuration: the training actor omits that decoder and uses the FastSAC
policy head.

## Observations and actions

The `g1-sonic` policy observation has width 1355 (with `num_future_frames=5`;
the G1/SMPL blocks scale linearly with that value and the teleop block adds a
constant 27-wide current-frame tail) and the privileged value observation has
width 1220. The value observation merges the gear_sonic and MimicLite critic
contracts: per-future-frame reference joint positions and velocities plus the
reference root pose in the robot root frame (MimicLite
`ref_root_*_future_b`), MimicLite-style per-body reference-minus-actual
differences (position, 6D orientation, and world-frame linear/angular
velocities over all tracked bodies at diff steps `[0, 1]`, each side
canonicalized in the projected-yaw frame of its own pelvis anchor), and the
windowed proprioception history (`lin_vel | ang_vel | joint_pos | joint_vel |
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
episode-length bound. With adaptive sampling disabled (the default), rotation
continues to use the original uniform frame sampler: it neither changes the
random draw sequence nor allocates adaptive state. With adaptive sampling
enabled, failure statistics belong to the original source clip and are
restored when that clip returns in a later window. Statistics are created
lazily as clips are decoded, split into 50-frame bins by default, and an old
window's draining lanes still report failures to their own source clip. The
active distribution reserves `uniform_ratio` (default 20%) for uniform
exploration; the remainder follows source-local failure mass. Corpus roots
smaller than one window fall back to the plain resident load, so smoke corpora
are unaffected.

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

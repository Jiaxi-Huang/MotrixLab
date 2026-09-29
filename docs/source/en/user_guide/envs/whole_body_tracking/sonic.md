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
width 890. Actions are 29 normalized joint-position targets; the environment
action term owns action scaling and offsets, so the FastSAC actor keeps an
identity action affine.

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

## Build a corpus

Download a paired BONES-SEED subset (G1 retargeted CSVs + SMPL PKLs) from
HuggingFace and convert it into a corpus directory:

```bash
python scripts/motion/download_bone_seed.py   # raw subset -> ~/.cache/motrixlab/bones_seed/g1
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

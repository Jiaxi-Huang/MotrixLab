# Motrix RL Builtin

`motrix-rl-builtin` owns MotrixLab's built-in reinforcement-learning integrations, FastSAC and APG. Its Python namespace
is `motrix_rl_builtin`; implementations live under `fastsac` and `apg`.

The plugin owns FastSAC models, replay buffers, environment wrappers, synchronous/asynchronous trainers and policy-export
implementation. The method remains `motrix.fastsac`, with the Torch training backend; `algo.asynchronous` selects the
execution topology rather than a separate method.

It depends only on `motrix-rl` for control-plane services and provider/trainer contracts, plus its
Torch runtime dependencies. It registers via the `motrix_rl.frameworks` entry-point group:
`builtin = "motrix_rl_builtin.plugin:register"`. Consumers query `motrix_rl.frameworks` rather than importing plugin
internals to trigger registration.

APG (`motrix.apg`) learns a generalized-coordinate world model and trains a motion encoder/decoder through imagined
rollouts. It consumes Manager modeling views and the shared sequence replay, with checkpoint/resume, playback and ONNX
export through the standard `mxlab` commands:

```bash
mxlab train task=g1-wbt-supermimic/motrix.apg
mxlab play env=g1-wbt-supermimic
mxlab export-onnx run_dir=<run-dir> output=/tmp/supermimic.onnx
```

APG’s Muon optimizer and compiled imagination path have been validated with Torch 2.11.
The workspace currently pins Torch 2.7, which does not provide `torch.optim.Muon`;
the default APG recipe therefore requires a separately provisioned newer Torch runtime.
Changing the optimizer to Adam alone does not establish compiled-path compatibility with Torch 2.7.

The task default uses smaller networks for local training. On a 32 GiB server GPU, select the explicit
full-size Uni-SuperMimic recipe (including asynchronous collection):

```bash
mxlab train task=g1-wbt-supermimic/motrix.apg +experiment=g1-wbt-supermimic/apg-server
```

APG follows FastSAC's configuration layout: `algo.agent` owns learning, network and replay settings;
`algo.trainer` owns the training budget. For example, use `algo.agent.amp=false`,
`algo.agent.batch_size=1024` or `algo.agent.buffer_size=256`. APG-specific world-model settings
live under `algo.agent.world_model`. Logging and checkpoints use the shared root settings.
Enable bounded asynchronous collection with `algo.asynchronous=true`; collector settings live under
`algo.trainer.async_options`, following FastSAC's layout. This mode uses one collector process and
shared RL transports; it does not support interactive rendering. CPU placement uses
`algo.trainer.async_options.learner_cpu_cores` and `collector_cpu_cores` (for example, `"0:3"`).
Without explicit placement, Torch respects the launch environment; use
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 mxlab train ...` for a process-wide CPU thread budget.
See the
[execution and file responsibilities](../wiki/design/wm-bptt-trainer.md#同步与异步采样执行) for its contracts.

`g1-wbt-supermimic` is a task in the G1 WBT family, alongside dance and backflip. The synchronous implementation is a
reference path; it does not yet establish SuperMimic training-quality parity. Its intentional differences and validation
status are recorded in the [APG design](../wiki/design/wm-bptt-trainer.md#同步实现规格).

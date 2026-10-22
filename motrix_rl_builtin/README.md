# Motrix RL Builtin

`motrix-rl-builtin` owns MotrixLab's built-in reinforcement-learning integrations: FastSAC and the SuperMimic world-model
learner. Its Python namespace is `motrix_rl_builtin`; FastSAC configuration and implementation live under
`motrix_rl_builtin.fastsac`, SuperMimic under `motrix_rl_builtin.supermimic`.

The FastSAC plugin owns FastSAC models, replay buffers, environment wrappers, synchronous/asynchronous trainers and
policy-export implementation. The method remains `motrix.fastsac`, with the Torch training backend; `algo.asynchronous`
selects the execution topology rather than a separate method.

The SuperMimic plugin (`motrix.supermimic`, Torch backend, synchronous trainer) ports the SuperMimic world-model
learner: a learned latent dynamics model with the task-owned analytic reward/termination heads, a goal-conditioned
policy trained on imagined rollouts, and a lambda-returns critic. It couples to environments through the structural
contract in `motrix_rl_builtin.supermimic.contracts` (`supermimic_mdp` plus a motion command term publishing
state/frame/context buffers) and never imports task packages.

Both depend only on `motrix-rl` for control-plane services and provider/trainer contracts, plus their
Torch runtime dependencies. They register via the `motrix_rl.frameworks` entry-point group:
`builtin = "motrix_rl_builtin.plugin:register"`. Consumers query `motrix_rl.frameworks` rather than importing plugin
internals to trigger registration.

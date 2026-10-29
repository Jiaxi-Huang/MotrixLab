# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class ApgAgentCfg:
    """World-model / policy / critic hyperparameters.

    Ported from the upstream reference defaults; the fields are
    MISSING so the algo base yaml must supply every value.
    """

    # replay capacity in control steps (time axis; num_envs is the batch axis)
    buffer_steps: int = MISSING
    # env steps collected per training iteration
    utd: float = MISSING
    # imagination horizon for the policy update
    horizon: int = MISSING
    # observation history length consumed by the policy
    hist: int = MISSING
    # goal encoder latent width
    z_dim: int = MISSING
    # hidden widths
    wm_hidden: list[int] = MISSING
    policy_hidden: list[int] = MISSING
    critic_hidden: list[int] = MISSING
    # world-model training
    wm_window: int = MISSING
    wm_batches: int = MISSING
    wm_batch: int = MISSING
    # policy-update batch
    pi_batch: int = MISSING
    # optimization
    lr: float = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    action_noise: float = MISSING
    critic_epochs: int = MISSING
    critic_minibatches: int = MISSING
    # schedule
    warmup_steps: int = MISSING
    wm_only_iters: int = MISSING
    lr_decay_iters: int = MISSING
    # perf knobs
    amp: str = MISSING
    compile: bool = MISSING
    defer_wm_loss_sync: bool = True
    cache_critic_features: bool = True


@dataclass
class ApgTrainerCfg:
    # Number of learner updates. Collection is driven by ``utd`` and the
    # collector block size below, so an iteration does not imply a fixed
    # number of environment steps.
    num_learning_iterations: int = MISSING
    # Initialize network weights and normalizers without resuming run state.
    initialize_from: str | None = None
    # Reference async trainer topology and bounded transport sizes.
    block_steps: int = 2
    queue_blocks: int = 2
    collector_threads: int = 8
    snapshot_every: int = 10
    collector_policy_device: str = "cuda:0"
    # Optional second CUDA device owning the world model (dual-GPU async
    # topology). Default off and not covered by automated tests; verify on a
    # two-GPU host before relying on it.
    wm_device: str = ""
    worker_timeout: float = 180.0


@dataclass
class ApgCfg:
    """Provider-specific APG configuration.

    ``head`` is the task binding: a ``"package.module:factory"`` dotted path
    resolved against the task package's parsing head; task yamls must supply
    it because the algo base carries no task default.
    """

    # Task-side parsing-head factory (``"package.module:factory"``).
    head: str = MISSING
    # learning device; None -> cuda if available else cpu
    device: str | None = MISSING
    # Topology switch: False runs the single-process sync trainer (in-process
    # collection, serial WM -> policy updates); True runs the reference async
    # collector/learner trainer.
    asynchronous: bool = False
    agent: ApgAgentCfg = field(default_factory=ApgAgentCfg)
    trainer: ApgTrainerCfg = field(default_factory=ApgTrainerCfg)

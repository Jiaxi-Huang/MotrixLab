# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class SuperMimicAgentCfg:
    """SuperMimic world-model / policy / critic hyperparameters.

    Ported from the upstream ``SuperMimicConfig`` defaults; the fields are
    MISSING so the algo base yaml must supply every value.
    """

    # replay capacity in control steps (time axis; num_envs is the batch axis)
    buffer_steps: int = MISSING
    # env steps collected per training iteration
    utd: float = MISSING
    # imagination horizon for the policy update
    horizon: int = MISSING
    # future reference frames in the goal observation
    goal_frames: int = MISSING
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


@dataclass
class SuperMimicTrainerCfg:
    # number of environment-interaction iterations (each collects num_envs transitions)
    num_learning_iterations: int = MISSING


@dataclass
class SuperMimicCfg:
    """Provider-specific SuperMimic configuration (synchronous trainer only)."""

    # learning device; None -> cuda if available else cpu
    device: str | None = MISSING
    agent: SuperMimicAgentCfg = field(default_factory=SuperMimicAgentCfg)
    trainer: SuperMimicTrainerCfg = field(default_factory=SuperMimicTrainerCfg)

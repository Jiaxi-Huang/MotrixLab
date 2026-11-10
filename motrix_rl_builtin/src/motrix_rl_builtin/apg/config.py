# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the APG (world-model + BPTT) provider.

The training recipe mirrors Uni-SuperMimic's ``SuperMimicConfig`` (shared
learning rate with cosine decay to 3%, horizon-32 imagination, windowed
world-model supervision, lambda-return critic on imagined rollouts, Gaussian
action noise), recovered from the prior ``motrix.mbrl`` attempt. Task binding
is derived: the trainer asks the env for its modeling view, so no adapter
paths appear here (D3).
"""

from dataclasses import dataclass, field


@dataclass
class CollectorCfg:
    # Std of Gaussian exploration noise added to policy commands during
    # collection, as a fraction of the task action limit; evaluation stays
    # deterministic. Upstream: ``action_noise``.
    exploration_noise_std: float = 0.1
    # Collection-only clamp after exploration, in task action-scale units.
    exploration_clip: float = 1.0


@dataclass
class ReplayCfg:
    # Transition slots per environment lane; a short on-policy window as
    # upstream (``buffer_steps``), not a large off-policy ring.
    capacity: int = 1000
    # Policy steps collected before learning, including after checkpoint restore.
    warmup_steps: int = 128


@dataclass
class WorldModelCfg:
    learning_rate: float = 0.0002
    # Transitions per supervised window batch. Upstream: ``wm_batch``.
    batch_size: int = 512
    # Window batches per training iteration. Upstream: ``wm_batches``.
    batches_per_iteration: int = 10
    # Multi-step supervision window. Upstream: ``wm_window``.
    window: int = 8
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    max_grad_norm: float = 1.0
    # World-model implementation: ``residual`` predicts the whole modeling
    # state flat with an MLP; ``floating_base`` predicts the physical block
    # (root state + joints) in frame-local coordinates and derives the
    # remaining declared fields through differentiable FK.
    scheme: str = "residual"


@dataclass
class ActorCfg:
    learning_rate: float = 0.0002
    # Imagination start states per policy update. Upstream: ``pi_batch``.
    batch_size: int = 1024
    # Policy updates per training iteration.
    updates_per_iteration: int = 1
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    # Policy-observation history stack length fed to the actor (upstream
    # ``hist``). 1 disables stacking.
    history: int = 1
    # "tanh" bounds policy outputs; "identity" does not (the collector scales
    # by the task action limit either way).
    output_activation: str = "tanh"


@dataclass
class CriticCfg:
    learning_rate: float = 0.0002
    hidden: list[int] = field(default_factory=lambda: [256, 128])
    # Regression passes over the imagined rollout. Upstream: ``critic_epochs``.
    epochs: int = 5
    # Upstream: ``critic_minibatches``.
    minibatches: int = 4
    # Lambda-return mixing. Upstream: ``lam``.
    lam: float = 0.95
    max_grad_norm: float = 1.0


@dataclass
class ImaginationCfg:
    # BPTT unroll length. Upstream: ``horizon``.
    horizon: int = 32
    # Actor gradient clipping; upstream additionally clips gradient values to
    # 0.1 (a recipe constant applied in the learner).
    max_grad_norm: float = 1.0


@dataclass
class TrainerCfg:
    iterations: int = 150
    # Reserved; eager execution only in this provider version.
    compile: bool = False
    # Real env steps collected per iteration and lane. Fractional values
    # carry their remainder across iterations (upstream 1/utd): 2.5 yields
    # alternating 2/3-step iterations.
    collect_steps_per_iteration: float = 200.0
    # Learner MLP precision: null keeps fp32; "bf16" runs the MLP forwards
    # under bf16 autocast and casts outputs back to fp32 (upstream ``amp``).
    # Geometry and rewards stay fp32. CUDA learners only.
    amp: str | None = None
    # WM-only iterations before policy updates start. Upstream:
    # ``wm_only_iters``.
    warmup_iterations: int = 5
    # Cosine lr decay horizon; None decays over ``iterations``. Upstream:
    # ``lr_decay_iters`` (decaying to 3% of the base learning rate).
    lr_decay_iterations: int | None = None
    # Deterministic real evaluation every N iterations; 0 evaluates only
    # after training (D9: an evaluation profile, not a framework concept).
    eval_interval: int = 10
    eval_steps: int = 2000
    # Run the trainer's own end-of-training evaluation.
    final_eval: bool = True


@dataclass
class ApgCfg:
    """Provider configuration for the APG world-model/BPTT algorithm."""

    # Learning device; None -> cuda if available else cpu.
    device: str | None = None
    gamma: float = 0.99
    # Observation-group names the actor and critic consume; both groups are
    # produced by the same derived program on the imagination side.
    policy_group: str = "policy"
    value_group: str = "value"
    collector: CollectorCfg = field(default_factory=CollectorCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)

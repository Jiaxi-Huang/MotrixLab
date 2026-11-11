# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the APG (world-model + BPTT) provider.

The training recipe mirrors Uni-SuperMimic's ``SuperMimicConfig`` (fixed
shared learning rate, horizon-32 imagination, windowed world-model
supervision, lambda-return critic on imagined rollouts, Gaussian action
noise), recovered from the prior ``motrix.mbrl`` attempt. Task binding
is derived: the trainer asks the env for its modeling view, so no adapter
paths appear here (D3).
"""

from dataclasses import dataclass, field


@dataclass
class CollectorCfg:
    # Std of Gaussian exploration noise added to the raw policy command
    # during collection, before the env's per-joint action scaling (joint-
    # space magnitude = std * per-joint scale, as upstream); evaluation stays
    # deterministic. Upstream: ``action_noise`` (unclamped — the env's own
    # control-range clamping is the only bound).
    exploration_noise_std: float = 0.1


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
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    # Policy-observation history stack length fed to the actor (upstream
    # ``hist``). 1 disables stacking.
    history: int = 1
    # Goal-encoder latent width (upstream ``latent_dim``'s tanh bottleneck)
    # when the task declares goal terms; the decoder sees latent + stacked
    # proprioception. The actor's output is always unsquashed, as upstream.
    latent_dim: int = 32


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
    # Name of a one-dim value-group observation term carrying the soft
    # termination probability of the post-step state. When set, the
    # imagination survival gate is ``1 - p`` (upstream's soft-termination
    # discounting); truncations never gate imagination (clip-end is
    # clamp-and-continue). Empty falls back to hard ``terminated`` gates.
    soft_termination_term: str = ""


@dataclass
class TrainerCfg:
    iterations: int = 150
    # Policy updates per env step per lane, as upstream's steady-state utd
    # (0.4 => 1/0.4 = 2.5 collected steps per update, alternating 2/3 via
    # the fractional carry). Upstream derives 0.4 from a replay_ratio
    # formula coupled to its 4096-env width; the bare utd is the portable
    # semantic.
    utd: float = 0.005
    # Learner MLP precision: null keeps fp32 end to end (upstream trains
    # fp32); "bf16" is a repo-side throughput option that runs the MLP
    # forwards under bf16 autocast and casts outputs back to fp32. Geometry
    # and rewards stay fp32 either way. CUDA learners only.
    amp: str | None = None
    # Deterministic real evaluation every N iterations; 0 evaluates only
    # after training (D9: an evaluation profile, not a framework concept).
    eval_interval: int = 10
    eval_steps: int = 2000
    # Run the trainer's own end-of-training evaluation.
    final_eval: bool = False


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
    # Value-group observation terms the actor consumes as flat, never-stacked
    # goal inputs — the upstream convention: only proprioception stacks,
    # goals enter as single current frames. All terms share one encoder
    # (upstream's per-task encoders arrive with a second task, not before).
    goal_terms: tuple[str, ...] = ()
    collector: CollectorCfg = field(default_factory=CollectorCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    actor: ActorCfg = field(default_factory=ActorCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)

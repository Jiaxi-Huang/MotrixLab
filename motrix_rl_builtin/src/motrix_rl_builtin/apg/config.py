# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Analytic policy gradient (learned dynamics + BPTT)."""

from dataclasses import dataclass, field


@dataclass
class NetworkCfg:
    amp: bool = True
    compile: bool = True
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    latent_dim: int = 32
    history_length: int = 10


@dataclass
class WorldModelCfg:
    optimizer: str = "muon"
    learning_rate: float = 0.0004
    num_updates: int = 10
    batch_size: int = 2048
    horizon: int = 8
    root_position_error: float = 0.01
    root_orientation_error: float = 0.02
    root_linear_velocity_error: float = 0.1
    root_angular_velocity_error: float = 0.3
    joint_position_error: float = 0.1
    joint_velocity_error: float = 3.14


@dataclass
class AgentCfg(NetworkCfg):
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    imagination_noise: bool = False
    actor_optimizer: str = "muon"
    actor_learning_rate: float = 0.0004
    critic_learning_rate: float = 0.0004
    batch_size: int = 4096
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    exploration_noise: float = 0.1
    grad_clip_value: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    buffer_size: int = 1000
    learning_starts: int = 0
    # Joint WM/policy update rounds per collector iteration; fractional UTD is supported.
    num_updates: float = 0.4
    policy_ema: float = 0.995
    learning_rate_decay_steps: int = 250000


@dataclass
class AsyncOptionsCfg:
    ring_capacity: int = 2
    collector_inference_device: str = "auto"
    weight_publish_interval: int = 1
    collector_compile: bool = False
    idle_sleep_s: float = 0.001
    learner_cpu_cores: str | None = None
    collector_cpu_cores: str | None = None


@dataclass
class TrainerCfg:
    async_options: AsyncOptionsCfg = field(default_factory=AsyncOptionsCfg)
    num_learning_iterations: int = 1000000


@dataclass
class ApgCfg:
    asynchronous: bool = False
    device: str | None = None
    agent: AgentCfg = field(default_factory=AgentCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)

    @staticmethod
    def migrate_run_config(config: dict) -> dict:
        """Keep archived runs playable after moving thread budgets to workers."""
        config.get("trainer", {}).pop("torch_threads", None)
        return config

    def validate(self) -> None:
        agent = self.agent
        if self.trainer.async_options.ring_capacity < 1 or self.trainer.async_options.idle_sleep_s <= 0:
            raise ValueError("APG async ring_capacity and idle_sleep_s must be positive")
        for optimizer in (agent.actor_optimizer, agent.world_model.optimizer):
            if optimizer not in ("adam", "muon"):
                raise ValueError("APG optimizer must be adam or muon")
        for name, value in (
            ("trainer.async_options.weight_publish_interval", self.trainer.async_options.weight_publish_interval),
            ("agent.history_length", agent.history_length),
            ("agent.world_model.horizon", agent.world_model.horizon),
            ("agent.world_model.batch_size", agent.world_model.batch_size),
            ("agent.world_model.num_updates", agent.world_model.num_updates),
            ("agent.horizon", agent.horizon),
            ("agent.batch_size", agent.batch_size),
            ("agent.critic_epochs", agent.critic_epochs),
            ("agent.critic_minibatches", agent.critic_minibatches),
            ("agent.num_updates", agent.num_updates),
            ("agent.learning_rate_decay_steps", agent.learning_rate_decay_steps),
        ):
            if not 0 < value < float("inf"):
                raise ValueError(f"APG {name} must be finite and positive")
        if agent.learning_starts < 0:
            raise ValueError("APG learning_starts must be nonnegative")
        if agent.buffer_size < max(agent.world_model.horizon, agent.history_length):
            raise ValueError("APG buffer_size must cover WM horizon and policy history")
        if not 0 < agent.gamma < 1 or not 0 <= agent.lam <= 1:
            raise ValueError("APG requires 0 < gamma < 1 and 0 <= lam <= 1")
        if not 0 <= agent.policy_ema < 1:
            raise ValueError("APG policy_ema must be in [0, 1)")

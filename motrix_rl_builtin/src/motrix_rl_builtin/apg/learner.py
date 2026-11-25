# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Multi-step world-model supervision and frozen-dynamics policy BPTT."""

import math

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from motrix_env_core.observations import append_observation
from motrix_rl_builtin.apg.dynamics import Dynamics, state_loss, velocity_increment
from motrix_rl_builtin.apg.networks import Critic, MotionActor, Policy
from motrix_rl_builtin.apg.observations import ApgObservationLayout
from motrix_rl_builtin.apg.optim import make_optimizer


def lambda_returns(rewards, next_values, terminated, truncated, gamma, lam):
    """Time-major returns: truncate bootstraps, terminate does not.

    Recursion stops at either episode boundary; the terminal next value is
    explicit, so auto-reset values cannot leak into a target.
    """
    carry = next_values[-1]
    returns = []
    for t in reversed(range(len(rewards))):
        continuation = torch.where(truncated[t], next_values[t], (1 - lam) * next_values[t] + lam * carry)
        carry = rewards[t] + gamma * (1 - terminated[t].to(rewards.dtype)) * continuation
        returns.append(carry)
    return torch.stack(returns[::-1])


def optimize(optimizer, loss, parameters, clip_value=None):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    return finish_update(optimizer, parameters, clip_value)


def finish_update(optimizer, parameters, clip_value=None):
    """Clip gradients and reject nonfinite norms before optimizer device work.

    A CUDA failure aborts the training process: its device assertion poisons
    the CUDA context, rather than synchronizing the host at every update.
    The assertion and optimizer are enqueued on the same stream in order.
    """
    parameters = list(parameters)
    with torch.profiler.record_function("apg/gradient_clip"):
        cuda = parameters[0].is_cuda
        grad_norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=not cuda)
        if cuda:
            torch._assert_async(torch.isfinite(grad_norm), "APG gradient norm is nonfinite")
    if clip_value is not None:
        nn.utils.clip_grad_value_(parameters, clip_value)
    with torch.profiler.record_function("apg/optimizer_step"):
        optimizer.step()
    return grad_norm.detach()


class Learner:
    def __init__(self, env, view, layout, cfg, device, *, seed=0):
        self.cfg, self.layout, self.device = cfg, layout, torch.device(device)
        self.world_model_iterations = 0
        self.observation_layout = ApgObservationLayout.from_env(env)
        self.proprio_dim = self.observation_layout.proprio_dim
        self.actor = MotionActor(
            self.proprio_dim, self.observation_layout.goal_dim, self.observation_layout.action_dim, cfg.agent
        ).to(device)
        self.policy = Policy(self.actor, env.action_space.low, env.action_space.high).to(device)
        self.policy_ema = (
            AveragedModel(self.policy, multi_avg_fn=get_ema_multi_avg_fn(cfg.agent.policy_ema), use_buffers=True)
            if cfg.agent.policy_ema > 0
            else None
        )
        self.dynamics = Dynamics(
            layout.joint_count,
            cfg.agent.dynamics_hidden_dims,
            env.cfg.ctrl_dt,
            round(env.cfg.ctrl_dt / env.cfg.sim.dt),
            cfg.agent.amp,
        ).to(device)
        self.critic = Critic(
            10 + 2 * layout.joint_count + self.observation_layout.value_dim,
            cfg.agent.critic_hidden_dims,
            cfg.agent.amp,
        ).to(device)
        self.program = view.program.fork(
            cfg.agent.batch_size, device, random_seed=seed if cfg.agent.imagination_noise else None
        )
        self.exogenous = tuple(view.exogenous)
        self.actor_optimizer = make_optimizer(
            self.actor,
            (self.actor.encoder, self.actor.decoder),
            cfg.agent.actor_optimizer,
            cfg.agent.actor_learning_rate,
            betas=(0.9, 0.95),
            device=self.device,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.agent.critic_learning_rate, fused=self.device.type == "cuda"
        )
        self.wm_optimizer = make_optimizer(
            self.dynamics,
            (self.dynamics.net,),
            cfg.agent.world_model.optimizer,
            cfg.agent.world_model.learning_rate,
            betas=(0.9, 0.999),
            device=self.device,
        )
        self._actor_graph = None
        self._encode_runtime = self.layout.encode
        self._normalization_runtime = self._update_normalization
        self._lambda_returns_runtime = lambda_returns
        self._world_model_backward_runtime = self._world_model_backward
        self._critic_backward_runtime = self._critic_backward
        self._imagination_step_runtime = self._imagination_step
        self._observation_runtime = self._observe
        self._critic_runtime = self.critic
        if cfg.agent.compile and self.device.type == "cuda":
            # Canonical modules own parameters/checkpoints; update callables own
            # compiled forward/backward. Optimizer mutation stays eager.
            self._world_model_backward_runtime = torch.compile(self._world_model_backward, mode="reduce-overhead")
            self._critic_backward_runtime = torch.compile(self._critic_backward, mode="reduce-overhead")
            # Single-step outputs survive until trajectory backward: use ordinary
            # compilation, not per-step CUDA graph generations.
            self._imagination_step_runtime = torch.compile(self._imagination_step, fullgraph=True)
            self._observation_runtime = torch.compile(self._observe)
            self._critic_runtime = torch.compile(self.critic)
            # Replay preparation runs outside backward graphs. Fuse its
            # quaternion arithmetic without retaining CUDA-graph output storage.
            self._encode_runtime = torch.compile(self.layout.encode, fullgraph=True)
            self._normalization_runtime = torch.compile(self._update_normalization, fullgraph=True)
            self._lambda_returns_runtime = torch.compile(lambda_returns, fullgraph=True)
        maximum_reward = sum(max(term.weight, 0) for term in env.cfg.reward_cfgs().values()) * env.cfg.ctrl_dt
        self.value_max = maximum_reward / (1 - cfg.agent.gamma)

    def update(self, replay):
        # The reference decay clock is learner rounds, not collected environment steps.
        progress = min(self.world_model_iterations / self.cfg.agent.learning_rate_decay_steps, 1.0)
        scale = 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress))
        for optimizer, base in (
            (self.actor_optimizer, self.cfg.agent.actor_learning_rate),
            (self.critic_optimizer, self.cfg.agent.critic_learning_rate),
            (self.wm_optimizer, self.cfg.agent.world_model.learning_rate),
        ):
            for group in optimizer.param_groups:
                group["lr"] = base * scale
        with torch.profiler.record_function("apg/world_model"):
            metrics = self.update_world_model(replay)
        with torch.profiler.record_function("apg/policy"):
            metrics |= self.update_policy(replay)
        metrics["learning_rate"] = torch.tensor(self.cfg.agent.actor_learning_rate * scale, device=self.device)
        return metrics

    @torch.no_grad()
    def update_normalization(self, state, actions, observations, transition):
        keys = self.layout.root_keys + self.layout.joint_keys
        self._normalization_runtime(
            {key: state[key].to(self.device) for key in keys},
            {key: transition["next_" + key].to(self.device) for key in keys},
            actions.to(self.device),
            transition["obs"].to(self.device),
            self.observation_layout.goal(observations).to(self.device),
        )

    @torch.no_grad()
    def _update_normalization(self, state, next_state, actions, proprio, goal):
        """One device region for encoding and all running-statistic updates."""
        aux, rep = self.layout.encode(state)
        next_aux, next_rep = self.layout.encode(next_state)
        self.dynamics.output_normalizer.update(velocity_increment(aux, rep, next_aux, next_rep))
        self.dynamics.state_normalizer.update(rep)
        self.dynamics.action_normalizer.update(self.layout.control(actions))
        self.actor.obs_normalizer.update(proprio)
        self.actor.goal_normalizer.update(goal)

    def _world_model_loss(self, aux, rep, target_aux, target_rep, actions, valid):
        cfg = self.cfg.agent.world_model
        losses = []
        for t in range(cfg.horizon):
            aux, rep = self.dynamics(aux, rep, self.layout.control(actions[:, t]))
            losses.append(state_loss(aux, rep, target_aux[:, t], target_rep[:, t], cfg))
        loss = torch.where(valid, torch.stack(losses, 1), 0).sum() / valid.sum().clamp_min(1)
        final_error = (losses[-1].detach() * valid[:, -1]).sum() / valid[:, -1].sum().clamp_min(1)
        return loss, final_error

    def _world_model_backward(self, aux, rep, target_aux, target_rep, actions, valid):
        self.wm_optimizer.zero_grad(set_to_none=True)
        loss, final_error = self._world_model_loss(aux, rep, target_aux, target_rep, actions, valid)
        loss.backward()
        return loss.detach(), final_error

    def _critic_backward(self, inputs, targets, mask):
        self.critic_optimizer.zero_grad(set_to_none=True)
        error = (self.critic.normalized(inputs).squeeze(-1) - targets).square()
        loss = (error * mask).sum() / mask.sum().clamp_min(1)
        loss.backward()
        return loss.detach()

    def _begin_update(self):
        if self.cfg.agent.compile and self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()

    def update_world_model(self, replay):
        self.dynamics.requires_grad_(True)
        cfg = self.cfg.agent.world_model
        metrics, final_errors = [], []
        for _ in range(cfg.num_updates):
            with torch.profiler.record_function("apg/wm/sample"):
                batch = replay.sample_windows(
                    cfg.batch_size,
                    cfg.horizon,
                    state_fields=tuple(self.layout.root_keys + self.layout.joint_keys),
                    initial_state_only=True,
                    include_metadata=False,
                )
            aux, rep = self._encode_runtime({key: value[:, 0] for key, value in batch["states"].items()})
            target_aux, target_rep = self._encode_runtime(batch["next_states"])
            self._begin_update()
            with torch.profiler.record_function("apg/wm/backward"):
                loss, final_error = self._world_model_backward_runtime(
                    aux, rep, target_aux, target_rep, batch["actions"], batch["valid"]
                )
            # Copies outside the graph own storage across subsequent generations.
            metrics.append(loss.clone())
            final_errors.append(final_error.clone())
            with torch.profiler.record_function("apg/wm/optimizer"):
                finish_update(self.wm_optimizer, self.dynamics.parameters())
        self.world_model_iterations += 1
        return {
            "world_model_loss": torch.stack(metrics).mean(),
            "world_model_final_loss": torch.stack(final_errors).mean(),
        }

    def _observe(self, aux, rep):
        return self.program.observe(self.layout.decode(aux, rep))

    def _imagination_step(self, aux, rep, history, observation, workspace):
        action = self.actor(history, self.observation_layout.goal(observation["policy"]))
        action = action.clamp(self.policy.low, self.policy.high)
        aux, rep = self.dynamics(aux, rep, self.layout.control(action))
        observation, reward, terminated, truncated, workspace = self.program.transition(
            self.layout.decode(aux, rep), action, workspace, soft_termination=True
        )
        history = append_observation(history, self.observation_layout.proprio(observation["policy"]))
        return aux, rep, history, observation, reward, terminated, truncated, workspace

    def _actor_rollout_loss(self, aux, rep, history, observation, workspace, accepted):
        """Build the actor objective and critic data with explicit trajectory state."""
        cfg = self.cfg.agent
        survival = accepted.to(rep.dtype)
        objective = torch.zeros_like(survival)
        critic_inputs, next_inputs, rewards, terminateds, truncateds, masks = [], [], [], [], [], []
        bootstrap_weights = []
        for t in range(cfg.horizon):
            critic_input = torch.cat([rep, observation["value"]], -1)
            aux, rep, history, observation, reward, terminated, truncated, workspace = self._imagination_step_runtime(
                aux, rep, history, observation, workspace
            )
            next_input = torch.cat([rep, observation["value"]], -1)
            # Ended clips bootstrap at their true final state and then leave
            # the rollout. Hard failure has zero bootstrap.
            objective = objective + cfg.gamma**t * survival * reward
            bootstrap = truncated * (1 - terminated)
            bootstrap_weights.append(cfg.gamma ** (t + 1) * survival * bootstrap)
            critic_inputs.append(critic_input.detach())
            next_inputs.append(next_input)
            rewards.append(reward.detach())
            terminateds.append(terminated.detach())
            truncateds.append(truncated)
            masks.append(survival.detach())
            survival = survival * (1 - terminated) * (~truncated)
        # One batched critic forward replaces H small forwards. The truncation
        # bootstrap weights retain survival gradients and true terminal inputs.
        successor_graph = torch.stack(next_inputs)
        values = self._critic_runtime(successor_graph).squeeze(-1).clamp(max=self.value_max)
        objective = objective + (torch.stack(bootstrap_weights) * values).sum(0)
        objective = objective + cfg.gamma**cfg.horizon * survival * values[-1]
        actor_loss = -objective.sum() / accepted.sum()
        return (
            actor_loss,
            (
                torch.stack(critic_inputs),
                successor_graph.detach(),
                torch.stack(rewards),
                torch.stack(terminateds),
                torch.stack(truncateds),
                torch.stack(masks),
            ),
            workspace,
        )

    def _actor_backward(self, aux, rep, history, observation, workspace, accepted):
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss, trajectory, workspace = self._actor_rollout_loss(aux, rep, history, observation, workspace, accepted)
        loss.backward()
        return loss.detach(), trajectory, workspace

    def _captured_actor_backward(self, aux, rep, history, observation, workspace, accepted):
        if self._actor_graph is None:
            self._stage_actor_workspace = self.program.working_state_buffer()
            self._actor_inputs = (
                aux.detach().clone(),
                rep.detach().clone(),
                history.detach().clone(),
                {key: value.detach().clone() for key, value in observation.items()},
                self._stage_actor_workspace(workspace),
                accepted.detach().clone(),
            )
            stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                # Compile both initial and recurrent step forward/backward before
                # capture. Functional transitions do not consume the input RNG.
                for _ in range(3):
                    self._actor_backward(*self._actor_inputs)
            current.wait_stream(stream)
            self.actor_optimizer.zero_grad(set_to_none=True)
            # Release uncaptured warmup allocations before reserving the graph pool.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            graph = torch.cuda.CUDAGraph()
            # Match Inductor's cuBLAS lifetime protocol: allocate workspace in
            # this graph's pool, not warmup's global pool. Later WM/critic graph
            # recordings clear the cuBLAS cache and must not free our workspace.
            torch._C._cuda_clearCublasWorkspaces()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    self._actor_outputs = self._actor_backward(*self._actor_inputs)
            finally:
                torch._C._cuda_clearCublasWorkspaces()
            current.wait_stream(stream)
            self._actor_graph = graph
        else:
            with torch.no_grad():
                for target, source in zip(self._actor_inputs[:3], (aux, rep, history), strict=True):
                    target.copy_(source)
                for key, value in observation.items():
                    self._actor_inputs[3][key].copy_(value)
                self._stage_actor_workspace(workspace)
                self._actor_inputs[5].copy_(accepted)
        # Capture records work but does not execute it, including on the first call.
        self._actor_graph.replay()
        return self._actor_outputs

    def update_policy(self, replay):
        cfg = self.cfg.agent
        with torch.profiler.record_function("apg/policy/sample"):
            start = replay.sample_starts(cfg.batch_size)
        accepted = start["intact"]
        if not accepted.any():
            return {}
        state = start["states"]
        aux, rep = self._encode_runtime(state)
        self.program.load_exogenous(
            {name: state["command/" + name] for name in self.exogenous}, start["previous_action"]
        )
        history = replay.history(start["envs"], start["positions"], "obs", self.cfg.agent.history_length)
        self._begin_update()
        observation = self._observation_runtime(aux, rep)
        workspace = self.program.working_state()
        self.dynamics.requires_grad_(False)
        self.critic.requires_grad_(False)
        backward = (
            self._captured_actor_backward
            if self.cfg.agent.compile and self.device.type == "cuda"
            else self._actor_backward
        )
        actor_loss, trajectory, workspace = backward(aux, rep, history, observation, workspace, accepted)
        # All consumers survive the following critic graph generations.
        actor_loss = actor_loss.detach().clone()
        inputs, successors, rewards, terminateds, truncateds, masks = (value.detach().clone() for value in trajectory)
        self.program.load_working_state(workspace)
        actor_grad_norm = finish_update(self.actor_optimizer, self.actor.parameters(), cfg.grad_clip_value)
        if self.policy_ema is not None:
            with torch.profiler.record_function("apg/policy/ema"):
                self.policy_ema.update_parameters(self.policy)
        self.critic.input_normalizer.update(torch.cat([inputs, successors], dim=0))
        with torch.no_grad():
            next_values = self._critic_runtime(successors).squeeze(-1)
            targets = self._lambda_returns_runtime(
                rewards,
                next_values,
                terminateds,
                truncateds,
                cfg.gamma,
                cfg.lam,
            ).clamp(max=self.value_max)
        self.critic.requires_grad_(True)
        inputs, targets = inputs.flatten(0, 1), targets.flatten()
        mask = masks.flatten()
        valid_targets = targets[mask > 0].unsqueeze(-1)
        self.critic.return_normalizer.update(valid_targets)
        normalized_targets = self.critic.return_normalizer(targets.unsqueeze(-1)).squeeze(-1)
        losses = []
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(len(targets), device=self.device)
            for indices in torch.tensor_split(order, cfg.critic_minibatches):
                if not indices.numel():
                    continue
                self._begin_update()
                with torch.profiler.record_function("apg/critic/backward"):
                    loss = self._critic_backward_runtime(inputs[indices], normalized_targets[indices], mask[indices])
                losses.append(loss.clone())
                with torch.profiler.record_function("apg/critic/optimizer"):
                    finish_update(self.critic_optimizer, self.critic.parameters())
        alive = masks
        return {
            "actor_loss": actor_loss.detach(),
            "actor_grad_norm": actor_grad_norm,
            "critic_loss": torch.stack(losses).mean(),
            "imagined_length": (alive.sum() / accepted.sum()).detach(),
            "imagined_reward": ((rewards * alive).sum() / alive.sum().clamp_min(1)).detach(),
        }

    def policy_state(self):
        return {
            "policy": (self.policy_ema.module if self.policy_ema is not None else self.policy).state_dict(),
            "proprio_dim": self.proprio_dim,
            "goal_dim": self.actor.goal_dim,
            "action_dim": self.policy.low.numel(),
        }

    def state_dict(self):
        return {
            **self.policy_state(),
            "policy": self.policy.state_dict(),
            "policy_ema": self.policy_ema.state_dict() if self.policy_ema is not None else None,
            "dynamics": self.dynamics.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "world_model_iterations": self.world_model_iterations,
            "imagination_rng": self.program.rng_state(),
        }

    def load_state_dict(self, state):
        self.observation_layout.validate_policy_dimensions(state["proprio_dim"], state["goal_dim"], state["action_dim"])
        # Older checkpoints without a round counter restart the learning-rate clock.
        self.world_model_iterations = state.get("world_model_iterations", 0)
        if "imagination_rng" in state:
            self.program.load_rng_state(state["imagination_rng"])
        self.policy.load_state_dict(state["policy"])
        if self.policy_ema is not None:
            if state.get("policy_ema") is not None:
                self.policy_ema.load_state_dict(state["policy_ema"])
            else:
                self.policy_ema.module.load_state_dict(state["policy"])
        for name in ("dynamics", "critic", "actor_optimizer", "critic_optimizer", "wm_optimizer"):
            getattr(self, name).load_state_dict(state[name])

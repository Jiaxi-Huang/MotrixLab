# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Derived dual-path modeling for manager environments.

The resolver's ``ResolvedManagerProgram`` is pure data: typed term-call
records bound to the live ``ManagerContext`` when the Torch executor builds
its kernels. This module binds the *same* call records to a second, shadow
context whose ``sim`` entries are the caller's modeling-state tensors and
whose commands/actions are shadow copies of the resident terms, yielding an
eager imagination-side program (``ModelingProgram``) plus the derived
``ModelingView`` consumed by sequence-model trainers. Task code stays
unaware: derivability is a property of the env config (see
[Manager 双路径建模设计](../../../wiki/design/manager/modeling.md)).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.exogenous import ExogenousKind, ExogenousRule
from motrix_env_core.manager.kernel_data import is_kernel_data, iter_leaves
from motrix_env_core.manager.modeling_random import (
    DeterministicRandomSource,
    ModelingRandomSource,
    initialize_rand_states,
)
from motrix_env_core.manager.resolver import (
    ResolvedManagerProgram,
)
from motrix_env_core.sim.read import SimDataQuery

logger = logging.getLogger(__name__)


class ModelingDerivationError(RuntimeError):
    """A manager program cannot be derived onto the modeling state."""


@dataclass(frozen=True)
class FieldDesc:
    """One modeling-state field: a declared read key with query semantics.

    ``query_type`` is the concrete ``SimDataQuery`` subclass the key was
    declared with — the semantic type trainer-side dynamics structure is
    assembled from. ``shape`` is the trailing shape (without the batch dim).
    """

    key: str
    query_type: type[SimDataQuery]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelingView:
    """Derived dual-path view of one manager environment.

    ``fields`` covers the declared read set; ``exogenous`` maps
    ``"<command>.<leaf-path>"`` channels to their imagination-advance rules;
    ``program`` executes the production term functions eagerly against
    caller-supplied modeling state; ``action_process`` holds the action
    terms' ``process`` functions (the very same objects the real path uses,
    not re-implementations).
    """

    fields: Mapping[str, FieldDesc]
    exogenous: Mapping[str, ExogenousRule]
    program: ModelingProgram
    action_process: Mapping[str, Callable]


@dataclass(frozen=True)
class _ExogenousEntry:
    channel: str
    tensor: torch.Tensor
    rule: ExogenousRule
    action_slice: slice | None = None


@dataclass(frozen=True)
class _ShadowAction:
    """One action term's shallow clone with a fully copied shadow state."""

    term: Any
    # The real term class's ``process`` — the same function object the real
    # path executes; identity, not re-implementation, is the contract.
    process: Callable
    input_slice: slice
    width: int


@dataclass
class _ShadowHost:
    ctx: ManagerContext
    commands: dict[str, Any]
    actions: dict[str, _ShadowAction]
    field_keys: tuple[str, ...]
    num_envs: int


def _shadow_kernel_data(value: Any, num_envs: int, metric_tensors: set[int]) -> Any:
    """Rebuild one command kernel-data tree under the frozen shadow rules.

    Per-environment rows (leading dim == num_envs) clone into independent
    storage so imagination writes never leak back; shared tensors and scalar
    leaves pass through as the same objects (episode-independent constants,
    shared read-only); declared metric tensors become fresh zero sinks that
    terms may write and nobody reads.
    """
    if isinstance(value, torch.Tensor):
        if id(value) in metric_tensors:
            return torch.zeros_like(value)
        if value.ndim and value.shape[0] == num_envs:
            return value.clone()
        return value
    if isinstance(value, Mapping):
        return dict({key: _shadow_kernel_data(entry, num_envs, metric_tensors) for key, entry in value.items()})
    if is_kernel_data(value):
        rebuilt = []
        changed = False
        for data_field in fields(type(value)):
            leaf = getattr(value, data_field.name)
            shadow = _shadow_kernel_data(leaf, num_envs, metric_tensors)
            changed = changed or shadow is not leaf
            rebuilt.append(shadow)
        return type(value)(*rebuilt) if changed else value
    return value


def _shadow_action_state(value: Any) -> Any:
    """Copy every tensor leaf of one action state.

    Action state is working memory — the raw-action queue, the shared
    cursor, and custom per-env rows — all of it mutable and none of it an
    episode-independent constant, so sharing any leaf with the real term
    would let imagination advance the collector's history.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return dict({key: _shadow_action_state(entry) for key, entry in value.items()})
    if is_kernel_data(value):
        return type(value)(*(_shadow_action_state(getattr(value, f.name)) for f in fields(type(value))))
    return value


def _per_env_leaves(term: Any, num_envs: int, metric_tensors: set[int]) -> dict[str, torch.Tensor]:
    """Map dotted leaf paths to one term's per-environment state tensors."""
    leaves: dict[str, torch.Tensor] = {}
    for path, _annotation, value in iter_leaves(term):
        if (
            isinstance(value, torch.Tensor)
            and id(value) not in metric_tensors
            and value.ndim
            and value.shape[0] == num_envs
        ):
            leaves[".".join(path)] = value
    return leaves


def _map_tensors(value: Any, transform: Callable) -> Any:
    """Rebuild kernel-data/argument containers, preserving non-tensor constants."""
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, Mapping):
        return dict({key: _map_tensors(entry, transform) for key, entry in value.items()})
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{f.name: _map_tensors(getattr(value, f.name), transform) for f in fields(value) if f.init}
        )
    if isinstance(value, tuple):
        return tuple(_map_tensors(entry, transform) for entry in value)
    return value


def _working_copy(value: Any, clone_tensor: Callable, *, buffer: bool = False) -> Callable:
    """Resolve a fixed workspace tree once; hot calls only rebuild its leaves."""
    if isinstance(value, torch.Tensor):
        if not clone_tensor(value):
            return (lambda tensor: value) if buffer else (lambda tensor: tensor)
        if not buffer:
            return lambda tensor: tensor.clone()
        target = value.detach().clone()

        def stage(tensor):
            target.copy_(tensor)
            return target

        return stage
    if isinstance(value, Mapping):
        entries = tuple((key, _working_copy(value[key], clone_tensor, buffer=buffer)) for key in value.keys())
        return lambda current: dict({key: build(current[key]) for key, build in entries})
    if is_dataclass(value) and not isinstance(value, type):
        constructor = type(value)
        entries = tuple(
            (f.name, _working_copy(getattr(value, f.name), clone_tensor, buffer=buffer))
            for f in fields(value)
            if f.init
        )
        return lambda current: constructor(**{name: build(getattr(current, name)) for name, build in entries})
    if isinstance(value, tuple):
        entries = tuple(_working_copy(entry, clone_tensor, buffer=buffer) for entry in value)
        return lambda current: tuple(build(entry) for build, entry in zip(entries, current, strict=True))
    return lambda current: current


def _command_reader(channel: str) -> Callable:
    name, *path = channel.split(".")

    def read(commands):
        value = commands[name]
        for part in path:
            value = value[part] if isinstance(value, Mapping) else getattr(value, part)
        return value

    return read


def _guarded(kind: str, name: str, function: Callable) -> Callable:
    """Wrap one dispatch so context misses attribute themselves precisely."""

    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except KeyError as error:
            raise ModelingDerivationError(
                f"modeling derivation failed: {kind}.{name} accesses a context field outside "
                f"the modeling state (declared read set + exogenous channels): {error}"
            ) from error

    return run


class ModelingProgram:
    """Imagination-side execution of the same term call records.

    ``evaluate`` mirrors the real evaluate kernel's phase order up to (but
    excluding) the command ``advance`` hooks: applying the raw action,
    command ``update`` hooks, rewards, terminations, truncations, weighted
    total. Command ``advance`` is replaced by rule-based
    :meth:`advance_exogenous` — the numerical equivalence is fixture-tested.
    ``observe`` runs the per-group observation calls. ``transition`` exposes
    the same schedule with explicit workspace inputs/outputs for compilation,
    while the eager facade retains lifecycle and boundary validation.
    """

    def __init__(
        self,
        env: Any,
        program: ResolvedManagerProgram,
        host: _ShadowHost,
        fields: Mapping[str, FieldDesc],
        exogenous_entries: tuple[_ExogenousEntry, ...],
        reward_weights: torch.Tensor,
    ):
        self._env = env
        self._program = program
        self._host = host
        self._fields = dict(fields)
        self._exogenous_entries = exogenous_entries
        self._advance_rules = tuple(
            (
                _command_reader(entry.channel),
                _command_reader(entry.channel.split(".", 1)[0] + "." + entry.rule.upper_bound)
                if entry.rule.upper_bound
                else None,
                entry.rule,
                entry.action_slice,
            )
            for entry in exogenous_entries
        )
        # Same tensor the real kernels weight with — values are never copied.
        self._reward_weights = reward_weights
        self._action_receivers = {name: copy.copy(shadow.term) for name, shadow in host.actions.items()}
        self._copy_commands = _working_copy(host.commands, lambda t: t.ndim and t.shape[0] == host.num_envs)
        self._copy_actions = _working_copy(
            {name: shadow.term.state for name, shadow in host.actions.items()}, lambda t: True
        )

    def fork(self, num_envs: int, device: torch.device | str, *, random_seed: int | None = None) -> ModelingProgram:
        """Independent imagination workspace sized for a replay batch.

        Shared constants move once; per-lane working buffers are resized.
        Use load_exogenous() to restore a sampled start before evaluating.
        A seed enables task-owned stochastic terms on an independent stream;
        omitting it preserves deterministic midpoint draws.
        """
        device = torch.device(device)
        old_size = self._host.num_envs

        def resize(tensor):
            if tensor.ndim and tensor.shape[0] == old_size:
                return tensor[:1].to(device).expand(num_envs, *tensor.shape[1:]).clone()
            return tensor.to(device).clone()

        commands = _map_tensors(self._host.commands, resize)
        actions = {}
        for name, shadow in self._host.actions.items():
            term = copy.copy(shadow.term)
            shared = {id(getattr(shadow.term.state, name)) for name in shadow.term.modeling_shared_fields}
            term.state = _map_tensors(
                shadow.term.state, lambda t: t.to(device).clone() if id(t) in shared else resize(t)
            )
            # The cursor is shared, never a per-environment channel.
            term.state = replace(term.state, action_ptr=shadow.term.state.action_ptr.to(device).clone())
            actions[name] = replace(shadow, term=term)
        ctx = _map_tensors(self._host.ctx, resize)
        ctx.env_id = torch.arange(num_envs, device=device)
        ctx.commands = dict(commands)
        ctx.actions = dict({name: shadow.term.state for name, shadow in actions.items()})
        ctx.rand = (
            DeterministicRandomSource(torch.zeros((num_envs, 1), dtype=torch.int64, device=device))
            if random_seed is None
            else ModelingRandomSource(initialize_rand_states(num_envs, random_seed).to(device))
        )
        host = _ShadowHost(ctx, commands, actions, self._host.field_keys, num_envs)
        program = _map_tensors(self._program, lambda t: t.to(device))
        result = ModelingProgram(self._env, program, host, self._fields, (), self._reward_weights.to(device))
        result._exogenous_entries = tuple(
            replace(entry, tensor=result._channel(entry.channel)) for entry in self._exogenous_entries
        )
        result._advance_rules = self._advance_rules
        return result

    def rng_state(self) -> torch.Tensor:
        """Snapshot the independent modeling stream for training checkpoints."""
        return self._host.ctx.rand.state.clone()

    def load_rng_state(self, state: torch.Tensor) -> None:
        """Restore draws without changing command/action rollout state."""
        self._host.ctx.rand.state.copy_(state)

    def action_controls(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Use the production mapping without advancing raw action history.

        For dynamics supervision only; delayed control needs a queue replay
        contract and is explicitly rejected by this stateless interface.
        """
        result = {}
        for name, shadow in self._host.actions.items():
            if getattr(shadow.term.state, "delay_hi", 0):
                raise ValueError("Stateless modeling controls do not support delayed actions")
            result[name] = shadow.term._process(action[:, shadow.input_slice])
        return result

    def _channel(self, channel: str) -> torch.Tensor:
        name, *path = channel.split(".")
        value = self._host.commands[name]
        for part in path:
            value = value[part] if isinstance(value, Mapping) else getattr(value, part)
        return value

    def load_exogenous(self, channels: Mapping[str, torch.Tensor], previous_action: torch.Tensor) -> None:
        """Restore replay's exogenous state and raw action predecessor.

        Action history is initialized without processing a new transition.
        Delay-based action terms require their full delay queue and are not
        supported by this start contract.
        """
        self._renew_working_state(detach=True)
        for entry in self._exogenous_entries:
            entry.tensor.copy_(channels[entry.channel])
        for shadow in self._host.actions.values():
            if getattr(shadow.term.state, "delay_hi", 0):
                raise ValueError("Modeling replay starts do not support delayed actions")
            state = shadow.term.state
            state.action_queue.zero_()
            state.action_ptr.zero_()
            state.action_queue[:, 0] = previous_action[:, shadow.input_slice]

    def _renew_working_state(self, *, detach: bool = False) -> None:
        # Term implementations write into their buffers. Allocate new working
        # storage for each imagined step so later writes never invalidate
        # tensors saved by autograd for earlier rewards/observations.
        n = self._host.num_envs
        self._host.commands = _map_tensors(
            self._host.commands,
            lambda t: (t.detach().clone() if detach else t.clone()) if t.ndim and t.shape[0] == n else t,
        )
        for shadow in self._host.actions.values():
            shadow.term.state = _map_tensors(shadow.term.state, lambda t: t.detach().clone() if detach else t.clone())
        self._host.ctx.commands = dict(self._host.commands)
        self._host.ctx.actions = dict({name: shadow.term.state for name, shadow in self._host.actions.items()})
        self._exogenous_entries = tuple(
            replace(entry, tensor=self._channel(entry.channel)) for entry in self._exogenous_entries
        )

    # -- binding ---------------------------------------------------------------

    def _bind(self, sim_state: Mapping[str, torch.Tensor]) -> ManagerContext:
        state = dict(sim_state)
        missing = [key for key in self._host.field_keys if key not in state]
        if missing:
            raise ValueError(f"modeling state is missing declared read fields: {sorted(missing)}.")
        unknown = sorted(set(state) - set(self._host.field_keys))
        if unknown:
            raise ValueError(
                f"modeling state carries undeclared fields: {unknown}; declared: {list(self._host.field_keys)}."
            )
        return replace(self._host.ctx, sim=dict(state))

    def _apply_action(self, action: torch.Tensor) -> None:
        expected = (self._host.num_envs, self._env.action_space.shape[0])
        if tuple(action.shape) != expected:
            raise ValueError(f"Expected action shape {expected}, got {tuple(action.shape)}.")
        for shadow in self._host.actions.values():
            shadow.process(shadow.term, action[:, shadow.input_slice])

    # -- imagination API --------------------------------------------------------

    def compile(self) -> None:
        """Compile fixed numerical terms while retaining workspace lifecycle."""
        self._evaluate_terms = torch.compile(self._evaluate_terms)
        self.observe = torch.compile(self.observe)

    def evaluate(
        self,
        sim_state: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        *,
        soft_termination: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one imagination transition on the caller's state tensors.

        Writes the raw action into the shadow action history (the same
        ``process`` implementation the real path uses), then runs command
        ``update`` hooks, rewards, terminations, and truncations. Returns
        ``(weighted reward total, terminated union, truncated union)``.
        With ``soft_termination=True``, termination is a differentiable union
        probability using term-declared probability dispatches (hard fallback).
        """
        self._renew_working_state()
        ctx = self._bind(sim_state)
        self._apply_action(action)
        return self._evaluate_terms(ctx, soft_termination)

    def _evaluate_terms(
        self, ctx: ManagerContext, soft_termination: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for hook in self._program.command_updates:
            receiver = ctx.commands[hook.receiver_key]
            hook.function(receiver, ctx)
        total = None
        for term in self._program.rewards:
            value = term.invoke(ctx).to(torch.float32)
            weighted = value * self._reward_weights[term.index] * ctx.dt
            total = weighted if total is None else total + weighted
        terminated = None
        for term in self._program.terminations:
            function = term.probability_function if soft_termination and term.probability_function else term.function
            mask = replace(term, function=function).invoke(ctx)
            if soft_termination:
                mask = mask.to(torch.float32)
                terminated = mask if terminated is None else 1 - (1 - terminated) * (1 - mask)
            else:
                mask = mask.to(torch.bool)
                terminated = mask if terminated is None else terminated | mask
        truncated = None
        for term in self._program.truncations:
            mask = term.invoke(ctx).to(torch.bool)
            truncated = mask if truncated is None else truncated | mask
        num_envs = self._host.num_envs
        return (
            total
            if total is not None
            else torch.zeros(num_envs, dtype=torch.float32, device=self._reward_weights.device),
            terminated
            if terminated is not None
            else torch.zeros(
                num_envs, dtype=torch.float32 if soft_termination else torch.bool, device=self._reward_weights.device
            ),
            truncated
            if truncated is not None
            else torch.zeros(num_envs, dtype=torch.bool, device=self._reward_weights.device),
        )

    def observe(self, sim_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Construct per-group observations from the caller's state tensors.

        Noise terms use the fork's independent RNG. Default deterministic
        forks return identical observations for repeated calls on one state.
        """
        return self._observe_terms(self._bind(sim_state))

    def _observe_terms(self, ctx: ManagerContext) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for group_name, group_layout in self._program.layout.observations.items():
            target = torch.zeros(
                (self._host.num_envs, group_layout.size), dtype=torch.float32, device=self._reward_weights.device
            )
            calls = self._program.observations[group_name]
            for call, term_layout in zip(calls, group_layout.terms, strict=True):
                call.invoke(ctx, target[:, term_layout.output_slice])
            outputs[group_name] = target
        return outputs

    def working_state(self):
        """Borrow the current workspace as explicit inputs to transition()."""
        return (
            self._host.commands,
            {name: shadow.term.state for name, shadow in self._host.actions.items()},
            self._host.ctx.rand.state,
        )

    def working_state_buffer(self):
        """Build a staging callable with fixed mutable Tensor addresses for CUDA capture.

        The returned callable accepts working_state() values of this fork, copies
        per-environment command/action/RNG values, and returns the staged tree.
        Immutable shared assets remain borrowed. Staging does not commit host state.
        """
        commands, actions, rng = self.working_state()
        stage_commands = _working_copy(commands, lambda t: t.ndim and t.shape[0] == self._host.num_envs, buffer=True)
        stage_actions = _working_copy(actions, lambda t: True, buffer=True)
        stage_rng = _working_copy(rng, lambda t: True, buffer=True)

        @torch.no_grad()
        def stage(state):
            return stage_commands(state[0]), stage_actions(state[1]), stage_rng(state[2])

        return stage

    def load_working_state(self, state) -> None:
        """Commit a completed imagined trajectory, including its independent RNG."""
        commands, actions, rng = state
        # Completed backward outputs must outlive subsequent CUDA graph replays.
        commands = _map_tensors(
            commands,
            lambda t: t.detach().clone() if t.ndim and t.shape[0] == self._host.num_envs else t,
        )
        actions = _map_tensors(actions, lambda t: t.detach().clone())
        rng = rng.detach().clone()
        self._host.commands = commands
        for name, shadow in self._host.actions.items():
            shadow.term.state = actions[name]
        self._host.ctx.commands = dict(commands)
        self._host.ctx.actions = dict(actions)
        self._host.ctx.rand = type(self._host.ctx.rand)(rng)
        self._exogenous_entries = tuple(
            replace(entry, tensor=self._channel(entry.channel)) for entry in self._exogenous_entries
        )

    def transition(self, sim_state, action, state, *, soft_termination=False):
        """Evaluate, advance and observe without mutating the input workspace.

        Production dispatch functions are unchanged. Two owned working versions
        retain the evaluate/advance boundary needed by cross-step autograd.
        Only the fixed tree structure is cached; tensor values and RNG are inputs.
        """
        commands, actions, rng = state
        commands, actions = self._copy_commands(commands), self._copy_actions(actions)
        rand = type(self._host.ctx.rand)(rng.clone())
        ctx = replace(self._host.ctx, sim=dict(sim_state), commands=dict(commands), actions=dict(actions), rand=rand)
        for name, shadow in self._host.actions.items():
            term = self._action_receivers[name]
            saved = term.state
            term.state = actions[name]
            shadow.process(term, action[:, shadow.input_slice])
            term.state = saved
        reward, terminated, truncated = self._evaluate_terms(ctx, soft_termination)
        commands, actions = self._copy_commands(commands), self._copy_actions(actions)
        ctx = replace(ctx, commands=dict(commands), actions=dict(actions))
        self._advance_commands(commands, action)
        observation = self._observe_terms(ctx)
        return observation, reward, terminated, truncated, (commands, actions, rand.state)

    def advance_exogenous(self, action: torch.Tensor) -> None:
        """Advance the exogenous channels one imagination step, by rule.

        ``increment`` channels step, ``copy_action`` channels receive their
        action term's slice of ``action``, ``constant`` channels hold.
        """
        self._renew_working_state()
        self._advance_commands(self._host.commands, action)

    def _advance_commands(self, commands, action):
        for read, read_bound, rule, action_slice in self._advance_rules:
            tensor = read(commands)
            if rule.kind is ExogenousKind.INCREMENT:
                tensor.add_(int(rule.step) if not tensor.is_floating_point() else float(rule.step))
                if read_bound is not None:
                    tensor.copy_(torch.minimum(tensor, read_bound(commands)))
            elif rule.kind is ExogenousKind.COPY_ACTION:
                tensor.copy_(action[:, action_slice].reshape(tensor.shape))

    def sync_from_env(self) -> None:
        """Seed the shadow copies from the real resident per-environment state.

        The imagination start point: per-env command rows (excluding metric
        gauges) and the full action working state copy into the shadow
        tensors in place, after which ``evaluate``/``advance_exogenous``/
        ``observe`` mirror the real env step-for-step.
        """
        num_envs = self._host.num_envs
        metric_tensors = {id(value) for value in self._env.metrics.values()}
        for name, real in self._env.command_terms.items():
            real_leaves = _per_env_leaves(real, num_envs, metric_tensors)
            shadow_leaves = _per_env_leaves(self._host.commands[name], num_envs, set())
            for path, value in real_leaves.items():
                shadow_leaves[path].copy_(value)
        for name, real in self._env.action_terms.items():
            shadow_state = self._host.actions[name].term.state
            for real_leaf, shadow_leaf in zip(
                iter_leaves(real.state),
                iter_leaves(shadow_state),
                strict=True,
            ):
                if isinstance(real_leaf[2], torch.Tensor):
                    shadow_leaf[2].copy_(real_leaf[2])

    # -- capability enforcement -------------------------------------------------

    def dry_run(self) -> None:
        """Execute the derived schedule once on a dummy modeling state.

        Group-level all-or-nothing enforcement: every command update hook,
        reward, termination, truncation, and observation dispatch must run
        against the shadow context. The first context access outside the
        modeling state aborts derivation with the offending group, term,
        and field.
        """
        dummy = {
            key: torch.zeros((self._host.num_envs, *desc.shape), dtype=torch.float32)
            for key, desc in self._fields.items()
        }
        ctx = self._bind(dummy)
        self._apply_action(torch.zeros((self._host.num_envs, self._env.action_space.shape[0])))
        for hook in self._program.command_updates:
            receiver = ctx.commands[hook.receiver_key]
            _guarded("command", f"{hook.receiver_key}.update", hook.function)(receiver, ctx)
        for call, layout_term in zip(self._program.rewards, self._program.layout.rewards, strict=True):
            _guarded("reward", layout_term.name, call.invoke)(ctx)
        for call, layout_term in zip(self._program.terminations, self._program.layout.terminations, strict=True):
            _guarded("termination", layout_term.name, call.invoke)(ctx)
            if call.probability_function is not None:
                _guarded(
                    "termination probability",
                    layout_term.name,
                    replace(call, function=call.probability_function).invoke,
                )(ctx)
        for call, layout_term in zip(self._program.truncations, self._program.layout.truncations, strict=True):
            _guarded("truncation", layout_term.name, call.invoke)(ctx)
        for group_name, group_layout in self._program.layout.observations.items():
            target = torch.zeros(
                (self._host.num_envs, group_layout.size), dtype=torch.float32, device=self._reward_weights.device
            )
            calls = self._program.observations[group_name]
            for call, term_layout in zip(calls, group_layout.terms, strict=True):
                _guarded(f"observation group {group_name}", call.name, call.invoke)(
                    ctx, target[:, term_layout.output_slice]
                )


def derive_modeling_view(env: Any) -> ModelingView:
    """Derive the dual-path modeling view of one constructed manager env.

    Requires the resolved task program (``init_state()``). Every group must
    derive — a single non-imaginable group aborts with its first failing
    term, so no task trains on a partial reward or observation set.
    """
    program: ResolvedManagerProgram | None = env._compiled_manager_program
    if program is None:
        raise RuntimeError("modeling_view() requires init_state() to have resolved the task program.")
    for group in env.observation_manager.groups.values():
        if group.history_length != 1:
            raise ModelingDerivationError(
                f"modeling derivation failed: observation.{group.name} uses history_length={group.history_length}; "
                "modeling transitions require single-frame observation groups."
            )
    num_envs = env.num_envs
    metric_tensors = {id(value) for value in env.metrics.values()}

    fields = {}
    for sim_input in program.layout.sim_inputs:
        query_type = type(env.sim_data.query(sim_input.key))
        # Type-level imaginability complements the dry run: a declared query
        # whose semantic type imagination cannot serve (contact, sensing,
        # rendering...) aborts derivation with the key and type named,
        # before any term runs.
        if not query_type.modelable:
            raise ModelingDerivationError(
                f"modeling derivation failed: declared read key {sim_input.key!r} "
                f"({query_type.__name__}) is not modelable — the modeling state "
                "cannot serve its semantic type."
            )
        fields[sim_input.key] = FieldDesc(sim_input.key, query_type, sim_input.shape)
    commands = {name: _shadow_kernel_data(term, num_envs, metric_tensors) for name, term in env.command_terms.items()}
    metrics = {name: torch.zeros_like(value) for name, value in env.metrics.items()}
    actions: dict[str, _ShadowAction] = {}
    for name, term in env.action_terms.items():
        clone = copy.copy(term)
        clone.state = _shadow_action_state(term.state)
        actions[name] = _ShadowAction(
            term=clone,
            process=type(term).process,
            input_slice=env.action_slices[name],
            width=term.action_space.shape[0],
        )
    ctx = ManagerContext(
        # Read-only shared scalars and views carry over; everything mutable
        # is a shadow buffer. ``sim`` is rebound per call by the program.
        env_id=program.context.env_id,
        actions=dict({name: shadow.term.state for name, shadow in actions.items()}),
        commands=dict(commands),
        metrics=dict(metrics),
        rand=DeterministicRandomSource(torch.zeros((num_envs, 1), dtype=torch.int64)),
        sim=dict({}),
        dt=program.context.dt,
        # Imagination never rematerializes simulator state.
        sim_reset_requested=torch.zeros((num_envs, 1), dtype=torch.bool),
        terminated=torch.zeros(num_envs, dtype=torch.bool),
        truncated=torch.zeros(num_envs, dtype=torch.bool),
        reward=torch.zeros(num_envs, dtype=torch.float32),
    )
    host = _ShadowHost(
        ctx=ctx,
        commands=commands,
        actions=actions,
        field_keys=tuple(fields),
        num_envs=num_envs,
    )

    exogenous: dict[str, ExogenousRule] = {}
    entries: list[_ExogenousEntry] = []
    for name, cfg in env.command_cfgs.items():
        declared = getattr(cfg, "exogenous", None) or {}
        if not declared:
            continue
        real_leaves = _per_env_leaves(env.command_terms[name], num_envs, metric_tensors)
        shadow_leaves = _per_env_leaves(commands[name], num_envs, set())
        for field_path, rule in declared.items():
            if field_path not in real_leaves:
                raise ModelingDerivationError(
                    f"modeling derivation failed: command.{name} exogenous channel {field_path!r} "
                    f"is not a per-environment state leaf; available: {sorted(real_leaves)}."
                )
            tensor = shadow_leaves[field_path]
            if rule.kind is ExogenousKind.INCREMENT:
                if not tensor.is_floating_point() and not float(rule.step).is_integer():
                    raise ModelingDerivationError(
                        f"Integer exogenous channel {name}.{field_path} requires an integer step"
                    )
                if rule.upper_bound and rule.upper_bound not in shadow_leaves:
                    raise ModelingDerivationError(f"Unknown increment upper bound {name}.{rule.upper_bound}")
            elif rule.upper_bound:
                raise ModelingDerivationError("upper_bound is only supported for increment channels")
            action_slice = None
            if rule.kind is ExogenousKind.COPY_ACTION:
                if not rule.action_term:
                    raise ModelingDerivationError(
                        f"modeling derivation failed: command.{name} exogenous channel "
                        f"{field_path!r} is copy_action but declares no action_term."
                    )
                shadow_action = actions.get(rule.action_term)
                if shadow_action is None:
                    raise ModelingDerivationError(
                        f"modeling derivation failed: command.{name} exogenous channel "
                        f"{field_path!r} references unknown action term {rule.action_term!r}; "
                        f"available: {sorted(actions)}."
                    )
                if tensor.shape[1:] != (shadow_action.width,):
                    raise ModelingDerivationError(
                        f"modeling derivation failed: command.{name} exogenous channel "
                        f"{field_path!r} trailing shape {tuple(tensor.shape[1:])} does not match "
                        f"action term {rule.action_term!r} width {shadow_action.width}."
                    )
                action_slice = shadow_action.input_slice
            channel = f"{name}.{field_path}"
            exogenous[channel] = rule
            entries.append(_ExogenousEntry(channel, tensor, rule, action_slice))

    modeling = ModelingProgram(
        env=env,
        program=program,
        host=host,
        fields=fields,
        exogenous_entries=tuple(entries),
        # Single-sourced weights: the very tensor the real kernels weight by.
        reward_weights=env._torch_runtime.reward_weights,
    )
    modeling.dry_run()
    logger.info(
        "Manager modeling view derived: %d fields, %d exogenous channels",
        len(fields),
        len(entries),
    )
    return ModelingView(
        fields=fields,
        exogenous=exogenous,
        program=modeling,
        action_process={name: shadow.process for name, shadow in actions.items()},
    )


__all__ = [
    "FieldDesc",
    "ModelingDerivationError",
    "ModelingProgram",
    "ModelingView",
]

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
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.exogenous import ExogenousKind, ExogenousRule
from motrix_env_core.manager.kernel_data import Map, is_kernel_data, iter_leaves
from motrix_env_core.manager.rand import DeterministicRandValue
from motrix_env_core.manager.resolver import (
    ResolvedManagerProgram,
    SimArg,
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
    if isinstance(value, Map):
        return Map(
            {
                key: _shadow_kernel_data(entry, num_envs, metric_tensors)
                for key, entry in zip(value.keys_tuple, value.values_tuple, strict=True)
            }
        )
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
    if isinstance(value, Map):
        return Map(
            {key: _shadow_action_state(entry) for key, entry in zip(value.keys_tuple, value.values_tuple, strict=True)}
        )
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
    """Eager imagination-side execution of the same term call records.

    ``evaluate`` mirrors the real evaluate kernel's phase order up to (but
    excluding) the command ``advance`` hooks: applying the raw action,
    command ``update`` hooks, rewards, terminations, truncations, weighted
    total. Command ``advance`` is replaced by rule-based
    :meth:`advance_exogenous` — the numerical equivalence is fixture-tested.
    ``observe`` runs the per-group observation calls. All scheduling is
    eager; compiling the imagination path would be a separate, later domain.
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
        # Same tensor the real kernels weight with — values are never copied.
        self._reward_weights = reward_weights

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
        return replace(self._host.ctx, sim=Map(state))

    def _args(self, call: Any, ctx: ManagerContext) -> tuple[Any, ...]:
        return tuple(ctx.sim[arg.key] if isinstance(arg, SimArg) else arg.value for arg in call.args)

    def _apply_action(self, action: torch.Tensor) -> None:
        expected = (self._host.num_envs, self._env.action_space.shape[0])
        if tuple(action.shape) != expected:
            raise ValueError(f"Expected action shape {expected}, got {tuple(action.shape)}.")
        for shadow in self._host.actions.values():
            shadow.process(shadow.term, action[:, shadow.input_slice])

    # -- imagination API --------------------------------------------------------

    def evaluate(
        self,
        sim_state: Mapping[str, torch.Tensor],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one imagination transition on the caller's state tensors.

        Writes the raw action into the shadow action history (the same
        ``process`` implementation the real path uses), then runs command
        ``update`` hooks, rewards, terminations, and truncations. Returns
        ``(weighted reward total, terminated union, truncated union)``.
        """
        ctx = self._bind(sim_state)
        self._apply_action(action)
        for hook in self._program.command_updates:
            receiver = ctx.commands[hook.receiver_key]
            hook.function(receiver, ctx)
        total = None
        for term in self._program.rewards:
            value = term.function(ctx, *self._args(term, ctx)).to(torch.float32)
            weighted = value * self._reward_weights[term.index] * ctx.dt
            total = weighted if total is None else total + weighted
        terminated = None
        for term in self._program.terminations:
            mask = term.function(ctx, *self._args(term, ctx)).to(torch.bool)
            terminated = mask if terminated is None else terminated | mask
        truncated = None
        for term in self._program.truncations:
            mask = term.function(ctx, *self._args(term, ctx)).to(torch.bool)
            truncated = mask if truncated is None else truncated | mask
        num_envs = self._host.num_envs
        return (
            total if total is not None else torch.zeros(num_envs, dtype=torch.float32),
            terminated if terminated is not None else torch.zeros(num_envs, dtype=torch.bool),
            truncated if truncated is not None else torch.zeros(num_envs, dtype=torch.bool),
        )

    def observe(self, sim_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Construct per-group observations from the caller's state tensors.

        Noise terms consume the deterministic shadow RNG, so repeated calls
        on the same state return identical observations.
        """
        ctx = self._bind(sim_state)
        outputs: dict[str, torch.Tensor] = {}
        for group_name, group_layout in self._program.layout.observations.items():
            target = torch.zeros((self._host.num_envs, group_layout.size), dtype=torch.float32)
            calls = self._program.observations[group_name]
            for call, term_layout in zip(calls, group_layout.terms, strict=True):
                call.function(ctx, target[:, term_layout.output_slice], *self._args(call, ctx))
            outputs[group_name] = target
        return outputs

    def advance_exogenous(self, action: torch.Tensor) -> None:
        """Advance the exogenous channels one imagination step, by rule.

        ``increment`` channels step, ``copy_action`` channels receive their
        action term's slice of ``action``, ``constant`` channels hold.
        """
        for entry in self._exogenous_entries:
            rule = entry.rule
            if rule.kind is ExogenousKind.INCREMENT:
                entry.tensor.add_(float(rule.step))
            elif rule.kind is ExogenousKind.COPY_ACTION:
                assert entry.action_slice is not None
                entry.tensor.copy_(action[:, entry.action_slice].reshape(entry.tensor.shape))

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
            _guarded("reward", layout_term.name, call.function)(ctx, *self._args(call, ctx))
        for call, layout_term in zip(self._program.terminations, self._program.layout.terminations, strict=True):
            _guarded("termination", layout_term.name, call.function)(ctx, *self._args(call, ctx))
        for call, layout_term in zip(self._program.truncations, self._program.layout.truncations, strict=True):
            _guarded("truncation", layout_term.name, call.function)(ctx, *self._args(call, ctx))
        for group_name, group_layout in self._program.layout.observations.items():
            target = torch.zeros((self._host.num_envs, group_layout.size), dtype=torch.float32)
            calls = self._program.observations[group_name]
            for call, term_layout in zip(calls, group_layout.terms, strict=True):
                _guarded(f"observation group {group_name}", call.name, call.function)(
                    ctx,
                    target[:, term_layout.output_slice],
                    *self._args(call, ctx),
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
        actions=Map({name: shadow.term.state for name, shadow in actions.items()}),
        commands=Map(dict(commands)),
        metrics=Map(metrics),
        rand=DeterministicRandValue(torch.zeros((num_envs, 1), dtype=torch.int64)),
        sim=Map({}),
        dt=program.context.dt,
        # Imagination never rematerializes simulator state.
        sim_reset_requested=torch.zeros((num_envs, 1), dtype=torch.bool),
        reset_ids=torch.zeros(0, dtype=torch.int64),
        reset_episode=torch.zeros((num_envs, 1), dtype=torch.bool),
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

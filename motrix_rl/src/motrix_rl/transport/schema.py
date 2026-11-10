# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Field-set contract for one transition slot of an asynchronous trainer.

A :class:`TransitionSchema` declares which named tensors make up one
env-step batch flowing from a collector to a learner. The schema is a
provider-level decision: the transport primitives in this package allocate,
validate and address slots strictly by what the schema says, so the field
set is a configuration-time decision of the training backend instead of a
compile-time decision of the transport.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransitionField:
    """One named tensor of a transition batch, per environment.

    Args:
        name: Field name; ``push`` addresses batches by these names.
        per_env_shape: Shape of the field for ONE environment. A scalar
            field is ``()``; a vector observation is ``(dim,)``. Multi-dim
            shapes are carried exactly by the host ring (the fused CUDA-IPC
            ring returns them flattened; see ``gpu_ipc_ring``).
        dtype: Storage dtype of the shared slot tensor.
    """

    name: str
    per_env_shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def per_env_numel(self) -> int:
        n = 1
        for dim in self.per_env_shape:
            n *= dim
        return n


@dataclass(frozen=True)
class TransitionSchema:
    """The declared field set of one transition slot.

    Field order is part of the schema: consumers that convert the by-name
    ``read_span`` views to positional tuples (replay-buffer ingest paths)
    use :attr:`fields` order as the single ordering authority.
    """

    fields: tuple[TransitionField, ...]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("TransitionSchema requires at least one field")
        names = [field.name for field in self.fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"TransitionSchema field names must be unique, got duplicates: {duplicates}")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def slot_shape(self, field: TransitionField, num_envs: int) -> tuple[int, ...]:
        """Per-slot shape of one field: ``(num_envs,) + per_env_shape``."""
        return (num_envs,) + tuple(field.per_env_shape)

    @property
    def feature_width(self) -> int:
        """Summed per-env element count across fields (fused-slot width)."""
        return sum(field.per_env_numel for field in self.fields)


def wm_transition_schema(
    state_fields: Iterable[TransitionField],
    act_dim: int,
    *,
    obs_dim: int | None = None,
) -> TransitionSchema:
    """Build the world-model transition schema for one modeling-state geometry.

    The modeling state is task-derived (declared read set + exogenous
    channels), so it arrives as named fields rather than one fused vector.
    Every state field gets a same-shape ``next_<name>`` successor column: a
    transition slot is self-contained, and the done rows' true (pre-reset)
    successors — captured by the env's FinalState — are written by the
    provider-side assembler. ``terminated`` and ``truncated`` stay separate
    (lambda-return bootstrap semantics differ); the episode boundary is
    their union. Episode/step ids are deliberately absent: the sequence
    replay derives them from the done flags, and wide counters riding the
    fused CUDA-IPC ring as floats would lose precision (see
    ``gpu_ipc_ring``).

    Provider-specific extra fields concatenate onto the returned schema's
    ``fields``; the template itself stays the shared core.
    """
    states = tuple(state_fields)
    prefixed = sorted(field.name for field in states if field.name.startswith("next_"))
    if prefixed:
        raise ValueError(f"WM state field names must not start with 'next_': {prefixed}")
    successors = tuple(TransitionField(f"next_{field.name}", field.per_env_shape, field.dtype) for field in states)
    fields = (
        *states,
        *successors,
        TransitionField("actions", (act_dim,), torch.float32),
        TransitionField("rewards", (), torch.float32),
        TransitionField("terminated", (), torch.int64),
        TransitionField("truncated", (), torch.int64),
    )
    if obs_dim is not None:
        fields = (*fields, TransitionField("obs", (obs_dim,), torch.float32))
    return TransitionSchema(fields=fields)

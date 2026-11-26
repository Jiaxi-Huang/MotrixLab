# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bind resolved MotrixSim write fields to persistent CUDA storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from motrixsim import write

from motrix_env_core.sim.write import WriteProgram
from motrix_env_motrixsim.write_compiler import MotrixWriteCompiler, ResolvedWrite

if TYPE_CHECKING:
    from motrix_env_motrixsim.gpu_runtime import MotrixCudaBackend


@dataclass
class _Channel:
    slot: int
    target: torch.Tensor
    columns: torch.Tensor | None


@dataclass
class _Parameter:
    slot: int
    setters: tuple
    scratch: torch.Tensor
    spin_roll: torch.Tensor | None


class CudaWriteCompiler(MotrixWriteCompiler):
    def __init__(self, backend: MotrixCudaBackend):
        super().__init__(backend.host_model)
        self.backend = backend

    def allocate(self, *, reset: bool, refresh: bool) -> WriteProgram:
        with self.backend.physics_stream():
            return CudaWrites(self.backend, self._pending, reset, refresh)


class CudaWrites(WriteProgram):
    def __init__(self, backend: MotrixCudaBackend, fields: list[tuple[str, ResolvedWrite]], reset: bool, refresh: bool):
        self._backend = backend
        self._reset = reset
        self._refresh = refresh
        self._buffers: dict[str, torch.Tensor] = {}
        self._channels: list[_Channel] = []
        mocaps: dict[str, list] = {}
        self._parameters: list[_Parameter] = []
        for slot, (name, field) in enumerate(fields):
            for indices, target in (
                (field.ctrl_indices, backend.control),
                (field.pos_indices, backend.position),
                (field.vel_indices, backend.velocity),
            ):
                if indices is None:
                    continue
                shape = indices.shape
                flat = indices.ravel()
                if flat.size and np.array_equal(flat, np.arange(flat[0], flat[0] + flat.size)):
                    self._channels.append(_Channel(slot, target[:, int(flat[0]) : int(flat[-1]) + 1], None))
                else:
                    self._channels.append(_Channel(slot, target, torch.as_tensor(flat, device=backend.device)))
                break
            else:
                source = field.native
                if isinstance(source, (write.BodyPosition, write.BodyRotation)):
                    rotation = isinstance(source, write.BodyRotation)
                    shape = (len(source.selection), 4 if rotation else 3)
                    components = slice(3, 7) if rotation else slice(3)
                    for column, body in enumerate(source.selection):
                        mocaps.setdefault(body, []).append((slot, column, components))
                else:
                    shape = self._bind_parameter(slot, source)
            self._buffers[name] = backend.position.new_zeros((backend.data.num_worlds, *shape))
        self._mocap_fields = tuple((backend.model.get_body(body).mocap, fields) for body, fields in mocaps.items())
        self._values = tuple(self._buffers.values())

    def _bind_parameter(self, slot: int, source: write.WriteSource) -> tuple[int, ...]:
        backend = self._backend
        model = backend.model
        names = source.selection
        if isinstance(source, write.GeomFrictionOverride):
            handles = [model.get_geom(n) for n in names]
            accessors = [(h.get_friction_override, h.set_friction_override) for h in handles]
            spin_roll = backend.position.new_tensor(
                np.stack([backend.host_model.get_geom(n).friction[1:] for n in names])
            )
        else:
            spin_roll = None
            if isinstance(source, write.ActuatorKpOverride):
                handles = [model.get_actuator(n) for n in names]
                accessors = [(h.get_kp_override, h.set_kp_override) for h in handles]
            elif isinstance(source, write.ActuatorDampingOverride):
                handles = [model.get_actuator(n) for n in names]
                accessors = [(h.get_damping_override, h.set_damping_override) for h in handles]
            elif isinstance(source, write.LinkMassOverride):
                handles = [model.get_link(n) for n in names]
                accessors = [(h.get_mass_override, h.set_mass_override) for h in handles]
            elif isinstance(source, write.LinkCenterOfMassOverride):
                handles = [model.get_link(n) for n in names]
                accessors = [(h.get_center_of_mass_override, h.set_center_of_mass_override) for h in handles]
            else:
                raise NotImplementedError(f"Unsupported CUDA parameter {type(source).__name__}")
        components = (3,) if isinstance(source, write.LinkCenterOfMassOverride) else ()
        for name, (getter, setter) in zip(names, accessors):
            identity = (type(source), name)
            if identity not in backend._reset_parameters:
                default = backend.position.new_tensor(getter(backend.data)).reshape(-1, *components)
                default = default.expand(backend.data.num_worlds, *components).clone()
                backend._reset_parameters[identity] = setter, default, torch.empty_like(default)
        scratch = backend.position.new_empty((len(names), backend.data.num_worlds, *components))
        self._parameters.append(_Parameter(slot, tuple(s for _, s in accessors), scratch, spin_roll))
        return (len(names), 3) if spin_roll is not None else (len(names), *components)

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        self._execute(env_ids, self._values, compact=False)

    def execute_compact(self, env_ids: torch.Tensor, buffers: tuple[torch.Tensor, ...]) -> None:
        self._execute(env_ids, buffers, compact=True)

    def _execute(self, ids: torch.Tensor | None, buffers: tuple[torch.Tensor, ...], *, compact: bool) -> None:
        if ids is not None:
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("Simulator write env_ids must be a one-dimensional integer tensor.")
            if ids.numel() == 0:
                return
            ids = ids.long()
        rows = slice(None) if ids is None else ids
        source_rows = slice(0, ids.numel()) if compact else rows
        backend = self._backend
        with backend.physics_stream():
            # Validate restricted fields before changing simulator state.
            for parameter in self._parameters:
                if (
                    parameter.spin_roll is not None
                    and (buffers[parameter.slot][source_rows, :, 1:] != parameter.spin_roll).any()
                ):
                    raise NotImplementedError("MotrixSim CUDA 1.0.0 supports only sliding-friction overrides")
            mask = backend.mocap_mask(ids) if self._mocap_fields or (self._reset and backend._mocap_defaults) else None
            if self._reset:
                backend.reset_rows(rows, mask)
            for channel in self._channels:
                values = buffers[channel.slot].flatten(1)[source_rows]
                if channel.columns is None:
                    if ids is None:
                        channel.target.copy_(values)
                    else:
                        channel.target[ids] = values
                else:
                    row_index = rows if ids is None else ids[:, None]
                    channel.target[row_index, channel.columns] = values
            for parameter in self._parameters:
                values = buffers[parameter.slot][source_rows]
                if parameter.spin_roll is not None:
                    values = values[..., 0]
                if ids is not None:
                    parameter.scratch.fill_(torch.nan)
                parameter.scratch[:, rows] = values.transpose(0, 1)
                for column, setter in enumerate(parameter.setters):
                    setter(backend.data, wp.from_torch(parameter.scratch[column]))
            for mocap, fields in self._mocap_fields:
                poses = wp.to_torch(mocap.get_pose(backend.data))
                for slot, column, components in fields:
                    poses[rows, components] = buffers[slot][source_rows, column]
                mocap.set_pose(backend.data, wp.from_torch(poses), world_mask=mask)
            if self._refresh:
                backend.model.forward_kinematic(backend.data)

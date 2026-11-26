# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched CUDA physics on a Warp stream ordered with the calling Torch stream."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace

import torch
import warp as wp
from motrixsim import gpu

from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.sim.backend import RenderConfig, SimBackend, SimRenderer
from motrix_env_core.sim.model import ModelQuery, SimModel
from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.gpu_reads import CudaReadCompiler
from motrix_env_motrixsim.gpu_writes import CudaWriteCompiler
from motrix_env_motrixsim.runtime import MotrixSimModelCompiler


def _device_metadata(value, device):
    """Place tensors in the published static model on the simulation device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_metadata(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_metadata(item, device) for item in value)
    if is_dataclass(value):
        return replace(value, **{f.name: _device_metadata(getattr(value, f.name), device) for f in fields(value)})
    return value


class MotrixCudaBackend(SimBackend):
    name = "motrixsim"

    def __init__(self, scene: SceneCfg, sim: SimCfg, num_envs: int, *, render: RenderConfig | None = None) -> None:
        super().__init__(scene, sim, num_envs, render=render)
        device = torch.device(sim.device)
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        self._warp_device = wp.get_device(str(self.device))
        self._stream = wp.Stream(self._warp_device)
        self._stream.wait_stream(wp.get_stream(self._warp_device))
        self._caller_ready = wp.Event(self._warp_device)
        self._torch_stream = torch.cuda.ExternalStream(self._stream.cuda_stream, device=self.device)
        host_model = MotrixSimSceneCompiler().compile(scene, sim, render=render)
        self.host_model = host_model
        self._model_compiler = MotrixSimModelCompiler(host_model)
        with self.physics_stream():
            self.model = gpu.SceneModel.from_cmodel(
                host_model, max_contacts=sim.max_contacts, max_contact_constraints=sim.max_contact_constraints
            )
            self.data = gpu.SceneData(self.model, batch=num_envs)
            self.position = wp.to_torch(self.data.dof_pos)
            self.velocity = wp.to_torch(self.data.dof_vel)
            self.control = wp.to_torch(self.data.actuator_ctrls)
        self.initial_position = self.position[0].clone()
        self._domains = dict(self.data.low.domains.items())
        self._contact_sensors = {sensor.name: sensor for sensor in host_model.low.low_contact_sensors}
        self._touch_sensors = {sensor.name: sensor for sensor in host_model.low.low_touch_sensors}
        self._terrain = {}
        # Both parity banks are live. Invalidate the selected worlds in both
        # banks without advancing their shared clock or the other worlds.
        solver = self._domains["solver"]
        warm = solver.persistent_lambdas
        self._reset_arrays = []
        if warm is not None:
            for array in (warm.tags, warm.counts, warm.segment_counts, warm.row_counter):
                if array.size:
                    tensor = wp.to_torch(array)
                    self._reset_arrays.append((tensor, tensor[0].clone()))
        normal = solver.persistent_normal_lambdas
        collision = self._domains["rigid_collision"]
        for array in (
            normal.equality_mask,
            normal.unilateral_mask,
            collision.contact_count,
            collision.overflow,
            solver.constraint_lambdas.equality,
            solver.constraint_lambdas.unilateral,
        ):
            if array.size:
                tensor = wp.to_torch(array)
                self._reset_arrays.append((tensor, torch.zeros_like(tensor[0])))
        self._overflow = wp.to_torch(collision.overflow)
        self._reset_parameters = {}
        self._mocap_defaults = []
        with self.physics_stream():
            for body in self.model.bodies:
                if body.is_mocap:
                    mocap = body.mocap
                    self._mocap_defaults.append((mocap, wp.to_torch(mocap.get_pose(self.data))[0].clone()))
        self._write_compiler = CudaWriteCompiler(self)

    @contextmanager
    def physics_stream(self):
        current = torch.cuda.current_stream(self.device)
        with wp.ScopedStream(self._stream, sync_enter=False, sync_exit=False):
            if current.cuda_stream == self._stream.cuda_stream:
                yield
                return
            caller = wp.stream_from_torch(current)
            self._stream.wait_stream(caller, event=self._caller_ready)
            try:
                with torch.cuda.stream(self._torch_stream):
                    yield
            finally:
                caller.wait_stream(self._stream)

    step_context = physics_stream

    @property
    def model_compiler(self):
        return self._model_compiler

    @property
    def write_compiler(self):
        return self._write_compiler

    @property
    def num_dof_pos(self):
        return self.model.num_dof_pos

    @property
    def num_dof_vel(self):
        return self.model.num_dof_vel

    @property
    def num_actuators(self):
        return self.model.num_actuators

    def create_renderer(self, config, *, num_envs, render_spacing):
        from motrix_env_motrixsim.renderer import MotrixSimRenderer

        return MotrixSimRenderer(
            self.model,
            lambda: self.data,
            config,
            num_envs=num_envs,
            render_spacing=render_spacing,
            system_camera=self._scene.system_camera,
            render_app=gpu.render.RenderApp(headless=config.headless),
        )

    def set_mocap_pose(self, mocap, poses, rows):
        mask = None
        if isinstance(rows, torch.Tensor):
            selected = torch.zeros(self.data.num_worlds, device=self.device, dtype=torch.bool)
            selected[rows] = True
            mask = wp.from_torch(selected)
        mocap.set_pose(self.data, wp.from_torch(poses), world_mask=mask)

    def reset_rows(self, rows):
        self.position[rows] = self.initial_position
        self.velocity[rows] = 0
        self.control[rows] = 0
        for tensor, initial in self._reset_arrays:
            tensor[rows] = initial
        with self.physics_stream():
            for setter, default in self._reset_parameters.values():
                values = torch.full_like(default, torch.nan)
                values[rows] = default[rows]
                setter(self.data, wp.from_torch(values))
            for mocap, default in self._mocap_defaults:
                poses = wp.to_torch(mocap.get_pose(self.data))
                poses[rows] = default
                self.set_mocap_pose(mocap, poses, rows)

    def sample_terrain_height(self, geom_name: str, env_ids: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        from motrix_env_core.mdp.terrain import heightfield_lookup
        from motrix_env_core.sim.model import HeightFieldDataQuery

        if geom_name not in self._terrain:
            geom = self.host_model.get_geom(geom_name)
            if geom.hfield is None:
                self._terrain[geom_name] = float(geom.local_pose[2])
            else:
                self._terrain[geom_name] = self.compile_model({"grid": HeightFieldDataQuery(geom=geom_name)}).others[
                    "grid"
                ]
        terrain = self._terrain[geom_name]
        if isinstance(terrain, float):
            return xy.new_full(xy.shape[:-1], terrain)
        return heightfield_lookup(terrain, xy[..., 0].double(), xy[..., 1].double()).float()

    def compile_model(self, queries: Mapping[str, ModelQuery]) -> SimModel:
        return _device_metadata(super().compile_model(queries), self.device)

    def compile_reads(self, queries: Mapping[str, SimDataQuery]) -> PhysicsReadProgram:
        return CudaReadCompiler(self, queries).build()

    def step(self, substeps: int, result: torch.Tensor | None = None) -> None:
        with self.physics_stream():
            for _ in range(substeps):
                self.model.step(self.data, stream=self._stream)
        if result is not None:
            good = self.position.isfinite().all(dim=1) & self.velocity.isfinite().all(dim=1)
            result.copy_(~good | self._overflow.ne(0))

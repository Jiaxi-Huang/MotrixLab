# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched CUDA physics on a Warp stream ordered with the calling Torch stream."""

from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace

import torch
import warp as wp
from motrixsim import gpu

from motrix_env_core.sim import read
from motrix_env_core.sim import write as declarations
from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.read import ActuatorCtrlQuery, DofPositionQuery, DofVelocityQuery, PhysicsReadProgram
from motrix_env_core.sim.write import CtrlTargetsWrite, JointPositionWrite, JointVelocityWrite, WriteProgram
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
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


class CudaReads(PhysicsReadProgram):
    def __init__(self, backend, queries):
        self._queries = dict(queries)
        self._sources = {}
        self._outputs = {}
        self._backend = backend
        with backend.physics_stream():
            for key, query in queries.items():
                if query not in self._sources:
                    fetch = self._reader(backend, key, query)
                    self._sources[query] = fetch
                    self._outputs[query] = fetch().to(torch.float32).clone()

    @staticmethod
    def _reader(backend, key, query):
        model, data = backend.model, backend.data
        if isinstance(query, DofPositionQuery):
            return lambda: backend.position
        if isinstance(query, DofVelocityQuery):
            return lambda: backend.velocity
        if isinstance(query, ActuatorCtrlQuery):
            return lambda: backend.control
        if isinstance(query, (read.BodyJointPositionQuery, read.BodyJointVelocityQuery)):
            body = model.get_body(query.body)
            position = isinstance(query, read.BodyJointPositionQuery)
            target = backend.position if position else backend.velocity
            ids = body.get_dof_pos_indices(False) if position else body.get_dof_vel_indices(False)
            cols = torch.as_tensor(ids.astype("int64"), device=target.device)
            return lambda: target[:, cols]
        if isinstance(query, (read.JointPositionQuery, read.JointVelocityQuery)):
            position = isinstance(query, read.JointPositionQuery)
            target = backend.position if position else backend.velocity
            joints = [model.get_joint(n) for n in query.joints]
            ids = [j.dof_pos_index if position else j.dof_vel_index for j in joints]
            cols = torch.tensor(ids, device=target.device)
            return lambda: target[:, cols]
        if isinstance(query, read.SensorValuesQuery):
            readers = [backend.sensor_reader(name) for name in query.sensors]
            return lambda: torch.cat([fetch() for fetch in readers], dim=1)
        if isinstance(query, (read.SitePositionQuery, read.SiteQuaternionQuery)):
            site = model.get_site(query.site)
            columns = slice(3, 7) if isinstance(query, read.SiteQuaternionQuery) else slice(3)
            return lambda: wp.to_torch(site.get_pose(data))[:, columns]
        if isinstance(query, (read.GeomPositionQuery, read.GeomQuaternionQuery, read.GeomLinearVelocityQuery)):
            geom = model.get_geom(query.geom)
            if isinstance(query, read.GeomLinearVelocityQuery):
                return lambda: wp.to_torch(geom.get_linear_velocity(data))
            columns = slice(3, 7) if isinstance(query, read.GeomQuaternionQuery) else slice(3)
            return lambda: wp.to_torch(geom.get_pose(data))[:, columns]
        for single, batch, method in (
            (read.LinkPositionQuery, read.BatchLinkPositionQuery, lambda link: link.get_position),
            (read.LinkQuaternionQuery, read.BatchLinkQuaternionQuery, lambda link: link.get_rotation),
            (read.LinkLinearVelocityQuery, read.BatchLinkLinearVelocityQuery, lambda link: link.get_linear_velocity),
            (read.LinkAngularVelocityQuery, read.BatchLinkAngularVelocityQuery, lambda link: link.get_angular_velocity),
        ):
            if isinstance(query, single):
                fetch = method(model.get_link(query.link))
                return lambda: wp.to_torch(fetch(data))
            if isinstance(query, batch):
                getters = [method(model.get_link(n)) for n in query.links]
                return lambda: torch.stack([wp.to_torch(get(data)) for get in getters], dim=1)
        if isinstance(query, read.GeomPairCollidingQuery):
            low = backend.host_model.low
            readers = [
                backend.contact_count_reader(
                    [(a, b) for a in low.get_collider_indices(first) for b in low.get_collider_indices(second)]
                )
                for first, second in query.pairs
            ]
            return lambda: torch.cat([fetch() > 0 for fetch in readers], dim=1)
        raise NotImplementedError(f"MotrixSim CUDA cannot read {key!r} ({type(query).__name__})")

    @property
    def arena_bytes(self):
        return sum(out.numel() * out.element_size() for out in self._outputs.values())

    @property
    def keys(self):
        return tuple(self._queries)

    def query(self, key):
        return self._queries[key]

    def view(self, key):
        return self._outputs[self.query(key)]

    def execute(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        with self._backend.physics_stream():
            for query, fetch in self._sources.items():
                self._outputs[query][rows] = fetch()[rows].to(dtype=torch.float32)


class CudaWrites(WriteProgram):
    def __init__(self, backend, writes, reset, forward_kinematics):
        self._backend = backend
        self._reset = reset
        self._refresh = reset
        self._buffers = {}
        self._targets = {}
        self._mocap = {}
        self._parameters = {}
        for key, write in writes.items():
            shape = None
            if isinstance(write, (declarations.KinematicBodyPositionWrite, declarations.KinematicBodyRotationWrite)):
                rotation = isinstance(write, declarations.KinematicBodyRotationWrite)
                bodies = [backend.model.get_body(n).mocap for n in write.bodies]
                self._mocap[key] = bodies, slice(3, 7) if rotation else slice(3)
                self._buffers[key] = backend.position.new_zeros(
                    (backend.data.num_worlds, len(bodies), 4 if rotation else 3)
                )
                self._refresh = True
                continue
            overrides = []
            if isinstance(write, (declarations.ActuatorKpWrite, declarations.ActuatorDampingWrite)):
                for name in write.actuators:
                    actuator = backend.model.get_actuator(name)
                    if isinstance(write, declarations.ActuatorKpWrite):
                        overrides.append((name, actuator.get_kp_override, actuator.set_kp_override))
                    else:
                        overrides.append((name, actuator.get_damping_override, actuator.set_damping_override))
            elif isinstance(write, (declarations.LinkMassWrite, declarations.LinkComWrite)):
                for name in write.links:
                    link = backend.model.get_link(name)
                    if isinstance(write, declarations.LinkMassWrite):
                        overrides.append((name, link.get_mass_override, link.set_mass_override))
                    else:
                        overrides.append((name, link.get_center_of_mass_override, link.set_center_of_mass_override))
            elif isinstance(write, declarations.GeomFrictionWrite):
                for name in write.geoms:
                    geom = backend.model.get_geom(name)
                    overrides.append((name, geom.get_friction_override, geom.set_friction_override))
            if overrides:
                setters = [setter for _, _, setter in overrides]
                vector = isinstance(write, (declarations.GeomFrictionWrite, declarations.LinkComWrite))
                shape = (backend.data.num_worlds, len(setters)) + ((3,) if vector else ())
                self._buffers[key] = backend.position.new_zeros(shape)
                spin_roll = None
                if isinstance(write, declarations.GeomFrictionWrite):
                    spin_roll = backend.position.new_tensor(
                        [backend.host_model.get_geom(n).friction[1:] for n in write.geoms]
                    )
                for name, getter, setter in overrides:
                    identity = (type(write), name)
                    if identity not in backend._reset_parameters:
                        default = backend.position.new_tensor(getter(backend.data))
                        width = 3 if isinstance(write, declarations.LinkComWrite) else 1
                        default = default.reshape(-1, width).expand(backend.data.num_worlds, width).clone()
                        if width == 1:
                            default = default[:, 0]
                        backend._reset_parameters[identity] = setter, default
                self._parameters[key] = setters, spin_roll
                continue
            if isinstance(write, CtrlTargetsWrite):
                target = backend.control
                names = backend.model.actuator_names
                cols = range(len(names)) if write.actuators is None else [names.index(n) for n in write.actuators]
            elif isinstance(write, (JointPositionWrite, JointVelocityWrite)):
                position = isinstance(write, JointPositionWrite)
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.joints:
                    joint = backend.model.get_joint(name)
                    if joint is None or joint.num_dof_vel != 1:
                        raise ValueError(f"Expected a one-DOF joint for {key!r}, got {name!r}")
                    cols.append(joint.dof_pos_index if position else joint.dof_vel_index)
                self._refresh = True
            elif isinstance(write, (declarations.BodyJointPositionWrite, declarations.BodyJointVelocityWrite)):
                body = backend.model.get_body(write.body)
                position = isinstance(write, declarations.BodyJointPositionWrite)
                target = backend.position if position else backend.velocity
                cols = body.get_dof_pos_indices(False) if position else body.get_dof_vel_indices(False)
                self._refresh = True
            elif isinstance(
                write,
                (
                    declarations.BodyPositionWrite,
                    declarations.BodyRotationWrite,
                    declarations.BodyLinearVelocityWrite,
                    declarations.BodyAngularVelocityWrite,
                ),
            ):
                position = isinstance(write, (declarations.BodyPositionWrite, declarations.BodyRotationWrite))
                angular = isinstance(write, (declarations.BodyRotationWrite, declarations.BodyAngularVelocityWrite))
                width = 4 if isinstance(write, declarations.BodyRotationWrite) else 3
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.bodies:
                    base = backend.model.get_body(name).floatingbase
                    start = (base.dof_pos_start if position else base.dof_vel_start) + (3 if angular else 0)
                    cols.extend(range(start, start + width))
                shape = (len(write.bodies), width)
                self._refresh = True
            elif isinstance(write, (declarations.JointQuaternionWrite, declarations.JointAngularVelocityWrite)):
                position = isinstance(write, declarations.JointQuaternionWrite)
                width = 4 if position else 3
                target = backend.position if position else backend.velocity
                cols = []
                for name in write.joints:
                    joint = backend.model.get_joint(name)
                    start = joint.dof_pos_index if position else joint.dof_vel_index
                    cols.extend(range(start, start + width))
                shape = (len(write.joints), width)
                self._refresh = True
            else:
                raise NotImplementedError(f"MotrixSim CUDA cannot write {key!r} ({type(write).__name__})")
            self._buffers[key] = target.new_zeros((target.shape[0], *(shape or (len(cols),))))
            self._targets[key] = target, torch.tensor(list(cols), dtype=torch.long, device=target.device)
        self._refresh &= forward_kinematics

    def buffer(self, name):
        return self._buffers[name]

    def execute(self, env_ids=None):
        backend = self._backend
        rows = slice(None) if env_ids is None else env_ids
        if self._reset:
            backend.reset_rows(rows)
        for key, (target, columns) in self._targets.items():
            row_index = rows if env_ids is None else rows[:, None]
            target[row_index, columns] = self._buffers[key].flatten(1)[rows]
        if self._refresh or self._parameters:
            with backend.physics_stream():
                for key, (setters, spin_roll) in self._parameters.items():
                    values = self._buffers[key]
                    if spin_roll is not None:
                        if (values[rows, :, 1:] != spin_roll).any():
                            raise NotImplementedError("MotrixSim CUDA 1.0.0 supports only sliding-friction overrides")
                        values = values[:, :, 0]
                    if env_ids is not None:
                        selected = torch.full_like(values, torch.nan)
                        selected[rows] = values[rows]
                        values = selected
                    for column, setter in enumerate(setters):
                        setter(backend.data, wp.from_torch(values[:, column].contiguous()))
                for key, (bodies, components) in self._mocap.items():
                    for i, mocap in enumerate(bodies):
                        poses = wp.to_torch(mocap.get_pose(backend.data))
                        poses[rows, components] = self._buffers[key][rows, i]
                        backend.set_mocap_pose(mocap, poses, rows)
                if self._refresh:
                    backend.model.forward_kinematic(backend.data)


class CudaWriteCompiler:
    def __init__(self, backend):
        self.backend = backend

    def compile(self, writes, *, reset=False, forward_kinematics=True):
        return CudaWrites(self.backend, writes, reset, forward_kinematics)


class MotrixCudaBackend(SimBackend):
    name = "motrixsim"

    def __init__(self, scene, sim, num_envs, *, render=None):
        super().__init__(scene, sim, num_envs, render=render)
        device = torch.device(sim.device)
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        self._warp_device = wp.get_device(str(self.device))
        self._stream = wp.Stream(self._warp_device)
        self._torch_stream = torch.cuda.ExternalStream(self._stream.cuda_stream, device=self.device)
        host_model = MotrixSimSceneCompiler().compile(scene, sim, render=render)
        self.host_model = host_model
        self._model_compiler = MotrixSimModelCompiler(host_model)
        with self.physics_stream():
            self.model = gpu.SceneModel.from_cmodel(host_model)
            self.data = gpu.SceneData(self.model, batch=num_envs)
            self.position = wp.to_torch(self.data.dof_pos)
            self.velocity = wp.to_torch(self.data.dof_vel)
            self.control = wp.to_torch(self.data.actuator_ctrls)
        self.initial_position = self.position[0].clone()
        self._domains = dict(self.data.low.domains.items())
        self._contact_sensors = {sensor.name: sensor for sensor in host_model.low.low_contact_sensors}
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
        caller = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(self._warp_device), wp.ScopedStream(self._stream):
            self._stream.wait_stream(caller)
            try:
                with torch.cuda.stream(self._torch_stream):
                    yield
            finally:
                caller.wait_stream(self._stream)

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

    def sensor_reader(self, name):
        sensor = self._contact_sensors.get(name)
        if sensor is None:
            return lambda: wp.to_torch(self.model.get_sensor_value(name, self.data))
        report = sensor.report
        if (
            not report.found
            or any((report.force, report.torque, report.pos, report.normal, report.tangent, report.dist))
            or sensor.max_num != 1
        ):
            raise NotImplementedError(
                f"MotrixSim CUDA cannot provide contact sensor {name!r} with its requested report"
            )
        return self.contact_count_reader(sensor.collider_pairs)

    def contact_count_reader(self, collider_pairs):
        from motrix_env_motrixsim.gpu_contacts import count_geom_contacts

        collision = self._domains["rigid_collision"]
        pairs = wp.array(collider_pairs, dtype=wp.int32, device=self._warp_device).reshape((-1, 2))
        output = wp.zeros((self.data.num_worlds, 1), dtype=wp.float32, device=self._warp_device)

        def fetch():
            wp.launch(
                count_geom_contacts,
                self.data.num_worlds,
                inputs=[collision.contacts, collision.contact_count, pairs, output],
                device=self._warp_device,
            )
            return wp.to_torch(output)

        return fetch

    def sample_terrain_height(self, geom_name, env_ids, xy):
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

    def compile_model(self, queries):
        return _device_metadata(super().compile_model(queries), self.device)

    def compile_reads(self, queries):
        return CudaReads(self, queries)

    def step(self, substeps, result=None):
        with self.physics_stream():
            for _ in range(substeps):
                self.model.step(self.data, stream=self._stream)
        if result is not None:
            good = self.position.isfinite().all(dim=1) & self.velocity.isfinite().all(dim=1)
            result.copy_(~good | wp.to_torch(self._domains["rigid_collision"].overflow).ne(0))

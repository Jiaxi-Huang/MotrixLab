# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile CUDA read declarations into stable, selectively published snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import torch
import warp as wp

from motrix_env_core.sim.read import PhysicsReadProgram, SimDataQuery
from motrix_env_motrixsim.gpu_contacts import ContactReaders
from motrix_env_motrixsim.read_compiler import MotrixReadCompiler, canonical_reads

if TYPE_CHECKING:
    from motrix_env_motrixsim.gpu_runtime import MotrixCudaBackend

Projection = Callable[[tuple[torch.Tensor, ...]], torch.Tensor]


class CudaReadCompiler(MotrixReadCompiler):
    """Resolve names and bind device storage once for one SceneData instance."""

    def __init__(self, backend: MotrixCudaBackend, queries: Mapping[str, SimDataQuery]):
        self.backend = backend
        self._model = backend.host_model
        self.queries = dict(queries)
        self.fields: dict[str, Projection] = {}
        self.sources: list[Callable[[], torch.Tensor]] = []
        self.sensors: dict[str, Projection] = {}
        self.poses: dict[tuple[str, str], torch.Tensor] = {}
        self.contacts = ContactReaders(backend)
        self.link_poses = None

    def build(self) -> PhysicsReadProgram:
        queries, self.aliases = canonical_reads(self.queries)
        with self.backend.physics_stream():
            return self.compile(queries)

    def _build_program(self) -> PhysicsReadProgram:
        return CudaReads(self)

    def _source(self, fetch: Callable[[], torch.Tensor]) -> Projection:
        index = len(self.sources)
        self.sources.append(fetch)
        return lambda values: values[index]

    def _joint_columns(self, key, indices, target):
        columns = torch.as_tensor(indices, dtype=torch.long, device=target.device)
        self.fields[key] = lambda values: target[:, columns]

    def compile_body_joint_position(self, key: str, body: str) -> None:
        obj = self._get_body(self._validate_name(body, key), key)
        self._joint_columns(key, obj.get_dof_pos_indices(False), self.backend.position)

    def compile_body_joint_velocity(self, key: str, body: str) -> None:
        obj = self._get_body(self._validate_name(body, key), key)
        self._joint_columns(key, obj.get_dof_vel_indices(False), self.backend.velocity)

    def compile_joints_position(self, key: str, joints: tuple[str, ...]) -> None:
        resolved = self._one_dof_joints(key, joints, position=True)
        self._joint_columns(key, [joint.dof_pos_index for joint in resolved], self.backend.position)

    def compile_joints_velocity(self, key: str, joints: tuple[str, ...]) -> None:
        resolved = self._one_dof_joints(key, joints, position=False)
        self._joint_columns(key, [joint.dof_vel_index for joint in resolved], self.backend.velocity)

    def _link_pose(self, key, links, components):
        if self.link_poses is None:
            self.link_poses = wp.to_torch(self.backend.model.get_link_poses(self.backend.data))
        poses = self.link_poses
        if isinstance(links, str):
            index = self._get_link_index(self._validate_name(links, key), key)
        else:
            index = torch.tensor(self._link_indices(key, links), device=self.backend.device)
        self.fields[key] = lambda values: poses[:, index, components]

    def compile_links_position(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_pose(key, links, slice(3))

    def compile_links_quaternion(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_pose(key, links, slice(3, 7))

    def _link_velocity(self, key, links, method):
        single = isinstance(links, str)
        names = (links,) if single else links
        self._link_indices(key, names)
        velocities = tuple(wp.to_torch(method(self.backend.model.get_link(name))(self.backend.data)) for name in names)
        self.fields[key] = (lambda values: velocities[0]) if single else (lambda values: torch.stack(velocities, dim=1))

    def compile_links_linear_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_velocity(key, links, lambda link: link.get_linear_velocity)

    def compile_links_angular_velocity(self, key: str, links: str | tuple[str, ...]) -> None:
        self._link_velocity(key, links, lambda link: link.get_angular_velocity)

    def compile_links_net_contact_force(self, key: str, links: str | tuple[str, ...]) -> None:
        raise NotImplementedError(f"MotrixSim CUDA does not support link contact query {key!r}")

    def compile_body_link_net_contact_force(self, key: str, body: str, exclude_links: tuple[str, ...]) -> None:
        body = self._get_body(self._validate_name(body, key), key)
        excluded = set(self._validate_names(exclude_links, key, "excluded links", allow_empty=True))
        links = body.links
        unknown = excluded.difference(link.name for link in links)
        if unknown:
            raise KeyError(f"Simulator query {key!r} excludes unknown body links: {sorted(unknown)}.")
        self.fields[key] = self._source(
            self.contacts.link_contact_reader([link.index for link in links if link.name not in excluded])
        )

    def compile_sensor_values(self, key: str, sensors: tuple[str, ...]) -> None:
        for name in self._validate_names(sensors, key, "sensors"):
            if name not in self.sensors:
                fetch = self.contacts.sensor_reader(name)
                if fetch is None:
                    if name not in self._model.sensors.names:
                        raise KeyError(f"Simulator query {key!r} references unknown sensor {name!r}.")
                    view = wp.to_torch(self.backend.model.get_sensor_value(name, self.backend.data))
                    self.sensors[name] = lambda values, view=view: view
                else:
                    self.sensors[name] = self._source(fetch)
        sources = tuple(self.sensors[name] for name in sensors)
        self.fields[key] = (
            sources[0] if len(sources) == 1 else lambda values: torch.cat([s(values) for s in sources], dim=1)
        )

    def _pose(self, key, kind, name, components):
        token = (kind, name)
        if token not in self.poses:
            if kind == "geom":
                self._get_geom(self._validate_name(name, key), key)
                obj = self.backend.model.get_geom(name)
            else:
                self._get_site(self._validate_name(name, key), key)
                obj = self.backend.model.get_site(name)
            self.poses[token] = wp.to_torch(obj.get_pose(self.backend.data))
        pose = self.poses[token]
        self.fields[key] = lambda values: pose[:, components]

    def compile_site_position(self, key: str, site: str) -> None:
        self._pose(key, "site", site, slice(3))

    def compile_site_quaternion(self, key: str, site: str) -> None:
        self._pose(key, "site", site, slice(3, 7))

    def compile_geom_position(self, key: str, geom: str) -> None:
        self._pose(key, "geom", geom, slice(3))

    def compile_geom_quaternion(self, key: str, geom: str) -> None:
        self._pose(key, "geom", geom, slice(3, 7))

    def compile_geom_linear_velocity(self, key: str, geom: str) -> None:
        self._get_geom(self._validate_name(geom, key), key)
        obj = self.backend.model.get_geom(geom)
        velocity = wp.to_torch(obj.get_linear_velocity(self.backend.data))
        self.fields[key] = lambda values: velocity

    def compile_geom_pair_colliding(self, key: str, pairs: tuple[tuple[str, str], ...]) -> None:
        def colliders(name):
            return self._model.low.get_collider_indices(self._get_geom(name, key).index) or ()

        groups = [
            [(a, b) for a in colliders(left) for b in colliders(right)]
            for left, right in self._validate_name_pairs(pairs, key)
        ]
        self.fields[key] = self._source(self.contacts.pair_reader(groups))

    def compile_dof_position(self, key: str) -> None:
        target = self.backend.position
        self.fields[key] = lambda values: target

    def compile_dof_velocity(self, key: str) -> None:
        target = self.backend.velocity
        self.fields[key] = lambda values: target

    def compile_actuator_ctrl(self, key: str) -> None:
        target = self.backend.control
        self.fields[key] = lambda values: target


class CudaReads(PhysicsReadProgram):
    def __init__(self, compiler: CudaReadCompiler):
        backend = compiler.backend
        self._backend = backend
        self._queries = compiler.queries
        self._sources = tuple(compiler.sources)
        self._prepare_contacts = compiler.contacts.prepare_rows
        values = self._fetch()
        outputs = {key: project(values).to(torch.float32).clone() for key, project in compiler.fields.items()}
        self._outputs = {key: outputs[canonical] for key, canonical in compiler.aliases.items()}
        self._arena_bytes = sum(out.numel() * out.element_size() for out in outputs.values())
        fields = tuple((outputs[key], project) for key, project in compiler.fields.items())

        def publish(values):
            for output, project in fields:
                output.copy_(project(values))

        def publish_rows(values, rows):
            for output, project in fields:
                output[rows] = project(values)[rows].to(torch.float32)

        # Fuse indexed projection with the row scatter. Only the reset count varies.
        self._publish_rows = torch.compile(publish_rows, options={"triton.cudagraphs": False})

        # All sources either alias SceneData or write into persistent contact
        # buffers. Capture publication only; SDK solver bookkeeping stays live.
        self._values = values
        self._graph = None
        if fields:
            compiled = torch.compile(publish, dynamic=False, options={"triton.cudagraphs": False})
            compiled(values)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph, stream=backend._torch_stream):
                compiled(values)

    def _fetch(self) -> tuple[torch.Tensor, ...]:
        if self._prepare_contacts is not None:
            self._prepare_contacts()
        return tuple(fetch() for fetch in self._sources)

    @property
    def arena_bytes(self) -> int:
        return self._arena_bytes

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._queries)

    def query(self, key: str) -> SimDataQuery:
        return self._queries[key]

    def view(self, key: str) -> torch.Tensor:
        return self._outputs[key]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            if env_ids.dtype not in (torch.int32, torch.int64) or env_ids.ndim != 1:
                raise TypeError("Partial simulator read env_ids must be a one-dimensional integer tensor.")
            if env_ids.numel() == 0:
                return
        with self._backend.physics_stream():
            if self._sources:
                self._fetch()
            if env_ids is None:
                if self._graph is not None:
                    self._graph.replay()
            else:
                torch._dynamo.maybe_mark_dynamic(env_ids, 0)
                self._publish_rows(self._values, env_ids)

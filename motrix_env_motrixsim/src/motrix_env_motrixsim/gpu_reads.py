# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CUDA query projection and snapshot publication."""

import torch
import warp as wp

from motrix_env_core.sim import read
from motrix_env_core.sim.read import ActuatorCtrlQuery, DofPositionQuery, DofVelocityQuery, PhysicsReadProgram


class CudaReads(PhysicsReadProgram):
    def __init__(self, backend, queries):
        self._queries = dict(queries)
        self._sources = []
        self._link_poses = None
        self._outputs = {}
        self._backend = backend
        projections = {}
        with backend.physics_stream():
            for key, query in queries.items():
                if query not in projections:
                    projections[query] = self._reader(backend, key, query)
            values = self._fetch()
            for query, project in projections.items():
                output = project(values).to(torch.float32).clone()
                if self._sources:
                    torch._dynamo.mark_static_address(output)
                self._outputs[query] = output
        fields = tuple((self._outputs[q], project) for q, project in projections.items())
        self._fields = fields

        def publish(values):
            for output, project in fields:
                output.copy_(project(values))

        # Different environments specialize this closure for different query
        # layouts. Allow Dynamo to fall back when its per-code cache fills.
        if self._sources:
            self._publish = torch.compile(publish, dynamic=False, mode="reduce-overhead")
        elif self._fields:
            # All inputs here are persistent SceneData views or fixed model indices.
            # Capture only snapshot publication; initialization never advances physics.
            with backend.physics_stream():
                compiled = torch.compile(publish, dynamic=False, options={"triton.cudagraphs": False})
                compiled(())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=backend._torch_stream):
                    compiled(())
            self._publish = lambda values: graph.replay()
        else:
            self._publish = publish

    def _fetch(self):
        return tuple(fetch() for fetch in self._sources)

    def _source(self, fetch):
        index = len(self._sources)
        self._sources.append(fetch)
        return lambda values: values[index]

    def _reader(self, backend, key, query):
        model, data = backend.model, backend.data
        if isinstance(query, DofPositionQuery):
            return lambda values: backend.position
        if isinstance(query, DofVelocityQuery):
            return lambda values: backend.velocity
        if isinstance(query, ActuatorCtrlQuery):
            return lambda values: backend.control
        if isinstance(query, (read.BodyJointPositionQuery, read.BodyJointVelocityQuery)):
            body = model.get_body(query.body)
            position = isinstance(query, read.BodyJointPositionQuery)
            target = backend.position if position else backend.velocity
            ids = body.get_dof_pos_indices(False) if position else body.get_dof_vel_indices(False)
            cols = torch.as_tensor(ids.astype("int64"), device=target.device)
            return lambda values: target[:, cols]
        if isinstance(query, (read.JointPositionQuery, read.JointVelocityQuery)):
            position = isinstance(query, read.JointPositionQuery)
            target = backend.position if position else backend.velocity
            joints = [model.get_joint(n) for n in query.joints]
            ids = [j.dof_pos_index if position else j.dof_vel_index for j in joints]
            cols = torch.tensor(ids, device=target.device)
            return lambda values: target[:, cols]
        if isinstance(query, read.SensorValuesQuery):
            readers = [self._source(backend.sensor_reader(name)) for name in query.sensors]
            return lambda values: torch.cat([project(values) for project in readers], dim=1)
        if isinstance(query, (read.SitePositionQuery, read.SiteQuaternionQuery)):
            site = model.get_site(query.site)
            columns = slice(3, 7) if isinstance(query, read.SiteQuaternionQuery) else slice(3)
            pose = self._source(lambda: wp.to_torch(site.get_pose(data)))
            return lambda values: pose(values)[:, columns]
        if isinstance(query, (read.GeomPositionQuery, read.GeomQuaternionQuery, read.GeomLinearVelocityQuery)):
            geom = model.get_geom(query.geom)
            if isinstance(query, read.GeomLinearVelocityQuery):
                return self._source(lambda: wp.to_torch(geom.get_linear_velocity(data)))
            columns = slice(3, 7) if isinstance(query, read.GeomQuaternionQuery) else slice(3)
            pose = self._source(lambda: wp.to_torch(geom.get_pose(data)))
            return lambda values: pose(values)[:, columns]
        if isinstance(
            query,
            (
                read.LinkPositionQuery,
                read.LinkQuaternionQuery,
                read.BatchLinkPositionQuery,
                read.BatchLinkQuaternionQuery,
            ),
        ):
            # A single world-pose table serves all declared position/rotation
            # projections, including repeated or reordered links in a batch.
            if self._link_poses is None:
                self._link_poses = self._source(lambda: wp.to_torch(model.get_link_poses(data)))
            poses = self._link_poses
            columns = (
                slice(3) if isinstance(query, (read.LinkPositionQuery, read.BatchLinkPositionQuery)) else slice(3, 7)
            )
            if isinstance(query, (read.LinkPositionQuery, read.LinkQuaternionQuery)):
                link = model.get_link(query.link).index
                return lambda values: poses(values)[:, link, columns]
            links = torch.tensor(
                [model.get_link(n).index for n in query.links], dtype=torch.long, device=backend.device
            )
            return lambda values: poses(values)[:, links, columns]
        for single, batch, method in (
            (read.LinkLinearVelocityQuery, read.BatchLinkLinearVelocityQuery, lambda link: link.get_linear_velocity),
            (read.LinkAngularVelocityQuery, read.BatchLinkAngularVelocityQuery, lambda link: link.get_angular_velocity),
        ):
            if isinstance(query, single):
                fetch = method(model.get_link(query.link))
                return self._source(lambda: wp.to_torch(fetch(data)))
            if isinstance(query, batch):
                getters = [method(model.get_link(n)) for n in query.links]
                sources = [self._source(lambda get=get: wp.to_torch(get(data))) for get in getters]
                return lambda values: torch.stack([project(values) for project in sources], dim=1)
        if isinstance(query, read.BodyLinkNetContactForceQuery):
            body = backend.host_model.get_body(query.body)
            if body is None:
                raise KeyError(f"Simulator query {key!r} references unknown body {query.body!r}.")
            links = body.links
            excluded = set(query.exclude_links)
            unknown = excluded.difference(link.name for link in links)
            if unknown:
                raise KeyError(f"Simulator query {key!r} excludes unknown body links: {sorted(unknown)}.")
            return self._source(
                backend.link_contact_reader([link.index for link in links if link.name not in excluded])
            )
        if isinstance(query, read.GeomPairCollidingQuery):
            low = backend.host_model.low
            readers = [
                backend.contact_count_reader(
                    [(a, b) for a in low.get_collider_indices(first) for b in low.get_collider_indices(second)]
                )
                for first, second in query.pairs
            ]
            sources = [self._source(fetch) for fetch in readers]
            return lambda values: torch.cat([project(values) > 0 for project in sources], dim=1)
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
        with self._backend.physics_stream():
            values = self._fetch()
            if env_ids is None:
                self._publish(values)
            else:
                for output, project in self._fields:
                    output[env_ids] = project(values)[env_ids].to(torch.float32)

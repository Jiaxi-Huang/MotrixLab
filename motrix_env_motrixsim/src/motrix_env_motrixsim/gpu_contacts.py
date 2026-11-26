# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Read contact reports from the motrixsim-gpu rigid contact solver."""

from typing import Any

import warp as wp


@wp.kernel
def read_geom_contacts(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    collider_pairs: wp.array2d(dtype=wp.int32),
    normal_report: bool,
    result: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    found = wp.int32(0)
    normal = wp.vec3()
    for i in range(wp.min(counts[world], contacts.shape[1])):
        pair = contacts[world, i].pair_index
        for j in range(collider_pairs.shape[0]):
            a = collider_pairs[j, 0]
            b = collider_pairs[j, 1]
            if (pair[0] == a and pair[1] == b) or (pair[0] == b and pair[1] == a):
                if normal_report and found == 0:
                    normal = contacts[world, i].normal
                    if pair[0] != a:
                        normal = -normal
                found += 1
                break
    if normal_report:
        for axis in range(3):
            result[world, axis] = normal[axis]
    else:
        result[world, 0] = float(found)


@wp.kernel
def dense_contact_rows(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    slots: wp.array2d(dtype=wp.int32),
    records: wp.array2d(dtype=Any),
    record_counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
):
    world, i = wp.tid()
    if i >= counts[world]:
        return
    contact = contacts[world, i]
    for axis in range(6):
        rows[world, i, axis] = -1
    slot = slots[world, i]
    for j in range(wp.min(record_counts[world], records.shape[1])):
        r = records[world, j]
        if int(r.pos) == slot + 1 and r.feature_id == contact.feature_id and int(r.midx) == contact.index_in_manifold:
            rows[world, i, int(r.axis)] = int(r.lineno)


@wp.kernel
def link_contact_wrenches(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
    impulses: wp.array2d(dtype=wp.float32),
    poses: wp.array3d(dtype=wp.float32),
    local_com: wp.array3d(dtype=wp.float32),
    high_links: wp.array(dtype=wp.int32),
    low_links: wp.array(dtype=wp.int32),
    inv_dt: float,
    output: wp.array3d(dtype=wp.float32),
):
    world, column = wp.tid()
    link = high_links[column]
    low = low_links[column]
    origin = wp.vec3(poses[world, link, 0], poses[world, link, 1], poses[world, link, 2])
    rotation = wp.quat(poses[world, link, 3], poses[world, link, 4], poses[world, link, 5], poses[world, link, 6])
    com = origin + wp.quat_rotate(
        rotation, wp.vec3(local_com[world, link, 0], local_com[world, link, 1], local_com[world, link, 2])
    )
    force = wp.vec3()
    moment = wp.vec3()
    for i in range(wp.min(counts[world], rows.shape[1])):
        c = contacts[world, i]
        if c.link_index[0] != low and c.link_index[1] != low:
            continue
        n = c.normal
        # motrixsim-gpu 1.0.0 resolves tangent impulses in this world-frame basis.
        reference = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(n[0]) >= 0.9:
            reference = wp.vec3(0.0, 1.0, 0.0)
        t0 = wp.normalize(wp.cross(n, reference))
        t1 = wp.cross(n, t0)
        linear = wp.vec3()
        angular = wp.vec3()
        for axis in range(6):
            row = rows[world, i, axis]
            if row >= 0:
                direction = n
                if axis == 1 or axis == 4:
                    direction = t0
                elif axis == 2 or axis == 5:
                    direction = t1
                value = direction * impulses[world, row]
                if axis < 3:
                    linear += value
                else:
                    angular += value
        if c.link_index[0] == low:
            force -= linear
            moment -= angular + wp.cross(c.point - com, linear)
        if c.link_index[1] == low:
            force += linear
            moment += angular + wp.cross(c.point + n * c.penetration_depth - com, linear)
    for j in range(3):
        output[world, column, j] = force[j] * inv_dt
        output[world, column, 3 + j] = moment[j] * inv_dt


@wp.kernel
def touch_force(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    rows: wp.array3d(dtype=wp.int32),
    impulses: wp.array2d(dtype=wp.float32),
    frames: wp.array(dtype=Any),
    frame_count: int,
    frame_id: int,
    colliders: wp.array(dtype=wp.int32),
    size: wp.vec3,
    box: bool,
    inv_dt: float,
    output: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    frame = frames[world * frame_count + frame_id]
    total = wp.float32(0.0)
    for i in range(wp.min(counts[world], contacts.shape[1])):
        c = contacts[world, i]
        row = rows[world, i, 0]
        if row < 0 or row >= impulses.shape[1]:
            continue
        for j in range(colliders.shape[0]):
            collider = colliders[j]
            if collider != c.pair_index[0] and collider != c.pair_index[1]:
                continue
            point = c.point
            if collider == c.pair_index[1]:
                point = point + c.normal * c.penetration_depth
            local = wp.transpose(frame.rotation) * (point - frame.translation)
            inside = wp.length_sq(local) <= size[0] * size[0]
            if box:
                inside = wp.abs(local[0]) <= size[0] and wp.abs(local[1]) <= size[1] and wp.abs(local[2]) <= size[2]
            if inside:
                total += wp.abs(impulses[world, row])
                break
    output[world, 0] = total * inv_dt


@wp.kernel
def geom_pair_hits(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    pairs: wp.array2d(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float32),
):
    world, group = wp.tid()
    for i in range(wp.min(counts[world], contacts.shape[1])):
        pair = contacts[world, i].pair_index
        for j in range(offsets[group], offsets[group + 1]):
            a, b = pairs[j, 0], pairs[j, 1]
            if (pair[0] == a and pair[1] == b) or (pair[0] == b and pair[1] == a):
                output[world, group] = 1.0
                return
    output[world, group] = 0.0


class ContactReaders:
    """Bind solver storage and share one dense row lookup per snapshot refresh."""

    def __init__(self, backend):
        self.backend = backend
        self._rows = None
        self.prepare_rows = None

    def pair_reader(self, groups):
        offsets = [0]
        flat = []
        for group in groups:
            flat.extend(group)
            offsets.append(len(flat))
        backend = self.backend
        if not flat:
            result = backend.position.new_zeros((backend.data.num_worlds, len(groups)))
            return lambda: result
        device = backend._warp_device
        pairs = wp.array(flat, dtype=wp.int32, device=device).reshape((-1, 2))
        bounds = wp.array(offsets, dtype=wp.int32, device=device)
        shape = (backend.data.num_worlds, len(groups))
        output = wp.empty(shape, dtype=wp.float32, device=device)
        result = wp.to_torch(output)
        collision = backend._domains["rigid_collision"]

        def fetch():
            wp.launch(
                geom_pair_hits,
                shape,
                inputs=[collision.contacts, collision.contact_count, pairs, bounds, output],
                device=device,
            )
            return result

        return fetch

    def sensor_reader(self, name):
        if name in self.backend._touch_sensors:
            return self.touch_reader(self.backend._touch_sensors[name])
        sensor = self.backend._contact_sensors.get(name)
        if sensor is None:
            return None
        report = sensor.report
        if (
            report.normal
            and not any((report.found, report.force, report.torque, report.pos, report.tangent, report.dist))
            and sensor.max_num == 1
            and sensor.reduce == "none"
        ):
            return self.geom_contact_reader(sensor.collider_pairs, normal=True)
        if (
            not report.found
            or any((report.force, report.torque, report.pos, report.normal, report.tangent, report.dist))
            or sensor.max_num != 1
        ):
            raise NotImplementedError(
                f"MotrixSim CUDA cannot provide contact sensor {name!r} with its requested report"
            )
        return self.geom_contact_reader(sensor.collider_pairs)

    def link_contact_reader(self, indices):
        if not indices:
            empty = self.backend.position.new_empty((self.backend.data.num_worlds, 0, 6))
            return lambda: empty

        collision = self.backend._domains["rigid_collision"]
        solver = self.backend._domains["solver"]
        maps = self.backend.host_model.low.link_low_maps
        # A high-level link can span virtual joints. Its colliders belong to the final rigid link.
        low = [maps[i].index + maps[i].num - 1 for i in indices]
        high_ids = wp.array(indices, dtype=wp.int32, device=self.backend._warp_device)
        low_ids = wp.array(low, dtype=wp.int32, device=self.backend._warp_device)
        output = wp.zeros(
            (self.backend.data.num_worlds, len(indices), 6), dtype=wp.float32, device=self.backend._warp_device
        )
        result = wp.to_torch(output)

        rows = self._contact_rows()
        poses = self.backend.model.get_link_poses(self.backend.data)

        def fetch():
            wp.launch(
                link_contact_wrenches,
                (self.backend.data.num_worlds, len(indices)),
                inputs=[
                    collision.contacts,
                    collision.contact_count,
                    rows,
                    solver.constraint_lambdas.unilateral,
                    poses,
                    self.backend._domains["overrides"].apparent_link_com,
                    high_ids,
                    low_ids,
                    1.0 / self.backend.model.timestep,
                    output,
                ],
                device=self.backend._warp_device,
            )
            return result

        return fetch

    def _contact_rows(self):
        if self._rows is not None:
            return self._rows
        collision = self.backend._domains["rigid_collision"]
        solver = self.backend._domains["solver"]
        sparse = solver.sparse_jacobian
        if sparse is not None:
            self._rows = sparse.contact_rows.rows
        else:
            rows = wp.empty((*collision.contacts.shape, 6), dtype=wp.int32, device=self.backend._warp_device)
            warm = solver.persistent_lambdas

            def prepare():
                wp.launch(
                    dense_contact_rows,
                    collision.contacts.shape,
                    inputs=[collision.contacts, collision.contact_count, warm.slots, warm.rows, warm.row_counter, rows],
                    device=self.backend._warp_device,
                )

            self._rows = rows
            self.prepare_rows = prepare
        return self._rows

    def touch_reader(self, sensor):
        shape = sensor.shape.shape_type
        if shape not in ("sphere", "box") or sensor.loc.location_type != "Frame":
            raise NotImplementedError(f"Unsupported touch sensor geometry for {sensor.name!r}: {shape}")
        collision = self.backend._domains["rigid_collision"]
        impulses = self.backend._domains["solver"].constraint_lambdas.unilateral
        sensors = self.backend._domains["sensors"]
        colliders = wp.array(sensor.colliders, dtype=wp.int32, device=self.backend._warp_device)
        output = wp.zeros((self.backend.data.num_worlds, 1), dtype=wp.float32, device=self.backend._warp_device)
        result = wp.to_torch(output)
        rows = self._contact_rows()
        size = wp.vec3(*sensor.size)

        def fetch():
            wp.launch(
                touch_force,
                self.backend.data.num_worlds,
                inputs=[
                    collision.contacts,
                    collision.contact_count,
                    rows,
                    impulses,
                    sensors.base_to_frames,
                    sensors.frame_len,
                    sensor.loc.location_id,
                    colliders,
                    size,
                    shape == "box",
                    1.0 / self.backend.model.timestep,
                    output,
                ],
                device=self.backend._warp_device,
            )
            return result

        return fetch

    def geom_contact_reader(self, collider_pairs, *, normal=False):
        collision = self.backend._domains["rigid_collision"]
        if not collider_pairs:
            empty = self.backend.position.new_zeros((self.backend.data.num_worlds, 3 if normal else 1))
            return lambda: empty
        pairs = wp.array(collider_pairs, dtype=wp.int32, device=self.backend._warp_device).reshape((-1, 2))
        output = wp.zeros(
            (self.backend.data.num_worlds, 3 if normal else 1), dtype=wp.float32, device=self.backend._warp_device
        )

        result = wp.to_torch(output)

        def fetch():
            wp.launch(
                read_geom_contacts,
                self.backend.data.num_worlds,
                inputs=[collision.contacts, collision.contact_count, pairs, normal, output],
                device=self.backend._warp_device,
            )
            return result

        return fetch

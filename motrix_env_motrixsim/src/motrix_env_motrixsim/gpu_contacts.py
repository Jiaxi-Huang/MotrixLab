# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Read contact reports from the motrixsim-gpu rigid contact solver."""

from typing import Any

import warp as wp


@wp.kernel
def count_geom_contacts(
    contacts: wp.array2d(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    collider_pairs: wp.array2d(dtype=wp.int32),
    result: wp.array2d(dtype=wp.float32),
):
    world = wp.tid()
    found = wp.int32(0)
    for i in range(wp.min(counts[world], contacts.shape[1])):
        pair = contacts[world, i].pair_index
        for j in range(collider_pairs.shape[0]):
            a = collider_pairs[j, 0]
            b = collider_pairs[j, 1]
            if (pair[0] == a and pair[1] == b) or (pair[0] == b and pair[1] == a):
                found += 1
                break
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

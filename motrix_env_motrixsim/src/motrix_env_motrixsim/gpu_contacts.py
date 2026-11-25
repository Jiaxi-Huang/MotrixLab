# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Count actual contact points for MotrixSim's found-only geom-pair sensors."""

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

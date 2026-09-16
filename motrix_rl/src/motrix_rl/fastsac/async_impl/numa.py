# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Best-effort NUMA binding for multi-collector async FastSAC training.

Each collector on a multi-NUMA-node server should run its env, staging buffers
and pinned host allocations with node-local memory, and the scheduler should not
migrate it across nodes. This module applies the ``numactl --cpunodebind=
--membind=`` equivalent from inside the worker process:

* CPU affinity via ``os.sched_setaffinity`` (portable stdlib);
* memory policy via libnuma's ``set_membind`` (the same call ``numactl`` uses),
  loaded with :mod:`ctypes` — future allocations of the calling process come
  from the bound node.

Everything is best-effort: when libnuma is unavailable or the node is unknown
the binding logs a warning and continues with OS default placement, so single
NUMA machines and containers keep working unchanged.

Binding must happen at process start, before the env and any staging buffer is
allocated — ``set_membind`` only affects future allocations.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import re
from pathlib import Path

_NODE_SYSFS = Path("/sys/devices/system/node")

logger = logging.getLogger(__name__)

_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_cpulist(text: str) -> list[int]:
    """Parse a sysfs ``cpulist`` (e.g. ``"0-3,8,10-11"``) into CPU ids."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        match = _RANGE_RE.match(part)
        if match:
            cpus.extend(range(int(match.group(1)), int(match.group(2)) + 1))
        else:
            cpus.append(int(part))
    return cpus


def available_numa_nodes() -> list[int]:
    """NUMA node ids visible in sysfs; empty when the host is not NUMA-aware."""
    if not _NODE_SYSFS.is_dir():
        return []
    return sorted(int(p.name.removeprefix("node")) for p in _NODE_SYSFS.iterdir() if p.name.startswith("node"))


def numa_node_cpus(node: int) -> list[int]:
    """CPU ids of a NUMA node; raises ``ValueError`` for unknown nodes."""
    cpulist = _NODE_SYSFS / f"node{node}" / "cpulist"
    if not cpulist.is_file():
        raise ValueError(f"NUMA node {node} does not exist ({cpulist} missing)")
    return parse_cpulist(cpulist.read_text())


def select_collector_cpus(
    base: list[int],
    collector_id: int,
    num_collectors: int,
    cpus_per_collector: int | None,
) -> list[int]:
    """Pick this collector's CPU slice from ``base`` (its node's or the process's CPUs).

    Without ``cpus_per_collector`` all collectors share the full ``base`` set
    and the OS load-balances them; with it, the set is split into contiguous
    chunks so collectors never compete for the same cores.
    """
    if not base:
        raise ValueError("empty CPU set for collector binding")
    if cpus_per_collector is None:
        return list(base)
    if cpus_per_collector <= 0:
        raise ValueError(f"cpus_per_collector must be positive, got {cpus_per_collector}")
    start = collector_id * cpus_per_collector
    end = min(start + cpus_per_collector, len(base))
    if start >= end:
        raise ValueError(
            f"cpus_per_collector={cpus_per_collector} leaves no CPUs for collector {collector_id} "
            f"(base set has {len(base)} CPUs)"
        )
    return base[start:end]


def apply_cpu_affinity(cpus: list[int], role: str) -> None:
    """Pin the current process to ``cpus``, intersected with allowed CPUs."""
    allowed = os.sched_getaffinity(0)
    selected = sorted(set(cpus) & allowed)
    if not selected:
        raise ValueError(
            f"{role}: none of the requested CPUs {sorted(cpus)} are allowed (process affinity is {sorted(allowed)})"
        )
    os.sched_setaffinity(0, selected)
    if set(cpus) - allowed:
        logger.warning("%s: dropped non-allowed CPUs %s from binding", role, sorted(set(cpus) - allowed))


def _load_libnuma():
    try:
        return ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return None


def _membind(bitmask_ptr, role: str, what: str, node: int | None = None) -> bool:
    libnuma = _load_libnuma()
    if libnuma is None:
        logger.warning("%s: libnuma unavailable; %s left to the OS", role, what)
        return False
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    if libnuma.numa_set_membind(ctypes.c_void_p(bitmask_ptr)) != 0:
        logger.warning("%s: numa_set_membind(%s) failed; %s not bound", role, node, what)
        return False
    return True


def _node_bitmask(node: int):
    libnuma = _load_libnuma()
    if libnuma is None:
        return None, None
    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        return None, None
    libnuma.numa_bitmask_setbit(ctypes.c_void_p(mask), ctypes.c_uint(node))
    return libnuma, mask


def _free_bitmask(libnuma, mask) -> None:
    libnuma.numa_bitmask_free(ctypes.c_void_p(mask))


def apply_memory_policy(node: int, role: str) -> None:
    """Bind future allocations of this process to ``node`` (libnuma ``numa_set_membind``)."""
    libnuma, mask = _node_bitmask(node)
    if libnuma is None:
        logger.warning("%s: libnuma unavailable; memory policy left to the OS (node %d not bound)", role, node)
        return
    try:
        _membind(mask, role, "memory policy", node)
    finally:
        _free_bitmask(libnuma, mask)


@contextlib.contextmanager
def spawn_placement(node: int | None, role: str):
    """Place a spawn-started child on ``node`` by pre-placing its parent.

    A ``spawn`` child re-imports torch / the simulator / glibc arenas before
    the worker entry function runs, so a bind executed inside the child only
    affects allocations made after those imports — simulator thread pools and
    malloc arenas created at import time land on a random node and physics
    stepping pays cross-node access forever. Affinity and memory policy both
    survive fork+exec, so briefly switching the *parent* onto ``node`` around
    ``Process.start()`` makes the child's import-time allocations node-local;
    the worker's own :func:`bind_process` call then re-affirms the same
    binding (a no-op refinement).
    """
    if node is None:
        yield
        return
    saved_affinity = os.sched_getaffinity(0)
    try:
        cpus = sorted(set(numa_node_cpus(node)) & saved_affinity)
        if cpus:
            os.sched_setaffinity(0, cpus)
        libnuma, mask = _node_bitmask(node)
        if libnuma is not None and mask:
            try:
                _membind(mask, role, "spawn placement", node)
            finally:
                _free_bitmask(libnuma, mask)
        yield
    finally:
        os.sched_setaffinity(0, saved_affinity)
        libnuma = _load_libnuma()
        if libnuma is not None:
            # restore the default "any node" policy for the parent
            try:
                all_nodes = ctypes.c_void_p.in_dll(libnuma, "numa_all_nodes_ptr")
                libnuma.numa_set_membind(all_nodes)
            except (ValueError, OSError, AttributeError):
                pass


def bind_process(
    role: str, node: int | None, cpus_per_collector: int | None = None, collector_id: int = 0, num_collectors: int = 1
) -> None:
    """Apply CPU affinity + memory policy for a trainer worker process.

    ``node=None`` only applies CPU chunking when ``cpus_per_collector`` is set
    (chunked across the process's current affinity); with neither option this is
    a no-op.
    """
    if node is None:
        if cpus_per_collector is not None:
            base = sorted(os.sched_getaffinity(0))
            cpus = select_collector_cpus(base, collector_id, num_collectors, cpus_per_collector)
            apply_cpu_affinity(cpus, role)
        return
    cpus = numa_node_cpus(node)
    cpus = select_collector_cpus(cpus, collector_id, num_collectors, cpus_per_collector)
    apply_cpu_affinity(cpus, role)
    apply_memory_policy(node, role)


def validate_numa_options(
    num_collectors: int, numa_nodes: list[int] | None, learner_numa_node: int | None
) -> list[int] | None:
    """Validate the NUMA config in the parent process; returns the node list to use.

    Raises ``ValueError`` when the node list does not match ``num_collectors`` or
    references nodes the kernel does not expose.
    """
    nodes = available_numa_nodes()
    if numa_nodes is not None:
        if len(numa_nodes) != num_collectors:
            raise ValueError(
                f"numa_nodes must have exactly num_collectors={num_collectors} entries, got {len(numa_nodes)}"
            )
        unknown = [n for n in numa_nodes if n not in nodes]
        if unknown:
            raise ValueError(f"numa_nodes reference unknown NUMA node(s) {unknown}; available nodes: {nodes}")
    if learner_numa_node is not None and learner_numa_node not in nodes:
        raise ValueError(f"learner_numa_node={learner_numa_node} is unknown; available nodes: {nodes}")
    return numa_nodes

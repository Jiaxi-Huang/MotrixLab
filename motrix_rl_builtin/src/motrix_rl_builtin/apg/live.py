# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Host ``sys/*`` scalars the APG trainers add to their live metrics.

The fixed key list and units follow ``wiki/design/live-metrics.md`` (same as
FastSAC's panel extras): CPU load percent, resident memory in GiB, and
per-GPU utilization / memory in GiB. Per-core CPU load is left out on
purpose — it would spend the service's per-run key budget on large hosts.
"""

from __future__ import annotations

from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsageSampler,
    sample_gpu_devices,
)

_GIB = 1024**3


class SystemScalars:
    """Holds the host samplers; :meth:`sample` yields the ``sys/*`` dict.

    One instance per trainer, constructed at training start; ``sample()`` is
    passed as ``live_extra`` to :meth:`TrainingLogger.scalars` so the values
    go live only (TensorBoard never stores host curves).
    """

    def __init__(self) -> None:
        self._cpu = CpuLoadSampler()
        self._gpu = GpuUtilizationSampler()
        self._memory = MemoryUsageSampler()
        self._gpu_memory = GpuMemoryUsageSampler()

    def sample(self) -> dict[str, float]:
        scalars: dict[str, float] = {}
        cpu = self._cpu.sample()
        if cpu is not None:
            scalars["sys/cpu_percent"] = cpu.utilization_percent
        memory = self._memory.sample()
        if memory is not None:
            scalars["sys/mem_used_gb"] = memory.used_bytes / _GIB
        devices = sample_gpu_devices(self._gpu, self._gpu_memory) or ()
        for device in devices:
            if device.utilization_percent is not None:
                scalars[f"sys/gpu{device.index}/util_percent"] = device.utilization_percent
            if device.memory is not None:
                scalars[f"sys/gpu{device.index}/mem_used_gb"] = device.memory.used_bytes / _GIB
        return scalars


__all__ = ["SystemScalars"]

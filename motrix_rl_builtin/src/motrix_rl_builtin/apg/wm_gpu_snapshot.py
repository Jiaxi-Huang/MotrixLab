# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Persistent GPU snapshot storage for the optional WM worker."""

from __future__ import annotations

import torch


class GPUSnapshot:
    def __init__(self, state, device, buffers=None):
        self.device = torch.device(device)
        self.layout = []
        totals = {}
        for name, value in state.items():
            offset = totals.get(value.dtype, 0)
            self.layout.append((name, value.dtype, offset, value.numel(), tuple(value.shape)))
            totals[value.dtype] = offset + value.numel()
        self.buffers = buffers or {
            dtype: torch.empty(count, dtype=dtype, device=self.device) for dtype, count in totals.items()
        }

    @torch.no_grad()
    def pack(self, state) -> None:
        for name, dtype, offset, count, shape in self.layout:
            self.buffers[dtype].narrow(0, offset, count).view(shape).copy_(state[name])

    @torch.no_grad()
    def install(self, module, staging) -> None:
        for dtype, source in self.buffers.items():
            staging[dtype].copy_(source)
        state = module.state_dict()
        for name, dtype, offset, count, shape in self.layout:
            state[name].copy_(staging[dtype].narrow(0, offset, count).view(shape))

    def staging(self, device):
        return {dtype: torch.empty_like(value, device=device) for dtype, value in self.buffers.items()}

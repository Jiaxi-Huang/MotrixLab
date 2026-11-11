# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph update capture for the APG learner (upstream supermimic's
``CapturedUpdate`` adapted to this provider).

The imagination unroll is dispatch-bound: ~25 term dispatches, FK, and
optimizer bookkeeping per step, times the horizon, all Python-launched. A
captured update records the whole update — forward, backward, gradient
clips, and optimizer steps included — once and replays it as one graph; the
Python interpreter never re-runs the loop.

Contract (upstream's, unchanged):

- ``fn`` must be capture-safe: fixed shapes, no host syncs (no ``.item()``,
  ``bool(tensor)``, CPU folds), every tensor it reads besides the staged
  inputs updated in place. Python values it reads are frozen at capture.
- Inputs are nested structures of tensors; the first call clones them into
  static buffers, later calls overwrite those buffers in place.
- Outputs are static tensors, overwritten by each replay.
- All captured updates on a device record on one stream into one memory
  pool and must replay in the order they were recorded.
- A replay raises if a tensor in ``state()`` moved (parameters, buffers,
  optimizer state, lr tensors — anything the graph mutates or reads).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence

import torch


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


def _stage(template: object, value: object) -> object:
    """Clone ``value`` into a static buffer shaped like ``template``."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: _stage(template[key], entry) for key, entry in value.items()}
    if isinstance(value, tuple):
        return tuple(_stage(template[index], entry) for index, entry in enumerate(value))
    raise TypeError(f"CapturedUpdate inputs must be tensors or nested dicts/tuples, got {type(value).__name__}")


def _restage(staged: object, value: object) -> None:
    """Overwrite the static buffers in place from a fresh ``value``."""
    if isinstance(value, torch.Tensor):
        staged.copy_(value)
    elif isinstance(value, Mapping):
        for key, entry in value.items():
            _restage(staged[key], entry)
    else:
        for staged_item, item in zip(staged, value, strict=True):
            _restage(staged_item, item)


def _identity(value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [value.data_ptr()]
    if isinstance(value, Mapping):
        return [ptr for entry in value.values() for ptr in _identity(entry)]
    return [ptr for entry in value for ptr in _identity(entry)]


class CapturedUpdate:
    """Runs ``fn`` eagerly for its first ``warmup`` calls (real updates, on
    the capture side stream), records it once as a CUDA graph, and replays it
    on subsequent calls. Non-CUDA devices fall back to plain eager calls."""

    def __init__(
        self,
        fn: Callable[[object], Mapping[str, torch.Tensor]],
        state: Callable[[], Sequence[torch.Tensor]],
        warmup: int = 3,
    ):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._staged: object | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: Mapping[str, torch.Tensor] = {}

    def __call__(self, inputs: object) -> Mapping[str, torch.Tensor]:
        if self._staged is None:
            self._staged = _stage(inputs, inputs)
        else:
            _restage(self._staged, inputs)
        device = _first_tensor_device(self._staged)
        if device is None or device.type != "cuda":
            return self._fn(self._staged)
        with torch.cuda.device(device):
            stream, pool = _capture_stream_and_pool(device)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._staged)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._staged)
                self._graph = graph
                self._identity = _identity(self._state())
            elif _identity(self._state()) != self._identity:
                raise RuntimeError("a tensor a captured update mutates moved; rebuild the capture")
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


def _first_tensor_device(value: object) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, Mapping):
        for entry in value.values():
            found = _first_tensor_device(entry)
            if found is not None:
                return found
        return None
    for entry in value:
        found = _first_tensor_device(entry)
        if found is not None:
            return found
    return None

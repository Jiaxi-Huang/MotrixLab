# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Parent-side coordinator for the optional independent WM worker."""

from __future__ import annotations

import time
from queue import Empty, Full
from typing import TYPE_CHECKING

import torch

from motrix_rl_builtin.apg.async_train import cpu_tree, pack_owned, unpack_owned
from motrix_rl_builtin.apg.wm_gpu_snapshot import GPUSnapshot
from motrix_rl_builtin.apg.wm_worker import wm_worker

if TYPE_CHECKING:
    from motrix_rl_builtin.apg.learner import ApgLearner


class WMClient:
    def __init__(self, ctx, trainer, learner: ApgLearner, stop):
        self.cfg = trainer
        self.learner = learner
        self.stop = stop
        self.commands = ctx.Queue(maxsize=trainer.queue_blocks + 1)
        self.results = ctx.Queue(maxsize=1)
        self.errors, tx = ctx.Pipe(duplex=False)
        initial = {
            "wm": {key: value.detach().cpu().clone() for key, value in learner.wm.state_dict().items()},
            "opt_wm": cpu_tree(learner.opt_wm.state_dict()),
            "version": learner.iteration,
            "rng": {
                "cpu": torch.get_rng_state(),
                "device": torch.cuda.get_rng_state(trainer.wm_device).cpu(),
            },
        }
        self.gpu_snapshot = GPUSnapshot(learner.wm.state_dict(), trainer.wm_device)
        self.staging = self.gpu_snapshot.staging(learner.device)
        self.process = ctx.Process(
            target=wm_worker,
            name="apg-wm-worker",
            args=(
                trainer.env_spec,
                learner.num_envs,
                trainer.seed,
                trainer.cfg,
                trainer.wm_device,
                pack_owned(initial),
                self.commands,
                self.results,
                stop,
                tx,
                self.gpu_snapshot.buffers,
            ),
        )
        self.process.start()
        tx.close()

    def check(self) -> None:
        if self.errors.poll():
            raise RuntimeError("APG WM worker failed:\n" + self.errors.recv())
        if self.process.exitcode is not None:
            raise RuntimeError(f"APG WM worker exited unexpectedly ({self.process.exitcode})")

    def send(self, message: dict) -> None:
        payload = pack_owned(message)
        deadline = time.monotonic() + self.cfg.worker_timeout
        while not self.stop.is_set():
            self.check()
            try:
                self.commands.put(payload, timeout=0.1)
                return
            except Full:
                if time.monotonic() >= deadline:
                    raise TimeoutError("WM command queue timeout")
        raise RuntimeError("WM submission cancelled")

    def receive(self, kind: str, version: int, watermark: int) -> dict:
        deadline = time.monotonic() + self.cfg.worker_timeout
        while not self.stop.is_set():
            self.check()
            try:
                result = unpack_owned(self.results.get(timeout=0.1))
            except Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("WM result timeout")
                continue
            if (result["kind"], result["version"], result["watermark"]) != (kind, version, watermark):
                raise RuntimeError("Unexpected WM result or replay watermark")
            return result
        raise RuntimeError("WM result cancelled")

    def install(self) -> None:
        self.gpu_snapshot.install(self.learner.wm, self.staging)
        torch.cuda.synchronize(self.learner.device)

    def close(self) -> None:
        self.stop.set()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        for queue in (self.commands, self.results):
            queue.cancel_join_thread()
            queue.close()
        self.errors.close()

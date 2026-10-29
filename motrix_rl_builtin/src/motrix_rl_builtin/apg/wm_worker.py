# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Spawned world-model owner for the optional dual-GPU trainer path."""

from __future__ import annotations

import time
import traceback
from queue import Empty

import torch

from motrix_rl_builtin.apg.async_train import cpu_tree, ingest_block, pack_owned, unpack_owned
from motrix_rl_builtin.apg.contracts import build_head
from motrix_rl_builtin.apg.learner import ApgLearner
from motrix_rl_builtin.apg.wm_gpu_snapshot import GPUSnapshot


def wm_worker(spec, num_envs, seed, cfg, device, initial, commands, results, stop, errors, gpu_buffers):
    try:
        torch.set_num_threads(cfg.trainer.collector_threads)
        torch.cuda.set_device(device)
        env = spec.make(num_envs=1, mode="train", seed=seed)
        head = build_head(cfg.head, env)
        learner = ApgLearner(
            env,
            cfg.agent,
            head,
            device=torch.device(device),
            num_envs=num_envs,
            seed=None if seed is None else seed + 100003,
        )
        state = unpack_owned(initial)
        learner.wm.load_state_dict(state["wm"])
        learner.opt_wm.load_state_dict(state["opt_wm"])
        if "rng" in state:
            torch.set_rng_state(state["rng"]["cpu"])
            torch.cuda.set_rng_state(state["rng"]["device"], device)
        version = int(state["version"])
        sequence = 0
        snapshot = GPUSnapshot(learner.wm.state_dict(), device, gpu_buffers)
        while not stop.is_set():
            try:
                message = unpack_owned(commands.get(timeout=0.1))
            except Empty:
                continue
            kind = message["kind"]
            if kind == "ingest":
                ingest_block(learner, message["block"], sequence)
                sequence += 1
                continue
            if kind == "train":
                if message["version"] != version + 1 or message["watermark"] != sequence:
                    raise RuntimeError("WM version or replay watermark mismatch")
                for group in learner.opt_wm.param_groups:
                    group["lr"] = message["lr"]
                started = time.monotonic()
                for batch in range(cfg.agent.wm_batches):
                    loss, error = learner.wm_update(return_tensor=batch < cfg.agent.wm_batches - 1)
                torch.cuda.synchronize(device)
                version += 1
                snapshot.pack(learner.wm.state_dict())
                torch.cuda.synchronize(device)
                result = {
                    "kind": "done",
                    "version": version,
                    "watermark": sequence,
                    "loss": float(loss.detach()),
                    "error": float(error.detach()),
                    "train_s": time.monotonic() - started,
                }
            elif kind == "snapshot":
                if message["version"] != version or message["watermark"] != sequence:
                    raise RuntimeError("WM snapshot barrier mismatch")
                result = {
                    "kind": "snapshot",
                    "version": version,
                    "watermark": sequence,
                    "wm": {key: value.detach().cpu().clone() for key, value in learner.wm.state_dict().items()},
                    "opt_wm": cpu_tree(learner.opt_wm.state_dict()),
                    "rng": {
                        "cpu": torch.get_rng_state(),
                        "device": torch.cuda.get_rng_state(device).cpu(),
                    },
                }
            else:
                raise ValueError(f"Unknown WM command: {kind}")
            results.put(pack_owned(result))
    except BaseException:
        errors.send(traceback.format_exc())
    finally:
        errors.close()
        results.cancel_join_thread()

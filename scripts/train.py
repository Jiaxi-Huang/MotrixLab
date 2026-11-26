#!/usr/bin/env python
# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Temporary jobd entry for paired FastSAC training on the Wuhan two-GPU node."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = "3ca072f9bcc97640926392089e5f366aae6a4c6f"
CANDIDATE = "c0c71b94822da1fc9af0db77852d4ee2aa5cd5da"
STEPS = 600


def result(work, envs, cards):
    (metadata_path,) = list((work / "runs").rglob("metadata.json"))
    run = metadata_path.parent
    metadata = json.loads(metadata_path.read_text())
    perf = metadata["performance"]
    assert perf["total_env_steps"] == STEPS * envs, perf
    assert len(set(metadata["collector_sim_devices"])) == cards, metadata["collector_sim_devices"]
    assert (run / "checkpoints" / f"model_{STEPS:07}.pt").is_file()
    (events,) = list(run.rglob("events.out.tfevents.*"))
    log = EventAccumulator(str(events), size_guidance={"scalars": 0}).Reload()
    losses = log.Scalars("train/qf_loss")
    assert losses and all(math.isfinite(point.value) for point in losses)
    utd = log.Scalars("async/utd")[-1].value
    assert utd > 0, utd
    window = [x for x in losses if 120 <= x.step <= 520]
    steady = None
    if len(window) >= 2 and window[-1].step > window[0].step:
        start, end = window[0], window[-1]
        steady = (end.step - start.step) * envs / (end.wall_time - start.wall_time)
    return {
        **perf,
        "steady_steps_per_s": steady,
        "utd": utd,
        "collectors": metadata["collector_sim_devices"],
        "cpu_binding": metadata["system"]["cpu"]["bound_cores"],
        "devices": metadata["system"]["gpus_used"],
        "run_dir": str(run.resolve()),
    }


def main():
    here = Path.cwd()
    out = (here / "runs").resolve()
    cases = out.with_name(out.name + "-cases")
    cases.mkdir()
    baseline = Path("/tmp/motrixlab-accept-base")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(baseline), BASE],
        check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
    )
    subprocess.run(["git", "lfs", "checkout", "motrix_robots/**", "motrix_envs/**"], cwd=baseline, check=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip() == CANDIDATE
    results = []
    for task in ("go2-walk-flat", "g1-wbt-dance"):
        for cards, envs in ((1, 256), (2, 256), (2, 512)):
            versions = (("base", baseline), ("candidate", here))
            if cards == 2 and envs == 256:
                versions = tuple(reversed(versions))
            for version, checkout in versions:
                label = f"{task}-{cards}x{envs}-{version}"
                final = task == "g1-wbt-dance" and cards == 2 and envs == 512 and version == "candidate"
                work = here if final else cases / label
                work.mkdir(exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(str(p.resolve()) for p in sorted(checkout.glob("*/src")))
                args = [
                    "mxlab",
                    "train",
                    "--config-dir",
                    str(checkout / "configs"),
                    f"task={task}/motrix.fastsac",
                    "task.train_backend=torch",
                    "sim_device=cuda",
                    f"num_envs={envs}",
                    "seed=42",
                    f"algo.trainer.num_learning_iterations={STEPS}",
                    "algo.agent.batch_size=1024",
                    "algo.agent.buffer_size=256",
                    f"algo.trainer.async_options.num_learners={cards}",
                    f"algo.trainer.async_options.num_collectors={cards}",
                    "algo.trainer.async_options.learner_devices=[" + ",".join(f"cuda:{i}" for i in range(cards)) + "]",
                    "logging.interval=20",
                    f"checkpoint.interval={STEPS}",
                ]
                log = cases / (label + ".log")
                print("ACCEPT_START", label, flush=True)
                try:
                    with log.open("w") as stream:
                        subprocess.run(
                            args, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=420, check=True
                        )
                    stats = result(work, envs, cards)
                except Exception as error:
                    stats = {"error": repr(error)}
                    print(log.read_text(errors="replace")[-4000:], flush=True)
                row = {"label": label, **stats}
                print("ACCEPT_RESULT", json.dumps(row), flush=True)
                results.append(row)
                (cases / "summary.json").write_text(json.dumps(results, indent=2))
    return int(any("error" in row for row in results))


if __name__ == "__main__":
    sys.exit(main())

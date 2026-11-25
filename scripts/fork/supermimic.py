# Copyright (C) 2026 Motphys Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a merge of a configured fork integration baseline into supermimic in an isolated worktree."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from sync import ancestor, git, guard, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="main-supermimic")
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--motion-file", required=True, type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    guard()
    if "supermimic" not in profile(args.baseline)["downstream"]:
        raise ValueError("Selected baseline does not declare supermimic as a downstream branch")
    python = str(args.python.resolve(strict=True))
    motion = str(args.motion_file.resolve(strict=True))
    git("fetch", "origin", args.baseline, "supermimic")
    main_sha, old = git("rev-parse", f"origin/{args.baseline}"), git("rev-parse", "origin/supermimic")
    if args.baseline_sha != main_sha:
        raise RuntimeError("Requested SHA is no longer the selected baseline tip")
    directory = Path(tempfile.mkdtemp(prefix="supermimic-validation-"))
    checkout = directory / "checkout"
    git("worktree", "add", "--detach", str(checkout), old)
    print(f"Validation checkout and logs: {directory}", flush=True)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OMP_NUM_THREADS="2")
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in checkout.glob("*/src"))

    def run(command):
        with (directory / "validation.log").open("a") as log:
            subprocess.run(command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    run(["git", "merge", "--no-ff", "--no-edit", main_sha])
    run([python, "-m", "pytest", "-m", "not integration"])
    run([python, "-m", "pytest", "motrix_rl_builtin/tests", "-m", "integration", "-k", "apg"])
    run([python, "scripts/fork/apg_smoke.py", motion])
    run(["prek", "run", "--all-files"])
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
        raise RuntimeError("Validation modified checkout; commit fixes and validate again")
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if args.publish:
        git("fetch", "origin", args.baseline, "supermimic")
        if git("rev-parse", f"origin/{args.baseline}") != main_sha or git("rev-parse", "origin/supermimic") != old:
            raise RuntimeError("Source changed during validation")
        if not ancestor(old, candidate):
            raise RuntimeError("Only fast-forward publication is supported")
        git("push", "origin", f"{candidate}:refs/heads/supermimic")
    print(f"Validated {candidate}; published={args.publish}")


if __name__ == "__main__":
    main()

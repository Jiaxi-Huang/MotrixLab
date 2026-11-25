# Copyright (C) 2026 Motphys Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and publish tested fork integration candidates. Never writes upstream."""

import argparse
import json
import os
import subprocess
from pathlib import Path

FORK = "Jiaxi-Huang/MotrixLab"
UPSTREAM = "https://github.com/Motphys/MotrixLab.git"
CONFIG = "scripts/fork/integrations.json"


def state_path(baseline):
    return f"scripts/fork/integrated/{baseline}.json"


def profile(baseline):
    profiles = json.loads(Path(CONFIG).read_text())
    if not baseline.startswith("main-") or baseline not in profiles:
        raise ValueError("Choose a configured main-* integration baseline")
    return profiles[baseline]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def ancestor(old, new):
    return subprocess.run(["git", "merge-base", "--is-ancestor", old, new], check=False).returncode == 0


def guard():
    if os.environ.get("GITHUB_REPOSITORY", FORK) != FORK:
        raise RuntimeError("This tool only publishes to the configured fork")
    url = git("remote", "get-url", "--push", "origin")
    if url.removesuffix(".git") not in (f"https://github.com/{FORK}", f"git@github.com:{FORK}"):
        raise RuntimeError(f"Unexpected origin push URL: {url}")


def snapshot(baseline, topics):
    git("fetch", "origin", "+refs/heads/*:refs/remotes/origin/*", "--prune")
    git("fetch", UPSTREAM, "main")
    target = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/remotes/origin/{baseline}"], capture_output=True, text=True
    )
    return {
        "target": target.stdout.strip() if target.returncode == 0 else None,
        "main": git("rev-parse", "origin/main"),
        "upstream": git("rev-parse", "FETCH_HEAD"),
        "topics": {name: git("rev-parse", f"refs/remotes/origin/{name}") for name in topics},
    }


def prepare(args):
    guard()
    if git("status", "--porcelain"):
        raise RuntimeError("Candidate checkout must be clean")
    baseline = args.baseline
    topics = profile(baseline)["topics"]
    state = state_path(baseline)
    source = snapshot(baseline, topics)
    start = source["target"] or source["main"]
    Path(args.output).write_text(json.dumps({"changed": False, "source": source}, indent=2) + "\n")
    previous = subprocess.run(["git", "show", f"{start}:{state}"], capture_output=True, text=True)
    if previous.returncode == 0:
        old = json.loads(previous.stdout)
        for name, sha in old["topics"].items():
            if name in source["topics"] and not ancestor(sha, source["topics"][name]):
                raise RuntimeError(f"Topic history rewritten: {name}; requires manual reconciliation")
        if not ancestor(old["upstream"], source["upstream"]):
            raise RuntimeError("Upstream history rewritten; requires manual reconciliation")
    refs = [source["main"], source["upstream"], *source["topics"].values()]
    if source["target"] and not args.resolved_candidate and all(ancestor(sha, start) for sha in refs):
        Path(args.output).write_text(json.dumps({"changed": False}))
        return
    if args.resolved_candidate:
        git("checkout", "--detach", args.resolved_candidate)
        if not all(ancestor(sha, "HEAD") for sha in [start, *refs]):
            raise RuntimeError("Resolved candidate must contain its baseline and every source SHA")
    else:
        git("checkout", "--detach", start)
        for sha in refs:
            git("merge", "--no-ff", "--no-edit", sha)
    Path(state).parent.mkdir(parents=True, exist_ok=True)
    Path(state).write_text(json.dumps({k: source[k] for k in ("upstream", "topics")}, indent=2) + "\n")
    git("add", state)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], check=False).returncode:
        git("commit", "-m", "chore(fork): record integrated source revisions")
    sha = git("rev-parse", "HEAD")
    ref = f"integration/{baseline}/candidate-{args.run_id}"
    git("push", "origin", f"{sha}:refs/heads/{ref}")
    result = {"baseline": baseline, "changed": True, "candidate": sha, "ref": ref, "source": source}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    git("bundle", "create", str(Path(args.output).with_suffix(".bundle")), "HEAD", "--branches", "--remotes")


def publish(args):
    guard()
    result = json.loads(Path(args.manifest).read_text())
    baseline = result["baseline"]
    profile(baseline)
    if snapshot(baseline, result["source"]["topics"]) != result["source"]:
        raise RuntimeError("Source refs changed during validation; leave integration baseline unchanged")
    sha = result["candidate"]
    if git("rev-parse", f"origin/{result['ref']}") != sha or not ancestor(
        result["source"]["target"] or result["source"]["main"], sha
    ):
        raise RuntimeError("Candidate changed or is not a fast-forward")
    # The workflow calls this only after all checks on this exact SHA succeed.
    git("push", "origin", f"{sha}:refs/heads/{baseline}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--baseline", required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--resolved-candidate")
    release = commands.add_parser("publish")
    release.add_argument("--manifest", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "publish": publish}[args.command](args)


if __name__ == "__main__":
    main()

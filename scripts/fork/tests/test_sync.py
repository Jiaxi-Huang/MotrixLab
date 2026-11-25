# Copyright (C) 2026 Motphys Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Git integration tests for the fork's maintained publication contract."""

import importlib.util
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
spec = importlib.util.spec_from_file_location("fork_sync", Path(__file__).parents[1] / "sync.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-b", "main"], check=True, capture_output=True)
    sync.git("config", "user.name", "test")
    sync.git("config", "user.email", "test@example.com")
    Path("scripts/fork").mkdir(parents=True)
    Path(sync.CONFIG).write_text(
        json.dumps(
            {
                "main-supermimic": {"topics": ["topic"], "downstream": ["supermimic"]},
                "main-other": {"topics": [], "downstream": []},
            }
        )
    )
    sync.git("add", ".")
    sync.git("commit", "-m", "base")
    base = sync.git("rev-parse", "HEAD")
    monkeypatch.setattr(sync, "guard", lambda: None)
    return base


def test_unchanged_sources_skip_candidate(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sync, "snapshot", lambda *_: {"target": repo, "main": repo, "upstream": repo, "topics": {"topic": repo}}
    )
    output = tmp_path.parent / "result.json"
    sync.prepare(Namespace(baseline="main-supermimic", resolved_candidate=None, output=str(output), run_id="test"))
    assert json.loads(output.read_text()) == {"changed": False}


def test_publish_rejects_source_race(repo, tmp_path, monkeypatch):
    source = {"target": repo, "main": repo, "upstream": repo, "topics": {"topic": repo}}
    manifest = tmp_path.parent / "manifest.json"
    manifest.write_text(
        json.dumps({"baseline": "main-supermimic", "source": source, "candidate": repo, "ref": "candidate"})
    )
    monkeypatch.setattr(sync, "snapshot", lambda *_: {**source, "topics": {"topic": "changed"}})
    with pytest.raises(RuntimeError, match="Source refs changed"):
        sync.publish(Namespace(manifest=manifest))


def test_rewritten_topic_rejected(repo, tmp_path, monkeypatch):
    Path("feature").write_text("old")
    sync.git("add", ".")
    sync.git("commit", "-m", "old topic")
    old = sync.git("rev-parse", "HEAD")
    Path(sync.state_path("main-supermimic")).parent.mkdir(parents=True)
    Path(sync.state_path("main-supermimic")).write_text(json.dumps({"upstream": repo, "topics": {"topic": old}}))
    sync.git("add", ".")
    sync.git("commit", "-m", "integrated")
    main = sync.git("rev-parse", "HEAD")
    monkeypatch.setattr(
        sync, "snapshot", lambda *_: {"target": main, "main": repo, "upstream": repo, "topics": {"topic": repo}}
    )
    with pytest.raises(RuntimeError, match="history rewritten"):
        sync.prepare(
            Namespace(
                baseline="main-supermimic",
                resolved_candidate=None,
                output=str(tmp_path.parent / "result.json"),
                run_id="test",
            )
        )


@pytest.mark.parametrize("conflict", [False, True])
def test_topics_merge_or_stop_without_publishing(repo, tmp_path, monkeypatch, conflict):
    tips = {}
    for name in ("first", "second"):
        sync.git("checkout", "--detach", repo)
        Path("shared" if conflict else name).write_text(name)
        sync.git("add", ".")
        sync.git("commit", "-m", name)
        tips[name] = sync.git("rev-parse", "HEAD")
    sync.git("checkout", "--detach", repo)
    monkeypatch.setattr(sync, "snapshot", lambda *_: {"target": repo, "main": repo, "upstream": repo, "topics": tips})
    real_git = sync.git
    pushes = []

    def git(*args):
        if args[0] == "push":
            pushes.append(args)
            return ""
        return real_git(*args)

    monkeypatch.setattr(sync, "git", git)
    output = tmp_path.parent / "result.json"
    args = Namespace(baseline="main-supermimic", resolved_candidate=None, output=str(output), run_id="test")
    if conflict:
        with pytest.raises(subprocess.CalledProcessError):
            sync.prepare(args)
        assert not pushes
        assert sync.git("diff", "--name-only", "--diff-filter=U") == "shared"
        assert json.loads(output.read_text())["source"]["topics"] == tips
    else:
        sync.prepare(args)
        result = json.loads(output.read_text())
        assert result["changed"]
        assert all(sync.ancestor(tip, result["candidate"]) for tip in tips.values())
        assert all("refs/heads/main" not in push[-1] for push in pushes)
        assert Path("first").read_text() == "first"
        assert Path("second").read_text() == "second"


def test_target_selection_is_explicit(repo, monkeypatch):
    assert sync.profile("main-other")["topics"] == []
    assert sync.state_path("main-other") != sync.state_path("main-supermimic")
    with pytest.raises(ValueError):
        sync.profile("main")
    with pytest.raises(ValueError):
        sync.profile("main-missing")


def test_publication_only_updates_selected_baseline(repo, tmp_path, monkeypatch):
    source = {"target": repo, "main": repo, "upstream": repo, "topics": {}}
    manifest = tmp_path.parent / "publish.json"
    manifest.write_text(
        json.dumps(
            {
                "baseline": "main-other",
                "source": source,
                "candidate": repo,
                "ref": "integration/main-other/candidate-test",
            }
        )
    )
    monkeypatch.setattr(sync, "snapshot", lambda baseline, topics: source)
    real_git = sync.git
    pushes = []

    def git(*args):
        if args[0] == "push":
            pushes.append(args)
            return ""
        if args == ("rev-parse", "origin/integration/main-other/candidate-test"):
            return repo
        return real_git(*args)

    monkeypatch.setattr(sync, "git", git)
    sync.publish(Namespace(manifest=manifest))
    assert pushes == [("push", "origin", f"{repo}:refs/heads/main-other")]

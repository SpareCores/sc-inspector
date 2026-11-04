"""git subprocesses must never hang forever (no network: a fake `git` on PATH hangs)."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

import repo


@pytest.fixture
def hanging_git(tmp_path: Path, monkeypatch):
    """Put a `git` on PATH that sleeps (like a stalled fetch/push)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "git"
    # like `git push` over SSH: a child process (ssh) inherits git's output pipes
    fake.write_text("#!/bin/sh\nsleep 30 &\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return fake


def test_network_and_local_commands_get_different_timeouts():
    assert repo.git_timeout(["fetch", "origin", "main"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo.git_timeout(["push", "origin", "HEAD:main"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo.git_timeout(["ls-remote", "--heads", "origin", "main"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo.git_timeout(["clone", "--depth", "1", "url", "path"]) == repo.GIT_CLONE_TIMEOUT_SEC
    assert repo.git_timeout(["status", "--porcelain"]) == repo.GIT_LOCAL_TIMEOUT_SEC
    assert repo.git_timeout(["config", "--global", "gc.auto", "0"]) == repo.GIT_LOCAL_TIMEOUT_SEC
    assert repo.GIT_NETWORK_TIMEOUT_SEC < repo.GIT_LOCAL_TIMEOUT_SEC


def test_hanging_git_is_killed_and_reported_as_failed_command(hanging_git, tmp_path):
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as exc:
        repo.run_git_command(["push", "origin", "HEAD:main"], cwd=tmp_path, timeout=0.5)
    assert time.monotonic() - started < 10
    assert exc.value.returncode == -9
    assert "timed out" in exc.value.stderr


def test_default_timeout_is_applied(hanging_git, tmp_path, monkeypatch):
    monkeypatch.setattr(repo, "GIT_NETWORK_TIMEOUT_SEC", 0.5)
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        repo.run_git_command(["fetch", "origin"], cwd=tmp_path)
    assert time.monotonic() - started < 10


def test_fast_commands_unaffected(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert repo.run_git_command(["status", "--porcelain"], cwd=tmp_path) == ""

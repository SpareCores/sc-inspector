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


def test_locks_left_by_killed_git_are_removed_but_older_ones_kept(tmp_path, monkeypatch):
    work = tmp_path / "work"
    refs = work / ".git" / "refs" / "remotes" / "origin"
    refs.mkdir(parents=True)
    # a lock that existed before our command: not ours, must survive
    old_lock = work / ".git" / "packed-refs.lock"
    old_lock.write_text("")
    past = time.time() - 3600
    os.utime(old_lock, (past, past))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "git"
    # killed mid-fetch: holds the index lock and the remote-tracking ref lock
    fake.write_text(
        "#!/bin/sh\n"
        "touch .git/index.lock .git/refs/remotes/origin/main.lock\n"
        "sleep 30\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(subprocess.CalledProcessError):
        repo.run_git_command(["fetch", "origin"], cwd=work, timeout=0.5)

    assert not (work / ".git" / "index.lock").exists()
    assert not (refs / "main.lock").exists()
    assert old_lock.exists()


def test_lock_cleanup_from_subdirectory_finds_repo_root(tmp_path):
    sub = tmp_path / "work" / "data" / "x"
    sub.mkdir(parents=True)
    (tmp_path / "work" / ".git").mkdir()
    assert repo._git_dir(sub) == str(tmp_path / "work" / ".git")
    assert repo._git_dir(tmp_path) is None


def test_git_env_adds_ssh_keepalives_and_http_stall_limit():
    env = repo._git_env({"PATH": "/usr/bin"})
    assert env["GIT_SSH_COMMAND"].startswith("ssh ")
    assert "ServerAliveInterval=15" in env["GIT_SSH_COMMAND"]
    assert "ConnectTimeout=30" in env["GIT_SSH_COMMAND"]
    assert env["GIT_HTTP_LOW_SPEED_LIMIT"] == "1000"
    assert env["GIT_HTTP_LOW_SPEED_TIME"] == "60"
    assert env["PATH"] == "/usr/bin"


def test_git_env_respects_existing_settings():
    custom = repo._git_env({"GIT_SSH_COMMAND": "ssh -i /k", "GIT_HTTP_LOW_SPEED_TIME": "300"})
    assert custom["GIT_SSH_COMMAND"] == f"ssh -i /k {repo.GIT_SSH_TIMEOUT_OPTIONS}"
    assert custom["GIT_HTTP_LOW_SPEED_TIME"] == "300"

    already = repo._git_env({"GIT_SSH_COMMAND": "ssh -o ServerAliveInterval=5"})
    assert already["GIT_SSH_COMMAND"] == "ssh -o ServerAliveInterval=5"

    # a custom GIT_SSH program must not be overridden by GIT_SSH_COMMAND
    wrapper = repo._git_env({"GIT_SSH": "/usr/local/bin/my-ssh"})
    assert "GIT_SSH_COMMAND" not in wrapper


def test_env_reaches_git_subprocess(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "git"
    fake.write_text('#!/bin/sh\necho "$GIT_SSH_COMMAND|$GIT_HTTP_LOW_SPEED_LIMIT"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_SSH", raising=False)
    monkeypatch.delenv("GIT_HTTP_LOW_SPEED_LIMIT", raising=False)
    out = repo.run_git_command(["fetch", "origin"], cwd=tmp_path).strip()
    assert out == f"ssh {repo.GIT_SSH_TIMEOUT_OPTIONS}|1000"

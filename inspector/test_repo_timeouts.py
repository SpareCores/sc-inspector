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
    assert exc.value.returncode < 0  # terminated by a signal
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


def _fake_git(tmp_path, monkeypatch, script):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "git"
    fake.write_text("#!/bin/sh\n" + script)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def test_timeout_sends_sigterm_so_git_can_remove_its_own_locks(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / ".git").mkdir(parents=True)
    # like git: holds index.lock while working and removes it when it gets SIGTERM
    _fake_git(tmp_path, monkeypatch,
              "touch .git/index.lock\n"
              "trap 'rm -f .git/index.lock; exit 143' TERM\n"
              "sleep 30 & wait\n")
    with pytest.raises(subprocess.CalledProcessError):
        repo.run_git_command(["fetch", "origin"], cwd=work, timeout=0.5)
    assert not (work / ".git" / "index.lock").exists()


def test_timeout_terminates_children_of_git(tmp_path, monkeypatch):
    pidfile = tmp_path / "child.pid"
    # a child (like ssh) that would otherwise outlive the killed git
    _fake_git(tmp_path, monkeypatch, f"sleep 60 &\necho $! > {pidfile}\nsleep 60\n")
    with pytest.raises(subprocess.CalledProcessError):
        repo.run_git_command(["push", "origin", "HEAD:main"], cwd=tmp_path, timeout=0.5)
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("git's child process survived the timeout")


def test_timeout_keeps_partial_git_output_as_text(tmp_path, monkeypatch):
    _fake_git(tmp_path, monkeypatch,
              "echo 'Enumerating objects: 5, done.'\n"
              "echo 'Writing objects: 40% (2/5)' >&2\n"
              "sleep 30\n")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        repo.run_git_command(["push", "origin", "HEAD:main"], cwd=tmp_path, timeout=0.5)
    assert "Writing objects: 40% (2/5)" in exc.value.stderr
    assert "timed out after" in exc.value.stderr
    assert exc.value.stdout == "Enumerating objects: 5, done.\n"


def test_sigterm_ignoring_git_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(repo, "GIT_TERM_GRACE_SEC", 0.5)
    _fake_git(tmp_path, monkeypatch, "trap '' TERM\nsleep 30\n")
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        repo.run_git_command(["fetch", "origin"], cwd=tmp_path, timeout=0.5)
    assert time.monotonic() - started < 10


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


def test_push_paths_gives_up_on_hanging_push_and_next_push_works(tmp_path, monkeypatch):
    """The 2026-11-04 failure mode: a stalled push must not wedge the inspector."""
    import shutil

    from test_repo_squash import _init_bare_and_clone

    real_git = shutil.which("git")
    bare, work = _init_bare_and_clone(tmp_path)

    marker = tmp_path / "hang-push"
    marker.write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in -*) ;; *) sub="$a"; break;; esac; done\n'
        f'if [ "$sub" = "push" ] && [ -e "{marker}" ]; then sleep 30; fi\n'
        f'exec "{real_git}" "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)
    monkeypatch.setattr(repo, "GIT_NETWORK_TIMEOUT_SEC", 0.5)
    monkeypatch.setattr(repo, "PUSH_DEADLINE_SEC", 2)
    monkeypatch.setattr(repo, "PUSH_INITIAL_BACKOFF_SEC", 0.1)

    meta = work / "data" / "vendor" / "inst" / "0" / "task" / "meta.json"
    meta.write_text('{"start":"first"}\n')
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        repo.push_paths([work / "data" / "vendor" / "inst" / "0"], "first result")
    assert time.monotonic() - started < 20  # bounded, instead of hanging forever

    # GitHub recovers: the lock is free and the repo is not wedged by stale locks
    marker.unlink()
    meta.write_text('{"start":"second"}\n')
    repo.push_paths([work / "data" / "vendor" / "inst" / "0"], "second result")

    tip = tmp_path / "verify"
    subprocess.run([real_git, "clone", "-q", str(bare), str(tip)], check=True)
    assert (tip / "data/vendor/inst/0/task/meta.json").read_text() == '{"start":"second"}\n'

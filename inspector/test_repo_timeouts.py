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
    """The 2026-11-04 failure mode: a stalled push must not wedge the inspector.

    Only the push deadline is shrunk: the real local fetch/push/ls-remote calls
    keep their normal timeouts, so a slow CI runner can't make them time out.
    """
    from test_repo_squash import _init_bare_and_clone

    bare, work = _init_bare_and_clone(tmp_path)
    marker = tmp_path / "hang-push"
    marker.write_text("")
    real_git = _hang_push_wrapper(tmp_path, monkeypatch, marker)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)
    monkeypatch.setattr(repo, "PUSH_DEADLINE_SEC", 3)
    monkeypatch.setattr(repo, "PUSH_INITIAL_BACKOFF_SEC", 0.1)

    meta = work / "data" / "vendor" / "inst" / "0" / "task" / "meta.json"
    meta.write_text('{"start":"first"}\n')
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):
        repo.push_paths([work / "data" / "vendor" / "inst" / "0"], "first result")
    # bounded by the push deadline + SIGTERM grace, instead of hanging forever
    assert time.monotonic() - started < 3 + repo.GIT_TERM_GRACE_SEC + 10

    # GitHub recovers: the push_paths lock is free and the repo is not wedged
    marker.unlink()
    meta.write_text('{"start":"second"}\n')
    repo.push_paths([work / "data" / "vendor" / "inst" / "0"], "second result")

    tip = tmp_path / "verify"
    subprocess.run([real_git, "clone", "-q", str(bare), str(tip)], check=True)
    assert (tip / "data/vendor/inst/0/task/meta.json").read_text() == '{"start":"second"}\n'


def test_subcommand_skips_global_options_and_their_values():
    assert repo.git_subcommand(["-c", "http.lowSpeedTime=10", "push", "origin"]) == "push"
    assert repo.git_subcommand(["-C", "/repo", "fetch", "origin"]) == "fetch"
    assert repo.git_subcommand(["--git-dir", "/repo/.git", "ls-remote", "origin"]) == "ls-remote"
    assert repo.git_subcommand(["--no-pager", "-c", "a=b", "-C", "x", "status"]) == "status"
    assert repo.git_subcommand(["--git-dir=/repo/.git", "push"]) == "push"
    assert repo.git_subcommand([]) == ""
    assert repo.git_timeout(["-c", "core.sshCommand=ssh", "push", "origin"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo.git_timeout(["-C", "/repo", "clone", "url", "dir"]) == repo.GIT_CLONE_TIMEOUT_SEC


def test_failed_clone_removes_the_incomplete_directory(tmp_path, monkeypatch):
    target = tmp_path / "clone"
    # like a clone SIGKILLed half-way: leaves a partial .git behind
    _fake_git(tmp_path, monkeypatch,
              'for last in "$@"; do :; done\n'
              'mkdir -p "$last/.git/objects"\n'
              "trap '' TERM\n"
              "sleep 30\n")
    monkeypatch.setattr(repo, "GIT_CLONE_TIMEOUT_SEC", 0.5)
    monkeypatch.setattr(repo, "GIT_TERM_GRACE_SEC", 0.5)
    monkeypatch.setattr(repo, "_git_low_memory_configured", True)
    with pytest.raises(subprocess.CalledProcessError):
        repo._clone_repo("https://example.invalid/repo.git", str(target), None)
    assert not target.exists()


def test_failed_clone_keeps_a_directory_that_existed_before(tmp_path, monkeypatch):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("not ours")
    _fake_git(tmp_path, monkeypatch, "exit 128\n")
    monkeypatch.setattr(repo, "_git_low_memory_configured", True)
    with pytest.raises(subprocess.CalledProcessError):
        repo._clone_repo("https://example.invalid/repo.git", str(target), None)
    assert (target / "keep.txt").exists()


def test_deadline_caps_a_command_with_a_longer_default_timeout(hanging_git, tmp_path):
    # push's own timeout is 300s; the deadline 0.5s away must win
    started = time.monotonic()
    with repo.git_deadline(time.monotonic() + 0.5):
        with pytest.raises(subprocess.CalledProcessError) as exc:
            repo.run_git_command(["push", "origin", "HEAD:main"], cwd=tmp_path)
    assert time.monotonic() - started < 10
    assert "timed out" in exc.value.stderr


def test_passed_deadline_does_not_start_git(tmp_path, monkeypatch):
    marker = tmp_path / "git-ran"
    _fake_git(tmp_path, monkeypatch, f"touch {marker}\n")
    with repo.git_deadline(time.monotonic() - 1):
        with pytest.raises(subprocess.CalledProcessError) as exc:
            repo.run_git_command(["fetch", "origin"], cwd=tmp_path)
    assert "deadline already passed" in exc.value.stderr
    assert not marker.exists()


def test_nested_deadlines_keep_the_earliest_and_restore():
    with repo.git_deadline(100.0):
        with repo.git_deadline(200.0):
            assert repo._git_deadline_state.deadline == 100.0
        assert repo._git_deadline_state.deadline == 100.0
    assert getattr(repo._git_deadline_state, "deadline", None) is None


def _hang_push_wrapper(tmp_path, monkeypatch, marker):
    """A `git` that delegates to the real one but hangs on `push` while ``marker`` exists."""
    import shutil

    real_git = shutil.which("git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    wrapper = bin_dir / "git"
    wrapper.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in -*) ;; *) sub="$a"; break;; esac; done\n'
        f'if [ "$sub" = "push" ] && [ -e "{marker}" ]; then sleep 30; fi\n'
        f'exec "{real_git}" "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return real_git


def test_results_survive_a_push_that_gives_up(tmp_path, monkeypatch):
    """Retry without rewriting anything (like _push_data_dirs) must still publish the results."""
    from test_repo_squash import _init_bare_and_clone

    bare, work = _init_bare_and_clone(tmp_path)
    marker = tmp_path / "hang-push"
    marker.write_text("")
    real_git = _hang_push_wrapper(tmp_path, monkeypatch, marker)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)
    monkeypatch.setattr(repo, "PUSH_DEADLINE_SEC", 3)
    monkeypatch.setattr(repo, "PUSH_INITIAL_BACKOFF_SEC", 0.1)

    variant = work / "data" / "vendor" / "inst" / "0"
    (variant / "task" / "meta.json").write_text('{"start":"result"}\n')
    (variant / "task" / "stdout").unlink()  # a deletion must survive too

    with pytest.raises(subprocess.CalledProcessError):
        repo.push_paths([variant], "result")
    # results are back as uncommitted changes, so the retry has something to push
    assert repo.path_has_changes(str(work), variant)
    assert (variant / "task" / "meta.json").read_text() == '{"start":"result"}\n'

    marker.unlink()
    repo.push_paths([variant], "result (retry)")  # nothing rewritten in between

    tip = tmp_path / "verify-retry"
    subprocess.run([real_git, "clone", "-q", str(bare), str(tip)], check=True)
    assert (tip / "data/vendor/inst/0/task/meta.json").read_text() == '{"start":"result"}\n'
    assert not (tip / "data/vendor/inst/0/task/stdout").exists()
    assert (tip / "data/vendor/inst/0/task/keep.txt").exists()

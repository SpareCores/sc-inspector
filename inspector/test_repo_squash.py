"""Squash-push must publish deletions (stale stdout/stderr) without pathspec failure."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

import lib
import repo


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_bare_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "clone", str(bare), str(work))
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "test")
    # Variant layout: task artifacts under 0/, schedule metas mirrored for the scheduler.
    data = work / "data" / "vendor" / "inst" / "0" / "task"
    data.mkdir(parents=True)
    (data / "meta.json").write_text('{"start":"old","end":"old","exit_code":0}\n')
    (data / "stdout").write_text("stale-stdout-from-prior-run\n")
    (data / "keep.txt").write_text("untouched-by-this-inspector\n")
    sched = work / "data" / "vendor" / "inst" / "schedule" / "task"
    sched.mkdir(parents=True)
    (sched / "meta.json").write_text('{"start":"old","end":"old","exit_code":0}\n')
    _git(work, "add", ".")
    _git(work, "commit", "-m", "base")
    _git(work, "branch", "-M", "main")
    _git(work, "push", "-u", "origin", "main")
    # Bare repos need HEAD → main or subsequent clones land on an empty default.
    _git(bare, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(work, "fetch", "origin")
    return bare, work


def test_squash_publishes_stale_stdout_deletion(tmp_path: Path):
    """Reproduce the r2-240 failure: empty stream deletes stdout, squash must git-rm it."""
    _bare, work = _init_bare_and_clone(tmp_path)
    task_dir = work / "data" / "vendor" / "inst" / "0" / "task"

    # Inspector result commit: updated meta, new stderr, deleted stale stdout.
    (task_dir / "meta.json").write_text('{"start":"new","end":"new","exit_code":0}\n')
    (task_dir / "stderr").write_text("bench-noise\n")
    (task_dir / "stdout").unlink()
    _git(work, "add", "-A", "data/vendor/inst")
    _git(work, "commit", "-m", "local results")

    saved_head = _git(work, "rev-parse", "HEAD")
    rel = "data/vendor/inst"
    changed = repo._changed_files_under(str(work), rel)
    assert "data/vendor/inst/0/task/stdout" in changed
    assert "data/vendor/inst/0/task/stderr" in changed
    assert "data/vendor/inst/0/task/meta.json" in changed

    # Concurrent tip advance on a path this squash must preserve.
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(_bare), str(other))
    _git(other, "config", "user.email", "test@example.com")
    _git(other, "config", "user.name", "test")
    keep = other / "data" / "vendor" / "inst" / "0" / "task" / "keep.txt"
    assert keep.is_file()
    keep.write_text("concurrent-edit\n")
    _git(other, "add", ".")
    _git(other, "commit", "-m", "concurrent")
    _git(other, "push", "origin", "main")

    _git(work, "fetch", "origin", "+refs/heads/main:refs/remotes/origin/main")
    repo._squash_commit_and_push(str(work), rel, "Inspecting", changed, saved_head)

    assert not (task_dir / "stdout").exists()
    assert (task_dir / "stderr").read_text() == "bench-noise\n"
    assert '"end":"new"' in (task_dir / "meta.json").read_text()
    assert (task_dir / "keep.txt").read_text() == "concurrent-edit\n"

    tip = tmp_path / "verify"
    _git(tmp_path, "clone", str(_bare), str(tip))
    tip_task = tip / "data" / "vendor" / "inst" / "0" / "task"
    assert not (tip_task / "stdout").exists()
    assert (tip_task / "stderr").read_text() == "bench-noise\n"
    assert (tip_task / "keep.txt").read_text() == "concurrent-edit\n"


def test_squash_retry_uses_saved_head_after_hard_reset(tmp_path: Path):
    """After a failed attempt's reset --hard, retries must still restore from saved_head."""
    _bare, work = _init_bare_and_clone(tmp_path)
    task_dir = work / "data" / "vendor" / "inst" / "0" / "task"

    (task_dir / "meta.json").write_text('{"done":1}\n')
    (task_dir / "stdout").unlink()
    _git(work, "add", "-A", "data/vendor/inst")
    _git(work, "commit", "-m", "local results")
    saved_head = _git(work, "rev-parse", "HEAD")
    changed = repo._changed_files_under(str(work), "data/vendor/inst")

    # Mimic a failed first squash: reset wiped the working tree to origin.
    _git(work, "reset", "--hard", "origin/main")
    assert (task_dir / "stdout").exists()
    assert _git(work, "rev-parse", "HEAD") != saved_head

    repo._squash_commit_and_push(
        str(work), "data/vendor/inst", "retry", changed, saved_head
    )
    assert not (task_dir / "stdout").exists()
    assert '"done":1' in (task_dir / "meta.json").read_text()


def test_push_paths_publishes_variant_and_schedule(tmp_path: Path, monkeypatch):
    """Starting-task dual-write must land 0/ and schedule/ in one squash push."""
    bare, work = _init_bare_and_clone(tmp_path)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)

    zero_meta = work / "data" / "vendor" / "inst" / "0" / "task" / "meta.json"
    sched_meta = work / "data" / "vendor" / "inst" / "schedule" / "task" / "meta.json"
    payload = '{"start":"real","end":null,"exit_code":null}\n'
    zero_meta.write_text(payload)
    sched_meta.write_text(payload)

    repo.push_paths(
        [
            work / "data" / "vendor" / "inst" / "0",
            work / "data" / "vendor" / "inst" / "schedule",
        ],
        "Starting task",
    )

    tip = tmp_path / "verify-multi"
    _git(tmp_path, "clone", str(bare), str(tip))
    assert (tip / "data/vendor/inst/0/task/meta.json").read_text() == payload
    assert (tip / "data/vendor/inst/schedule/task/meta.json").read_text() == payload


def test_push_data_dirs_pushes_all_roots_at_once(monkeypatch):
    """Each squash resets --hard, so roots must never be pushed one by one."""
    calls = []
    monkeypatch.setattr(repo, "push_paths", lambda paths, msg: calls.append(list(paths)))
    monkeypatch.setattr(
        repo, "push_path", lambda *a, **kw: pytest.fail("per-root push_path call")
    )

    lib._push_data_dirs(["/d/inst/0", "/d/inst/schedule", "/d/inst/0"], "Starting task")

    assert calls == [["/d/inst/0", "/d/inst/schedule"]]


def test_unlocked_start_lease_wiped_by_reset_yields_no_changes(
    tmp_path: Path, monkeypatch
):
    """Vultr multi-start race: reset --hard between write_meta and push → no lease."""
    bare, work = _init_bare_and_clone(tmp_path)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)

    sched = work / "data" / "vendor" / "inst" / "schedule"
    sched_meta = sched / "task" / "meta.json"
    lease = '{"start":"leased","end":null,"exit_code":null}\n'
    sched_meta.write_text(lease)
    assert repo.path_has_changes(str(work), str(sched))

    # Concurrent push_paths squash does reset --hard and drops uncommitted lease.
    _git(work, "reset", "--hard", "origin/main")
    assert '"start":"old"' in sched_meta.read_text()
    assert repo.push_paths([sched], "Starting server") is False

    tip = tmp_path / "verify-wiped"
    _git(tmp_path, "clone", str(bare), str(tip))
    assert '"start":"old"' in (
        tip / "data/vendor/inst/schedule/task/meta.json"
    ).read_text()


def test_mutate_and_push_publishes_start_lease(tmp_path: Path, monkeypatch):
    bare, work = _init_bare_and_clone(tmp_path)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)

    sched = work / "data" / "vendor" / "inst" / "schedule"
    sched_meta = sched / "task" / "meta.json"
    lease = '{"start":"leased","end":null,"exit_code":null}\n'

    published = repo.mutate_and_push(
        [sched],
        "Starting server",
        lambda: sched_meta.write_text(lease),
    )
    assert published is True

    tip = tmp_path / "verify-lease"
    _git(tmp_path, "clone", str(bare), str(tip))
    assert (tip / "data/vendor/inst/schedule/task/meta.json").read_text() == lease


def test_concurrent_mutate_and_push_calls_serialize(tmp_path: Path, monkeypatch):
    """Two mutate_and_push threads on sibling roots must both land on origin."""
    bare, work = _init_bare_and_clone(tmp_path)
    monkeypatch.setattr(repo, "REPO_PATH", str(work))
    monkeypatch.setattr(repo, "should_push", lambda: True)

    sched = work / "data" / "vendor" / "inst" / "schedule"
    sched_meta = sched / "task" / "meta.json"
    keep = work / "data" / "vendor" / "inst" / "0" / "task" / "keep.txt"
    zero = work / "data" / "vendor" / "inst" / "0"
    lease = '{"start":"leased","end":null,"exit_code":null}\n'
    results: dict[str, bool] = {}

    def starter() -> None:
        def write_lease() -> None:
            # Hold the git lock while "dirty" so a concurrent push cannot reset.
            time.sleep(0.2)
            sched_meta.write_text(lease)

        results["lease"] = repo.mutate_and_push(
            [sched], "Starting server", write_lease
        )

    def other() -> None:
        time.sleep(0.05)

        def write_keep() -> None:
            keep.write_text("concurrent-sibling\n")

        # Sibling must also write under the git lock; unlocked dirty files are
        # exactly what reset --hard drops (see unlocked test above).
        results["other"] = repo.mutate_and_push([zero], "Sibling push", write_keep)

    t1 = threading.Thread(target=starter)
    t2 = threading.Thread(target=other)
    t1.start()
    t2.start()
    t1.join(timeout=60)
    t2.join(timeout=60)
    assert results.get("lease") is True
    assert results.get("other") is True

    tip = tmp_path / "verify-race"
    _git(tmp_path, "clone", str(bare), str(tip))
    assert (tip / "data/vendor/inst/schedule/task/meta.json").read_text() == lease
    assert (tip / "data/vendor/inst/0/task/keep.txt").read_text() == "concurrent-sibling\n"

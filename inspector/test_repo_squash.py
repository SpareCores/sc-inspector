"""Squash-push must publish deletions (stale stdout/stderr) without pathspec failure."""
from __future__ import annotations

import subprocess
from pathlib import Path

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
    data = work / "data" / "vendor" / "inst" / "task"
    data.mkdir(parents=True)
    (data / "meta.json").write_text('{"start":"old","end":"old","exit_code":0}\n')
    (data / "stdout").write_text("stale-stdout-from-prior-run\n")
    (data / "keep.txt").write_text("untouched-by-this-inspector\n")
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
    task_dir = work / "data" / "vendor" / "inst" / "task"

    # Inspector result commit: updated meta, new stderr, deleted stale stdout.
    (task_dir / "meta.json").write_text('{"start":"new","end":"new","exit_code":0}\n')
    (task_dir / "stderr").write_text("bench-noise\n")
    (task_dir / "stdout").unlink()
    _git(work, "add", "-A", "data/vendor/inst")
    _git(work, "commit", "-m", "local results")

    saved_head = _git(work, "rev-parse", "HEAD")
    rel = "data/vendor/inst"
    changed = repo._changed_files_under(str(work), rel)
    assert "data/vendor/inst/task/stdout" in changed
    assert "data/vendor/inst/task/stderr" in changed
    assert "data/vendor/inst/task/meta.json" in changed

    # Concurrent tip advance on a path this squash must preserve.
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(_bare), str(other))
    _git(other, "config", "user.email", "test@example.com")
    _git(other, "config", "user.name", "test")
    keep = other / "data" / "vendor" / "inst" / "task" / "keep.txt"
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
    tip_task = tip / "data" / "vendor" / "inst" / "task"
    assert not (tip_task / "stdout").exists()
    assert (tip_task / "stderr").read_text() == "bench-noise\n"
    assert (tip_task / "keep.txt").read_text() == "concurrent-edit\n"


def test_squash_retry_uses_saved_head_after_hard_reset(tmp_path: Path):
    """After a failed attempt's reset --hard, retries must still restore from saved_head."""
    _bare, work = _init_bare_and_clone(tmp_path)
    task_dir = work / "data" / "vendor" / "inst" / "task"

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

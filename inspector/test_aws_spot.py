"""AWS Spot-first / on-demand self-heal helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest import mock

import lib

NOW = datetime(2026, 8, 24, 12, 0, 0)


def _task(name: str = "bench", timeout_h: float = 1) -> lib.Task:
    return lib.Task(name=name, command="true", timeout=timedelta(hours=timeout_h))


def test_aws_market_modes_spot_first():
    modes = lib.aws_instance_market_modes()
    assert modes == [True, False]
    assert lib.aws_instance_market_modes(force_ondemand=True) == [False]


def test_cleanup_destroy_reason_unfinished_within_timeout():
    task = _task()
    meta = lib.Meta(start=NOW - timedelta(minutes=30), spot=True)
    assert lib.cleanup_destroy_reason([(task, meta)], NOW) is None


def test_cleanup_destroy_reason_stale_unfinished():
    task = _task(timeout_h=1)
    meta = lib.Meta(start=NOW - timedelta(hours=1, minutes=20), spot=True)
    assert lib.cleanup_destroy_reason([(task, meta)], NOW) == "all tasks have finished"


def test_cleanup_destroy_reason_max_unfinished_timeout():
    # Same start + timeout: once past timeout+DESTROY_AFTER every task is treated
    # as finished, so the shared helper reports "all tasks have finished".
    t1 = _task("a", timeout_h=2)
    t2 = _task("b", timeout_h=2)
    start = NOW - timedelta(hours=2, minutes=20)
    metas = [
        (t1, lib.Meta(start=start, spot=True)),
        (t2, lib.Meta(start=start, spot=True)),
    ]
    assert lib.cleanup_destroy_reason(metas, NOW) == "all tasks have finished"


def test_force_ondemand_after_spot_timeout(tmp_path):
    task = _task()
    meta = lib.Meta(start=NOW - timedelta(hours=2), spot=True, task_hash="x")
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        assert lib.force_ondemand_after_spot_timeout("aws", "t3.micro", tmp_path, now=NOW)
        assert not lib.force_ondemand_after_spot_timeout("gcp", "e2-micro", tmp_path, now=NOW)


def test_force_ondemand_requires_spot_flag(tmp_path):
    task = _task()
    meta = lib.Meta(start=NOW - timedelta(hours=2), spot=False)
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        assert not lib.force_ondemand_after_spot_timeout("aws", "t3.micro", tmp_path, now=NOW)


def test_should_start_spot_self_heal(tmp_path):
    task = _task()
    # Older than WAIT_SINCE_LAST_START so the early skip does not fire.
    meta = lib.Meta(
        start=NOW - timedelta(hours=2, minutes=1),
        spot=True,
        task_hash=lib.task_hash(task),
    )
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    srv = SimpleNamespace(
        vendor_id="aws",
        api_reference="t3.micro",
        gpu_count=0,
        memory_amount=1024,
        vcpus=2,
    )
    real_datetime = lib.datetime

    class _DT:
        @staticmethod
        def now():
            return NOW

        def __call__(self, *a, **k):
            return real_datetime(*a, **k)

    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "datetime", _DT()),
        mock.patch.object(lib, "force_ondemand_after_spot_timeout", return_value=True),
    ):
        assert lib.should_start(task, tmp_path, srv)


def test_should_start_skips_recent_unfinished_spot(tmp_path):
    task = _task()
    meta = lib.Meta(
        start=NOW - timedelta(minutes=10),
        spot=True,
        task_hash=lib.task_hash(task),
    )
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    srv = SimpleNamespace(
        vendor_id="aws",
        api_reference="t3.micro",
        gpu_count=0,
        memory_amount=1024,
        vcpus=2,
    )
    with mock.patch("lib.datetime") as dt:
        dt.now.return_value = NOW
        assert not lib.should_start(task, tmp_path, srv)


def test_should_start_spot_self_heal_for_start_with_instance_task(tmp_path):
    """vllm/llm-style tasks (start_with_instance=True) never reach the scheduling
    checks below the early bail-out on their own; the Spot self-heal must still be
    able to trigger a boot for them, or an unfinished Spot run of such a task could
    never be retried on-demand unless some unrelated task also happened to be due."""
    task = lib.Task(
        name="vllm", command="true", timeout=timedelta(hours=1), start_with_instance=True
    )
    # Older than WAIT_SINCE_LAST_START so the early skip does not fire.
    meta = lib.Meta(
        start=NOW - timedelta(hours=2, minutes=1),
        spot=True,
        task_hash=lib.task_hash(task),
    )
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    srv = SimpleNamespace(
        vendor_id="aws",
        api_reference="t3.micro",
        gpu_count=0,
        memory_amount=1024,
        vcpus=2,
    )
    real_datetime = lib.datetime

    class _DT:
        @staticmethod
        def now():
            return NOW

        def __call__(self, *a, **k):
            return real_datetime(*a, **k)

    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "datetime", _DT()),
        mock.patch.object(lib, "force_ondemand_after_spot_timeout", return_value=True),
    ):
        assert lib.should_start(task, tmp_path, srv)
        # and it must actually make it into the boot task list, not just should_start
        assert lib.tasks_to_start("aws", tmp_path, srv) == [task]


def test_should_start_start_with_instance_task_without_spot_self_heal(tmp_path):
    """Without an unfinished-Spot-past-timeout condition, start_with_instance tasks
    still never trigger a boot on their own (unchanged prior behavior)."""
    task = lib.Task(
        name="vllm", command="true", timeout=timedelta(hours=1), start_with_instance=True
    )
    meta = lib.Meta(start=NOW - timedelta(hours=2, minutes=1), spot=False, task_hash=lib.task_hash(task))
    lib.write_meta(meta, tmp_path / task.name / lib.META_NAME)
    srv = SimpleNamespace(
        vendor_id="aws", api_reference="t3.micro", gpu_count=0, memory_amount=1024, vcpus=2
    )
    with mock.patch("lib.datetime") as dt:
        dt.now.return_value = NOW
        assert not lib.should_start(task, tmp_path, srv)


def test_update_task_metas_spot(tmp_path):
    task = _task()
    lib.write_meta(
        lib.Meta(start=NOW, spot=True),
        tmp_path / task.name / lib.META_NAME,
    )
    lib.update_task_metas_spot(tmp_path, [task], spot=False)
    assert lib.load_task_meta(task, tmp_path).spot is False

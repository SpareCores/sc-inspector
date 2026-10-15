"""Hardware variant layout: selection, schedule union, and path helpers."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import lib
from variant_id import (
    build_variant_json,
    match_variant,
    normalize_x86_sku,
    parse_lscpu_stdout,
    variant_id_from_fields,
)

NOW = datetime(2026, 8, 24, 12, 0, 0)

X86_FIELDS = {
    "Architecture:": "x86_64",
    "Vendor ID:": "GenuineIntel",
    "CPU family:": "6",
    "Model:": "85",
    "Stepping:": "4",
    "Model name:": "Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz",
    "Flags:": "avx avx2",
    "L3 cache:": "34603008 (1 instance)",
    "CPU(s):": "2",
    "Socket(s):": "1",
    "NUMA node(s):": "1",
    "Address sizes:": "46 bits physical, 48 bits virtual",
    "L1d cache:": "32768 (1 instance)",
    "L1i cache:": "32768 (1 instance)",
    "L2 cache:": "1048576 (1 instance)",
}

X86_OTHER = {
    **X86_FIELDS,
    "Stepping:": "7",
    "Model name:": "Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz",
}


def _task(name: str = "bench") -> lib.Task:
    return lib.Task(name=name, command="true", timeout=timedelta(hours=1))


def test_normalize_x86_sku():
    assert normalize_x86_sku("Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz") == "8175m"
    assert normalize_x86_sku("Intel(R) Xeon(R) CPU @ 2.20GHz") == ""
    assert normalize_x86_sku("AMD EPYC-Rome Processor") == ""


def test_variant_id_x86_and_arm():
    assert (
        variant_id_from_fields(X86_FIELDS)
        == "x86_64|genuineintel|6|85|4|8175m"
    )
    arm = {
        "Architecture:": "aarch64",
        "Vendor ID:": "ARM",
        "Model name:": "Neoverse-N1",
        "Model:": "1",
        "Stepping:": "r3p1",
    }
    assert variant_id_from_fields(arm) == "aarch64|arm|neoverse-n1|1|r3p1"


def test_match_variant_recomputes_from_stored_lscpu():
    stored = build_variant_json(X86_FIELDS, source="backfill")
    assert match_variant(X86_FIELDS, stored)
    assert not match_variant(X86_OTHER, stored)


def test_select_or_create_claims_zero(tmp_path: Path):
    idx, root, vid = lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    assert idx == 0
    assert root == str(tmp_path / "0")
    assert (tmp_path / "0" / "variant.json").is_file()
    assert vid.endswith("8175m")


def test_select_or_create_reuses_match(tmp_path: Path):
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    idx, root, _ = lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    assert idx == 0
    assert root == str(tmp_path / "0")
    assert list(lib.variant_roots(tmp_path)) == [str(tmp_path / "0")]


def test_select_or_create_new_index(tmp_path: Path):
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    idx, root, vid = lib.select_or_create_variant(tmp_path, fields=X86_OTHER)
    assert idx == 1
    assert root == str(tmp_path / "1")
    assert "8259cl" in vid


def test_refresh_schedule_prefers_current_hash_success(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.select_or_create_variant(tmp_path, fields=X86_OTHER)
    # variant 0: success current hash
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(hours=2), end=NOW - timedelta(hours=1), exit_code=0, task_hash=thash),
        tmp_path / "0" / task.name / lib.META_NAME,
    )
    # variant 1: failure current hash (newer)
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(minutes=30), end=NOW - timedelta(minutes=10), exit_code=1, task_hash=thash),
        tmp_path / "1" / task.name / lib.META_NAME,
    )
    selected = lib.refresh_schedule_meta(tmp_path, task)
    assert selected is not None
    assert selected.exit_code == 0
    sched = json.loads((tmp_path / "schedule" / task.name / lib.META_NAME).read_text())
    assert sched["exit_code"] == 0


def _meta_path(root: Path, task: lib.Task) -> Path:
    return root / task.name / lib.META_NAME


def _sched(instance_root: Path, task: lib.Task) -> lib.Meta:
    return lib.load_schedule_meta(task, instance_root)


def _two_variants(tmp_path: Path) -> None:
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.select_or_create_variant(tmp_path, fields=X86_OTHER)


def test_refresh_mirrors_active_variant_in_progress(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    running = lib.Meta(start=NOW - timedelta(minutes=5), task_hash=thash)
    lib.write_meta(running, _meta_path(tmp_path / "0", task))
    selected = lib.refresh_schedule_meta(tmp_path, task, active_root=tmp_path / "0")
    assert selected is not None and selected.end is None
    assert _sched(tmp_path, task).start == running.start


def test_abandoned_run_on_other_variant_does_not_pin_schedule(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    _two_variants(tmp_path)
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(hours=3), end=NOW - timedelta(hours=2), exit_code=0, task_hash=thash),
        _meta_path(tmp_path / "0", task),
    )
    # VM died mid-task on variant 1: start, no end, forever.
    lib.write_meta(lib.Meta(start=NOW - timedelta(hours=1), task_hash=thash), _meta_path(tmp_path / "1", task))
    selected = lib.refresh_schedule_meta(tmp_path, task, active_root=tmp_path / "0")
    assert selected is not None and selected.exit_code == 0
    assert lib.refresh_schedule_meta(tmp_path, task).exit_code == 0


def test_stale_retry_reset_survives_refresh(tmp_path: Path):
    """reconcile_stale_start_retries writes schedule only; a later refresh must not undo it."""
    task = _task()
    thash = lib.task_hash(task)
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.write_meta(lib.Meta(start=NOW - timedelta(days=4), task_hash=thash), _meta_path(tmp_path / "0", task))
    lib.write_meta(lib.Meta(start_retries=1, task_hash=thash), _meta_path(tmp_path / "schedule", task))
    assert lib.refresh_schedule_meta(tmp_path, task) is None
    sched = _sched(tmp_path, task)
    assert sched.start is None and sched.start_retries == 1


def test_refresh_keeps_schedule_only_state(tmp_path: Path):
    """Start failures, stale retries and -5 have no variant meta; refresh must not delete them."""
    task = _task()
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    failed = lib.Meta(
        start=NOW, end=NOW, exit_code=-1, error_msg="QuotaExceeded", start_retries=2,
        task_hash=lib.task_hash(task),
    )
    lib.write_meta(failed, _meta_path(tmp_path / "schedule", task))
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        lib.refresh_all_schedule_metas(tmp_path, "aws", active_root=tmp_path / "0")
    sched = _sched(tmp_path, task)
    assert sched.exit_code == -1 and sched.start_retries == 2


def test_variant_write_carries_spot_and_retries_from_boot_placeholder(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(minutes=10), spot=True, start_retries=2, task_hash=thash),
        _meta_path(tmp_path / "schedule", task),
    )
    # Real-start write before a blocking task: no spot on the variant side.
    real_start = lib.Meta(start=NOW, task_hash=thash)
    lib.write_variant_task_meta(real_start, task, tmp_path / "0", tmp_path)
    variant_text = _meta_path(tmp_path / "0", task).read_text()
    assert variant_text == _meta_path(tmp_path / "schedule", task).read_text()
    sched = _sched(tmp_path, task)
    assert sched.spot is True and sched.start_retries == 2 and sched.end is None
    assert lib.is_unfinished_spot_meta(sched)


def test_vm_task_meta_uses_schedule_placeholder(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(days=30), end=NOW - timedelta(days=30), exit_code=0, task_hash=thash),
        _meta_path(tmp_path / "0", task),
    )
    lib.write_meta(lib.Meta(start=NOW, spot=True, task_hash=thash), _meta_path(tmp_path / "schedule", task))
    meta = lib.vm_task_meta(task, tmp_path / "0", tmp_path)
    assert meta.start == NOW and meta.spot is True
    # DBaaS / no instance root: plain variant-less load.
    assert lib.vm_task_meta(task, tmp_path / "0", None).exit_code == 0


def test_finalize_clears_costart_placeholder_with_union_success(tmp_path: Path):
    task = _task()
    thash = lib.task_hash(task)
    _two_variants(tmp_path)
    lib.write_meta(
        lib.Meta(start=NOW - timedelta(days=1), end=NOW - timedelta(days=1), exit_code=0, task_hash=thash),
        _meta_path(tmp_path / "1", task),
    )
    lib.write_meta(lib.Meta(start=NOW, task_hash=thash), _meta_path(tmp_path / "schedule", task))
    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "should_run", return_value=False),
        mock.patch.object(lib, "_push_data_dirs"),
    ):
        lib.finalize_task_metas("aws", tmp_path / "0", "t3.micro", instance_root=tmp_path)
    assert lib.load_task_meta(task, tmp_path / "0").exit_code == -2
    sched = _sched(tmp_path, task)
    assert sched.exit_code == 0 and sched.end is not None


def test_finalize_marks_placeholder_when_no_variant_has_task(tmp_path: Path):
    task = _task()
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    lib.write_meta(lib.Meta(start=NOW, task_hash=lib.task_hash(task)), _meta_path(tmp_path / "schedule", task))
    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "should_run", return_value=True),
        mock.patch.object(lib, "_push_data_dirs"),
    ):
        lib.finalize_task_metas("aws", tmp_path / "0", "t3.micro", instance_root=tmp_path)
    assert lib.load_task_meta(task, tmp_path / "0").exit_code == -4
    assert _sched(tmp_path, task).exit_code == -4


def test_start_failure_writes_schedule_only(tmp_path: Path):
    import threading

    task = _task()
    with mock.patch.object(lib, "repo"):
        lib.record_instance_start_failure(
            threading.Lock(), lib.schedule_root(tmp_path), [task], [], fallback_msg="boom"
        )
    assert _sched(tmp_path, task).exit_code == -1
    assert lib.variant_roots(tmp_path) == []


def test_stale_retry_writes_schedule_only(tmp_path: Path):
    task = _task()
    stale = lib.Meta(start=datetime.now() - timedelta(days=4), task_hash=lib.task_hash(task))
    lib.write_meta(stale, _meta_path(tmp_path / "schedule", task))
    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "_task_matches_server_for_start", return_value=True),
        mock.patch.object(lib, "repo"),
    ):
        assert lib.reconcile_stale_start_retries("aws", lib.schedule_root(tmp_path), srv=None)
    sched = _sched(tmp_path, task)
    assert sched.start is None and sched.start_retries == 1
    assert lib.variant_roots(tmp_path) == []


def test_cleanup_keeps_pushed_active_nonparallel_task(tmp_path: Path):
    early = _task("early")
    blocking = _task("blocking")
    lib.select_or_create_variant(tmp_path, fields=X86_FIELDS)
    for task in (early, blocking):
        lib.write_meta(lib.Meta(start=NOW - timedelta(minutes=30), task_hash=lib.task_hash(task)),
                       _meta_path(tmp_path / "schedule", task))
    lib.write_variant_task_meta(
        lib.Meta(start=NOW - timedelta(minutes=30), end=NOW - timedelta(minutes=20), exit_code=0,
                 kernel_version="6.8", task_hash=lib.task_hash(early)),
        early, tmp_path / "0", tmp_path,
    )
    lib.write_variant_task_meta(
        lib.Meta(start=NOW - timedelta(minutes=19), task_hash=lib.task_hash(blocking)),
        blocking, tmp_path / "0", tmp_path,
    )
    metas = lib.instance_task_metas("aws", "t3.micro", lib.schedule_root(tmp_path), [early, blocking])
    assert lib.cleanup_destroy_reason(metas, NOW) is None


def test_cleanup_treats_untouched_placeholder_as_abandoned(tmp_path: Path):
    done = _task("done")
    placeholder = _task("never_ran")
    sched_root = Path(lib.schedule_root(tmp_path))
    lib.write_meta(lib.Meta(start=NOW - timedelta(minutes=40), end=NOW - timedelta(minutes=30), exit_code=0,
                            task_hash="x"), _meta_path(sched_root, done))
    lib.write_meta(lib.Meta(start=NOW - timedelta(minutes=40)), _meta_path(sched_root, placeholder))
    metas = lib.instance_task_metas("aws", "t3.micro", sched_root, [done, placeholder])
    assert lib.cleanup_destroy_reason(metas, NOW) == "all tasks have finished"


def test_s3_cleanup_guard_reads_schedule(tmp_path: Path):
    from types import SimpleNamespace

    task = _task()
    repo_data = tmp_path / "data"
    sched_root = repo_data / "aws" / "t3.micro" / "schedule"
    lib.write_meta(lib.Meta(start=NOW), _meta_path(sched_root, task))
    records = [SimpleNamespace(terminated_at=NOW - timedelta(hours=1))]
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        assert lib.active_run_blocks_s3_cleanup("aws", "t3.micro", records, repo_data)


def test_data_only_scan_accepts_schedule_only_instance(tmp_path: Path):
    task = _task()
    repo_data = tmp_path / "data"
    lib.write_meta(
        lib.Meta(start=datetime.now() - timedelta(minutes=10)),
        _meta_path(repo_data / "aws" / "t3.micro" / "schedule", task),
    )
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        assert lib.should_scan_for_cleanup(repo_data, "aws", "t3.micro", data_only=True)


def test_parse_walks_every_variant_and_skips_schedule(tmp_path: Path):
    calls: list[str] = []

    def record(meta, task, task_dir):
        calls.append(str(Path(task_dir).relative_to(tmp_path)))

    task = lib.Task(name="bench", command="true", timeout=timedelta(hours=1), parse_output=[record])
    _two_variants(tmp_path)
    for root in ("0", "1", "schedule"):
        lib.write_meta(lib.Meta(outputs=["stdout"]), _meta_path(tmp_path / root, task))
    walked = lib.parse_variant_outputs(tmp_path, [task])
    assert [Path(r).name for r in walked] == ["0", "1"]
    assert calls == ["0/bench", "1/bench"]


def test_s3_upload_skips_layout_dirs(tmp_path: Path):
    import s3_runs

    for name in ("bench", "schedule", "0"):
        (tmp_path / name).mkdir()
    uploaded: list[str] = []
    with (
        mock.patch.object(s3_runs, "_parse_task_logs_post", return_value={"prefix": "p/"}),
        mock.patch.object(s3_runs, "upload_task_artifact",
                          side_effect=lambda name, *a, **k: uploaded.append(name) or False),
    ):
        s3_runs.upload_task_logs_to_s3(str(tmp_path))
    assert set(uploaded) == {"bench"}


def test_dbaas_paths_unchanged(tmp_path: Path):
    """DBaaS passes no instance_root: flat task meta, no schedule/ or variant dirs."""
    task = _task()
    meta = lib.Meta(start=NOW, end=NOW, exit_code=0, task_hash=lib.task_hash(task))
    lib.write_variant_task_meta(meta, task, tmp_path, None)
    assert _meta_path(tmp_path, task).is_file()
    assert not (tmp_path / "schedule").exists()
    with (
        mock.patch.object(lib, "get_tasks", return_value=[task]),
        mock.patch.object(lib, "_push_data_dirs"),
    ):
        lib.finalize_task_metas("aws", tmp_path, "db", instance_root=None)
    assert not (tmp_path / "schedule").exists()
    assert lib.variant_roots(tmp_path) == []


def test_get_last_start_reads_schedule(tmp_path: Path):
    task = _task()
    repo_data = tmp_path / "data"
    lib.write_meta(lib.Meta(start=NOW), _meta_path(repo_data / "aws" / "t3.micro" / "schedule", task))
    with mock.patch.object(lib, "get_tasks", return_value=[task]):
        assert lib.get_last_start(repo_data, "aws", "t3.micro") == NOW


def test_validate_task_name_rejects_reserved():
    for name in ("schedule", "0", "12"):
        try:
            lib.validate_task_name(name)
        except ValueError:
            continue
        raise AssertionError(f"{name!r} accepted")
    lib.validate_task_name("lscpu")


def test_parse_lscpu_stdout_list():
    raw = {"lscpu": [{"field": "Architecture:", "data": "x86_64"}]}
    assert parse_lscpu_stdout(raw)["Architecture:"] == "x86_64"

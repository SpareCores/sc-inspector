"""Region is stored in meta.json from the GHA-injected inspect env, not IMDS."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import lib

NOW = datetime(2026, 11, 6, 12, 0, 0)


def _task(name: str = "geekbench") -> lib.Task:
    return lib.Task(name=name, command="true")


def _meta_path(root: Path, task: lib.Task) -> Path:
    return root / task.name / lib.META_NAME


def test_provisioned_region_prefers_region_env(monkeypatch):
    monkeypatch.setenv("REGION", "eu-west-1")
    monkeypatch.setenv("SC_PROVISION_REGION", "us-east-1")
    assert lib.provisioned_region() == "eu-west-1"


def test_provisioned_region_falls_back_to_sc_provision_region(monkeypatch):
    monkeypatch.delenv("REGION", raising=False)
    monkeypatch.setenv("SC_PROVISION_REGION", "eastus")
    assert lib.provisioned_region() == "eastus"


def test_boot_meta_for_task_reads_region_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REGION", "ap-southeast-1")
    task = _task()
    meta = lib.boot_meta_for_task(task, tmp_path)
    assert meta.region == "ap-southeast-1"


def test_boot_meta_for_task_keeps_explicit_region(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REGION", "ap-southeast-1")
    task = _task()
    meta = lib.boot_meta_for_task(task, tmp_path, region="us-west-2")
    assert meta.region == "us-west-2"


def test_write_variant_task_meta_fills_region_from_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REGION", "westeurope")
    task = _task()
    thash = lib.task_hash(task)
    (tmp_path / "0").mkdir()
    meta = lib.Meta(start=NOW, task_hash=thash)
    with mock.patch.object(lib, "refresh_schedule_meta"):
        lib.write_variant_task_meta(meta, task, tmp_path / "0", tmp_path)
    loaded = lib.load_task_meta(task, tmp_path / "0")
    assert loaded.region == "westeurope"


def test_write_variant_task_meta_carries_region_from_placeholder(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("REGION", raising=False)
    monkeypatch.delenv("SC_PROVISION_REGION", raising=False)
    task = _task()
    thash = lib.task_hash(task)
    (tmp_path / "0").mkdir()
    (tmp_path / "schedule" / task.name).mkdir(parents=True)
    lib.write_meta(
        lib.Meta(
            start=NOW - timedelta(minutes=5),
            region="us-east-2",
            spot=True,
            task_hash=thash,
        ),
        _meta_path(tmp_path / "schedule", task),
    )
    meta = lib.Meta(start=NOW, task_hash=thash)
    with mock.patch.object(lib, "refresh_schedule_meta"):
        lib.write_variant_task_meta(meta, task, tmp_path / "0", tmp_path)
    loaded = lib.load_task_meta(task, tmp_path / "0")
    assert loaded.region == "us-east-2"
    assert loaded.spot is True


def test_user_data_passes_region_into_inspect_containers():
    script = Path(__file__).with_name("user_data.sh").read_text()
    assert "-e REGION={REGION}" in script
    # Both the plain inspect and DBaaS client inspect paths get REGION.
    assert script.count("-e REGION={REGION}") >= 2

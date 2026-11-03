"""Meta-driven DBaaS absolute-timeout cleanup must not depend on the catalog or S3 records.

Regression for an Azure DBaaS stack (sc-e4adsv6-pg18) left running ~10h: the SKU's
``available_managed_dbs()`` happened to have 0 ACTIVE rows at the time (catalog-gated
cleanup never looked at it) and its client's boot/user_data hung before any benchmark
container - and thus any S3 run record - ever ran (S3-record-driven cleanup never saw
it either). ``_dbaas_instance_keys_on_disk`` + ``lib.dbaas_instance_task_metas`` +
``lib.cleanup_destroy_reason`` must still flag such a stack from the on-disk meta alone.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import inspector as insp
import lib


def _write_meta(path: Path, *, start: str, end: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"start": start, "end": end, "exit_code": 0 if end else None}
    import json

    path.write_text(json.dumps(body))


def test_discovers_instance_key_from_meta_on_disk_only(tmp_path: Path):
    repo_dbaas = tmp_path / "dbaas"
    instance_key = "Standard_E4ads_v6/postgres/18/standalone"
    task_name = "pgbench_postgres_dbaas_ro_durable"
    _write_meta(
        repo_dbaas / "azure" / instance_key / task_name / "meta.json",
        start="2026-11-02T23:36:17",
    )

    keys = insp._dbaas_instance_keys_on_disk(str(repo_dbaas), "azure")

    assert keys == {instance_key}


def test_discovers_nothing_for_absent_vendor_dir(tmp_path: Path):
    repo_dbaas = tmp_path / "dbaas"
    assert insp._dbaas_instance_keys_on_disk(str(repo_dbaas), "azure") == set()


def test_stuck_unfinished_task_triggers_destroy_reason_without_catalog(tmp_path: Path):
    repo_dbaas = tmp_path / "dbaas"
    instance_key = "Standard_E4ads_v6/postgres/18/standalone"
    task_name = "pgbench_postgres_dbaas_ro_durable"
    started = datetime.now() - timedelta(hours=10)
    _write_meta(
        repo_dbaas / "azure" / instance_key / task_name / "meta.json",
        start=started.isoformat(),
    )

    data_dir = str(repo_dbaas / "azure" / instance_key)
    task_metas = lib.dbaas_instance_task_metas("azure", instance_key, data_dir)

    assert task_metas, "expected the pgbench DbaasDbTask to be applicable"
    reason = lib.cleanup_destroy_reason(task_metas, datetime.now())
    assert reason, "a 10h-old unfinished DBaaS meta must be destroy-worthy"


def test_freshly_started_task_does_not_trigger_destroy_reason(tmp_path: Path):
    repo_dbaas = tmp_path / "dbaas"
    instance_key = "Standard_E4ads_v6/postgres/18/standalone"
    task_name = "pgbench_postgres_dbaas_ro_durable"
    _write_meta(
        repo_dbaas / "azure" / instance_key / task_name / "meta.json",
        start=datetime.now().isoformat(),
    )

    data_dir = str(repo_dbaas / "azure" / instance_key)
    task_metas = lib.dbaas_instance_task_metas("azure", instance_key, data_dir)

    reason = lib.cleanup_destroy_reason(task_metas, datetime.now())
    assert reason is None

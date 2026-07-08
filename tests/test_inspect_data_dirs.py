import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from lib import (  # noqa: E402
    DbaasDbTask,
    DockerTask,
    META_NAME,
    MultiVmDbTask,
    Task,
    load_task_meta,
    should_run,
    task_data_dir,
    task_hash,
    write_meta,
)


def test_task_data_dir_routes_dbaas_and_client_paths(tmp_path):
    dbaas_dir = tmp_path / "dbaas" / "azure" / "Standard_E16ds_v5/postgres/18/standalone"
    client_dir = tmp_path / "data" / "azure" / "Standard_D8als_v6"
    dbaas_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    dbaas_task = DbaasDbTask(image="bench", command=["run"], cache_tier="c100")
    dbaas_task.name = "hammerdb_postgres_dbaas_oltp_mixed_durable_c100"
    client_task = DockerTask(image="bench", command="echo", timeout=__import__("datetime").timedelta(minutes=1))
    client_task.name = "stressngfull"

    assert task_data_dir(dbaas_task, dbaas_dir, client_dir) == dbaas_dir
    assert task_data_dir(client_task, dbaas_dir, client_dir) == client_dir
    assert task_data_dir(client_task, client_dir, None) == client_dir


def test_should_run_skips_multi_vm_on_dbaas_topology(tmp_path):
    task = MultiVmDbTask(
        image="bench",
        command=["run"],
        benchmark_family="hammerdb",
        tool="hammerdb",
        workload="oltp",
        profile="mixed",
        durability="durable",
        cache_tier="c100",
        timeout=__import__("datetime").timedelta(minutes=1),
    )
    task.name = "hammerdb_postgres_multi_oltp_mixed_durable_c100"

    with patch.dict(os.environ, {"TOPOLOGY": "dbaas"}, clear=False):
        assert should_run(task, tmp_path, "azure", "Standard_D8als_v6", 0.0) is False


def test_should_run_reuses_client_meta_for_compute_tasks(tmp_path):
    dbaas_dir = tmp_path / "dbaas" / "azure" / "Standard_E16ds_v5/postgres/18/standalone"
    client_dir = tmp_path / "data" / "azure" / "Standard_D8als_v6"
    dbaas_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    task = DockerTask(image="bench", command="echo", timeout=__import__("datetime").timedelta(minutes=1))
    task.name = "stressngfull"
    task_dir = client_dir / task.name
    task_dir.mkdir(parents=True)
    meta = load_task_meta(task, client_dir)
    meta.start = datetime(2024, 11, 22, 7, 35, 48)
    meta.end = datetime(2024, 11, 22, 7, 37, 11)
    meta.exit_code = 0
    meta.task_hash = task_hash(task)
    write_meta(meta, task_dir / META_NAME)

    with patch.dict(os.environ, {"TOPOLOGY": "dbaas"}, clear=False):
        routed = task_data_dir(task, dbaas_dir, client_dir)
        assert routed == client_dir
        assert should_run(task, routed, "azure", "Standard_D8als_v6", 0.0) is False

    assert not (dbaas_dir / task.name).exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_task_data_dir_routes_dbaas_and_client_paths(root)
        test_should_run_skips_multi_vm_on_dbaas_topology(root)
        test_should_run_reuses_client_meta_for_compute_tasks(root)
    print("ok")

import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

# Heavy deps are unavailable in lightweight test runs; stub before inspector imports.
_sc = ModuleType("sc_crawler")
_sc_tables = ModuleType("sc_crawler.tables")
_sc_tf = ModuleType("sc_crawler.table_fields")
_sc_tf.CpuArchitecture = SimpleNamespace(X86_64="x86_64")
sys.modules["sc_crawler"] = _sc
sys.modules["sc_crawler.tables"] = _sc_tables
sys.modules["sc_crawler.table_fields"] = _sc_tf

_sc_runner = ModuleType("sc_runner")
_sc_runner_runner = ModuleType("sc_runner.runner")
_sc_runner_runner.create = MagicMock()
_sc_runner_runner.destroy = MagicMock()
_sc_runner_runner.destroy_stack = MagicMock()
_sc_runner_resources = ModuleType("sc_runner.resources")
_sc_runner_managed_db = ModuleType("sc_runner.resources.managed_db")


class _ManagedDbSpec:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DbaasStackSpec:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_sc_runner_managed_db.ManagedDbSpec = _ManagedDbSpec
_sc_runner_managed_db.DbaasStackSpec = _DbaasStackSpec
sys.modules["sc_runner"] = _sc_runner
sys.modules["sc_runner.runner"] = _sc_runner_runner
sys.modules["sc_runner.resources"] = _sc_runner_resources
sys.modules["sc_runner.resources.managed_db"] = _sc_runner_managed_db

_repo = ModuleType("repo")
_repo.pull = MagicMock()
_repo.push_path = MagicMock()
_repo.gha_url = MagicMock(return_value="https://example.test")
sys.modules["repo"] = _repo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from dbaas_start import _try_provision_dbaas_stack, try_start_dbaas_inspect  # noqa: E402
from lib import DbaasDbTask  # noqa: E402


def _client(api_reference: str, vcpus: int = 4):
    return SimpleNamespace(
        api_reference=api_reference,
        vcpus=vcpus,
        cpu_architecture="x86_64",
    )


def _target():
    return SimpleNamespace(
        instance_key="azure-pg-e16",
        native_id="Standard_E16ds_v5",
        engine="postgresql",
        engine_version="16",
        ha_mode="Disabled",
        cpu_count=16,
        memory_gib=128,
        vendor_id="azure",
        sku_id="Standard_E16ds_v5",
    )


def _task(cache_tier: str = "c100"):
    return DbaasDbTask(
        image="bench",
        command=["run"],
        cache_tier=cache_tier,
        benchmark_family="hammerdb",
        tool="hammerdb",
        timeout=timedelta(minutes=30),
    )


@patch("dbaas_start._try_provision_dbaas_stack")
@patch("dbaas_start.filter_clients_by_vm_quota")
@patch("dbaas_start.rank_client_instances")
@patch("dbaas_start.check_dbaas_postgres_quota", return_value=(True, ""))
@patch("dbaas_start.candidate_regions", return_value=["northeurope"])
@patch(
    "dbaas_start.provision_spec",
    return_value={"sku_name": "Standard_E16ds_v5", "sku_tier": "GeneralPurpose"},
)
@patch("dbaas_start.stack_slug", return_value="slug")
@patch("dbaas_start.target_sizing_stub")
def test_try_start_tries_next_client_after_vm_quota_skip(
    mock_target_stub,
    _mock_slug,
    _mock_provision_spec,
    _mock_regions,
    _mock_pg_quota,
    mock_rank,
    mock_filter,
    mock_provision,
):
    mock_target_stub.return_value = SimpleNamespace(
        memory_amount=128 * 1024,
        cpu_count=16,
        vcpus=16,
    )
    clients = [_client("Standard_B8als_v2"), _client("Standard_D4s_v3")]
    mock_rank.return_value = clients
    mock_filter.return_value = [_client("Standard_D4s_v3")]
    mock_provision.return_value = True

    started = try_start_dbaas_inspect(
        executor=MagicMock(),
        lock=MagicMock(),
        data_dir="/tmp",
        vendor="azure",
        target=_target(),
        tasks=[_task()],
        regions=["northeurope"],
        zones=[],
        zone_to_region={},
        timeout_mins=60,
        ssh_deploy_key_b64="",
        repo_url_ssh="git@github.com:example/repo.git",
        instance_logger=MagicMock(),
        instance_timing=MagicMock(),
        error_msgs=[],
    )

    assert started is True
    assert mock_provision.call_count == 1
    assert mock_provision.call_args[0][2].api_reference == "Standard_D4s_v3"


@patch("dbaas_start._try_provision_dbaas_stack")
@patch("dbaas_start.filter_clients_by_vm_quota")
@patch("dbaas_start.rank_client_instances")
@patch("dbaas_start.check_dbaas_postgres_quota", return_value=(True, ""))
@patch("dbaas_start.candidate_regions", return_value=["northeurope"])
@patch("dbaas_start.provision_spec", return_value={"sku_name": "Standard_E16ds_v5"})
@patch("dbaas_start.stack_slug", return_value="slug")
@patch("dbaas_start.target_sizing_stub")
def test_try_start_tries_next_client_after_create_failure(
    mock_target_stub,
    _mock_slug,
    _mock_provision_spec,
    _mock_regions,
    _mock_pg_quota,
    mock_rank,
    mock_filter,
    mock_provision,
):
    mock_target_stub.return_value = SimpleNamespace(
        memory_amount=128 * 1024,
        cpu_count=16,
        vcpus=16,
    )
    clients = [_client("Standard_B8als_v2"), _client("Standard_D4s_v3")]
    mock_rank.return_value = clients
    mock_filter.side_effect = lambda _vendor, _region, ranked: ranked
    mock_provision.side_effect = [False, True]

    started = try_start_dbaas_inspect(
        executor=MagicMock(),
        lock=MagicMock(),
        data_dir="/tmp",
        vendor="azure",
        target=_target(),
        tasks=[_task()],
        regions=["northeurope"],
        zones=[],
        zone_to_region={},
        timeout_mins=60,
        ssh_deploy_key_b64="",
        repo_url_ssh="git@github.com:example/repo.git",
        instance_logger=MagicMock(),
        instance_timing=MagicMock(),
        error_msgs=[],
    )

    assert started is True
    assert mock_provision.call_count == 2


@patch("dbaas_start.runner.destroy_stack")
@patch("dbaas_start.retry_locked", side_effect=RuntimeError("create failed"))
@patch("dbaas_start._dbaas_user_data_replacements", return_value={"USER_DATA_TEMPLATE": "tmpl"})
def test_try_provision_destroy_stack_on_create_failure(mock_repl, mock_retry, mock_destroy_stack):
    del mock_repl
    client = _client("Standard_D8als_v6")
    target = _target()
    provision = {
        "sku_name": "Standard_E16ds_v5",
        "sku_tier": "MemoryOptimized",
        "storage_gib": 128,
        "storage_type": "PremiumV2_LRS",
        "storage_edition": "ManagedDiskV2",
        "iops_tier": "P30",
        "cache_tier": "c100",
        "admin_login": "scadmin",
        "database_name": "bench",
    }

    started = _try_provision_dbaas_stack(
        vendor="azure",
        target=target,
        client=client,
        region="northeurope",
        zone=None,
        cache_tier="c100",
        provision=provision,
        slug="e16dsv5-pg18-c100",
        timeout_mins=60,
        ssh_deploy_key_b64="",
        repo_url_ssh="git@github.com:example/repo.git",
        instance_logger=MagicMock(),
        instance_timing=MagicMock(),
        error_msgs=[],
    )

    assert started is False
    assert mock_destroy_stack.call_count == 2
    first_opts = mock_destroy_stack.call_args_list[0].args[2]
    second_opts = mock_destroy_stack.call_args_list[1].args[2]
    assert first_opts["instance"] == "Standard_D8als_v6"
    assert first_opts["dbaas_slug"] == "e16dsv5-pg18-c100"
    assert second_opts["instance"] == first_opts["instance"]


@patch("dbaas_start.runner.destroy_stack")
@patch("dbaas_start.retry_locked", return_value=None)
@patch("dbaas_start._dbaas_user_data_replacements", return_value={"USER_DATA_TEMPLATE": "tmpl"})
def test_try_provision_passes_image_sku_only_for_azure(mock_repl, mock_retry, mock_destroy_stack):
    del mock_repl, mock_destroy_stack
    client = _client("n4-highcpu-16")
    target = _target()
    provision = {
        "sku_name": "db-perf-optimized-N-16",
        "sku_tier": "PerformanceOptimized",
        "storage_gib": 128,
        "storage_type": "PD_SSD",
        "storage_edition": "PD_SSD",
        "iops_tier": "P30",
        "cache_tier": "c100",
        "admin_login": "scadmin",
        "database_name": "bench",
    }

    _try_provision_dbaas_stack(
        vendor="gcp",
        target=target,
        client=client,
        region="europe-west1",
        zone="europe-west1-b",
        cache_tier="c100",
        provision=provision,
        slug="perfoptn16-pg18-c100",
        timeout_mins=60,
        ssh_deploy_key_b64="",
        repo_url_ssh="git@github.com:example/repo.git",
        instance_logger=MagicMock(),
        instance_timing=MagicMock(),
        error_msgs=[],
    )

    create_opts = mock_retry.call_args.args[3]
    assert "image_sku" not in create_opts

    mock_retry.reset_mock()
    _try_provision_dbaas_stack(
        vendor="azure",
        target=target,
        client=client,
        region="northeurope",
        zone=None,
        cache_tier="c100",
        provision=provision,
        slug="perfoptn16-pg18-c100",
        timeout_mins=60,
        ssh_deploy_key_b64="",
        repo_url_ssh="git@github.com:example/repo.git",
        instance_logger=MagicMock(),
        instance_timing=MagicMock(),
        error_msgs=[],
    )
    create_opts = mock_retry.call_args.args[3]
    assert create_opts["image_sku"] == "server"


def test_finalize_multi_vm_band_skipped_for_dbaas_topology():
    import os
    from lib import _finalize_multi_vm_band_if_done

    with patch.dict(os.environ, {"TOPOLOGY": "dbaas"}, clear=False):
        _finalize_multi_vm_band_if_done([(1.0, False)], 0, "/tmp")


if __name__ == "__main__":
    tests = [
        test_try_start_tries_next_client_after_vm_quota_skip,
        test_try_start_tries_next_client_after_create_failure,
        test_try_provision_destroy_stack_on_create_failure,
        test_try_provision_passes_image_sku_only_for_azure,
        test_finalize_multi_vm_band_skipped_for_dbaas_topology,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL PASSED")

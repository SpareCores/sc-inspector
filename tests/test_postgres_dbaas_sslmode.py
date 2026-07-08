import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from postgres_dbaas import _benchmark_env, _db_connect_kwargs, _db_sslmode  # noqa: E402


def test_db_sslmode_explicit_override():
    with patch.dict(os.environ, {"SC_DB_SSLMODE": "verify-full"}, clear=False):
        assert _db_sslmode() == "verify-full"


def test_db_sslmode_azure_private_vnet_defaults_to_require():
    with patch.dict(
        os.environ,
        {"SC_PROVISION_NETWORK_MODE": "private_vnet"},
        clear=True,
    ):
        assert _db_sslmode() == "require"


def test_db_connect_kwargs_disable_omits_sslmode():
    with patch.dict(os.environ, {"SC_DB_SSLMODE": "disable"}, clear=False):
        assert _db_connect_kwargs() == {}


def test_benchmark_env_includes_sslmode_for_dbaas():
    from types import SimpleNamespace

    task = SimpleNamespace(
        name="benchbase_postgres_dbaas_read_heavy_c100",
        tool="benchbase",
        workload_proxy="read_heavy",
        cache_ratio=1.0,
        cache_tier="c100",
        durability="durable",
        image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    )
    params = SimpleNamespace(build_vus=4, run_vus=4, scale_units=100)
    with patch.dict(
        os.environ,
        {
            "SC_DB_HOST": "db.example",
            "SC_DB_PASSWORD": "secret",
            "SC_PROVISION_NETWORK_MODE": "private_vnet",
            "SC_PROVISION_VENDOR_ID": "azure",
        },
        clear=False,
    ):
        env = _benchmark_env(task, params, mem_gib=128.0, db_vcpus=16, client_vcpus=16)
    assert env["SC_DB_SSLMODE"] == "require"


if __name__ == "__main__":
    test_db_sslmode_explicit_override()
    test_db_sslmode_azure_private_vnet_defaults_to_require()
    test_db_connect_kwargs_disable_omits_sslmode()
    test_benchmark_env_includes_sslmode_for_dbaas()
    print("ok")

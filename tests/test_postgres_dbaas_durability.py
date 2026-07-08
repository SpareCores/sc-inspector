import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from postgres_dbaas import _apply_durability, _durability_roles  # noqa: E402


def _mock_psycopg2(existing_roles: set[str]):
    cur = MagicMock()

    def execute(sql, params=None):
        if sql.startswith("SELECT 1 FROM pg_roles"):
            cur.fetchone.return_value = (1,) if params[0] in existing_roles else None

    cur.execute.side_effect = execute
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


def test_durability_roles_hammerdb_tpcc():
    task = SimpleNamespace(tool="hammerdb", workload_proxy="oltp_mixed")
    assert _durability_roles(task) == ["scadmin", "tpcc"]


def test_durability_roles_hammerdb_tpch():
    task = SimpleNamespace(tool="hammerdb", workload_proxy="olap")
    assert _durability_roles(task) == ["scadmin", "tpch"]


def test_durability_roles_benchbase():
    task = SimpleNamespace(tool="benchbase", workload_proxy="read_heavy")
    assert _durability_roles(task) == ["scadmin"]


def test_apply_durability_async_sets_role_off():
    conn, cur = _mock_psycopg2({"scadmin", "tpcc"})
    psycopg2 = MagicMock()
    psycopg2.connect.return_value = conn
    import postgres_dbaas as mod

    with (
        patch.dict(sys.modules, {"psycopg2": psycopg2}),
        patch.object(mod, "PG_USER", "scadmin"),
        patch.object(mod, "PG_PASSWORD", "secret"),
        patch.object(mod, "PG_DB", "bench"),
        patch.object(mod, "_db_host", return_value="db.example"),
        patch.object(mod, "_db_port", return_value=5432),
        patch.object(mod, "_db_connect_kwargs", return_value={"sslmode": "require"}),
    ):
        _apply_durability("async", ["scadmin", "tpcc"], sync_settable=True)

    alter_sql = [call.args[0] for call in cur.execute.call_args_list if call.args[0].startswith("ALTER ROLE")]
    grant_sql = [call.args[0] for call in cur.execute.call_args_list if call.args[0].startswith("GRANT ")]
    assert grant_sql == ['GRANT "tpcc" TO "scadmin" WITH ADMIN OPTION']
    assert alter_sql == [
        'ALTER ROLE "scadmin" SET synchronous_commit TO off',
        'ALTER ROLE "tpcc" SET synchronous_commit TO off',
    ]


def test_apply_durability_durable_resets_role():
    conn, cur = _mock_psycopg2({"scadmin", "tpcc"})
    psycopg2 = MagicMock()
    psycopg2.connect.return_value = conn
    import postgres_dbaas as mod

    with (
        patch.dict(sys.modules, {"psycopg2": psycopg2}),
        patch.object(mod, "PG_USER", "scadmin"),
        patch.object(mod, "PG_PASSWORD", "secret"),
        patch.object(mod, "PG_DB", "bench"),
        patch.object(mod, "_db_host", return_value="db.example"),
        patch.object(mod, "_db_port", return_value=5432),
        patch.object(mod, "_db_connect_kwargs", return_value={"sslmode": "require"}),
    ):
        _apply_durability("durable", ["scadmin", "tpcc"], sync_settable=True)

    alter_sql = [call.args[0] for call in cur.execute.call_args_list if call.args[0].startswith("ALTER ROLE")]
    grant_sql = [call.args[0] for call in cur.execute.call_args_list if call.args[0].startswith("GRANT ")]
    assert grant_sql == ['GRANT "tpcc" TO "scadmin" WITH ADMIN OPTION']
    assert alter_sql == [
        'ALTER ROLE "scadmin" RESET synchronous_commit',
        'ALTER ROLE "tpcc" RESET synchronous_commit',
    ]


def test_apply_durability_async_requires_settable():
    try:
        _apply_durability("async", ["scadmin"], sync_settable=False)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "does not allow relaxing" in str(exc)


if __name__ == "__main__":
    test_durability_roles_hammerdb_tpcc()
    test_durability_roles_hammerdb_tpch()
    test_durability_roles_benchbase()
    test_apply_durability_async_sets_role_off()
    test_apply_durability_durable_resets_role()
    test_apply_durability_async_requires_settable()
    print("ok")

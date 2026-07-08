import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

import postgres_dbaas as mod  # noqa: E402
from postgres_dbaas import _reset_benchmark_database  # noqa: E402


def _mock_conn():
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


def test_reset_benchmark_database_grants_workload_role_before_create_db():
    conn, cur = _mock_conn()
    psycopg2 = MagicMock()
    psycopg2.connect.return_value = conn

    with (
        patch.dict(sys.modules, {"psycopg2": psycopg2}),
        patch.object(mod, "PG_USER", "scadmin"),
        patch.object(mod, "PG_PASSWORD", "secret"),
        patch.object(mod, "PG_DB", "bench"),
        patch.object(mod, "_db_host", return_value="db.example"),
        patch.object(mod, "_db_port", return_value=5432),
        patch.object(mod, "_db_connect_kwargs", return_value={"sslmode": "prefer"}),
        patch.object(mod, "_fix_public_schema"),
    ):
        _reset_benchmark_database("tpcc", "tpcc", "tpcc")

    sql = [call.args[0] for call in cur.execute.call_args_list]
    grant_idx = sql.index('GRANT "tpcc" TO "scadmin" WITH ADMIN OPTION')
    create_db_idx = sql.index('CREATE DATABASE "tpcc" OWNER "tpcc"')
    assert grant_idx < create_db_idx


def test_reset_benchmark_database_grants_existing_workload_role():
    conn, cur = _mock_conn()

    def execute(sql, params=None):
        if sql.startswith("SELECT 1 FROM pg_roles"):
            cur.fetchone.return_value = (1,)

    cur.execute.side_effect = execute
    psycopg2 = MagicMock()
    psycopg2.connect.return_value = conn

    with (
        patch.dict(sys.modules, {"psycopg2": psycopg2}),
        patch.object(mod, "PG_USER", "scadmin"),
        patch.object(mod, "PG_PASSWORD", "secret"),
        patch.object(mod, "PG_DB", "bench"),
        patch.object(mod, "_db_host", return_value="db.example"),
        patch.object(mod, "_db_port", return_value=5432),
        patch.object(mod, "_db_connect_kwargs", return_value={"sslmode": "prefer"}),
        patch.object(mod, "_fix_public_schema"),
    ):
        _reset_benchmark_database("tpch", "tpch", "tpch")

    sql = [call.args[0] for call in cur.execute.call_args_list]
    assert 'GRANT "tpch" TO "scadmin" WITH ADMIN OPTION' in sql
    assert "CREATE ROLE" not in " ".join(sql)


if __name__ == "__main__":
    test_reset_benchmark_database_grants_workload_role_before_create_db()
    test_reset_benchmark_database_grants_existing_workload_role()
    print("ok")

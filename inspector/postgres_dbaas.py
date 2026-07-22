"""Run Postgres benchmarks against a managed database endpoint."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import docker

from benchmark_tiers import (
    BENCHBASE_RUN_SECONDS,
    BENCHBASE_WARMUP_SECONDS,
    BENCHMARK,
    concurrency_ladder,
    multi_vm_workload_params,
)
from db_dataset_cache import cdn_env_for_benchmark
from lib import DOCKER_OPTS, Meta, container_remove
from pg_repro import merge_postgres_into_stdout, safe_collect_postgres_repro
from resource_tracker import configure_resource_tracker_docker_opts

PG_USER = os.environ.get("SC_DB_USER", "scadmin")
PG_PASSWORD = os.environ.get("SC_DB_PASSWORD", "")
PG_DB = os.environ.get("SC_DB_NAME", "bench")
BOOTSTRAP_USER = os.environ.get("SC_DB_BOOTSTRAP_USER", PG_USER)
BOOTSTRAP_PASSWORD = os.environ.get("SC_DB_BOOTSTRAP_PASSWORD", PG_PASSWORD)
BENCHBASE_DB = "benchbase"
BOOTSTRAP_DB = os.environ.get("SC_DB_BOOTSTRAP_DATABASE", "postgres")
DB_WAIT_TIMEOUT_SEC = int(os.environ.get("DB_WAIT_TIMEOUT_SEC", "1200"))

_TRACKER_FORWARD_ENV = (
    "SENTINEL_API_TOKEN",
    "SENTINEL_API_BASE",
    "SENTINEL_API_URL",
)


def _db_host() -> str:
    host = os.environ.get("SC_DB_HOST", "").strip()
    if not host:
        raise RuntimeError("SC_DB_HOST is not configured")
    return host


def _db_port() -> int:
    return int(os.environ.get("SC_DB_PORT", "5432"))


def _mem_gib() -> float:
    raw = os.environ.get("MEM_GIB", "").strip()
    if raw:
        return float(raw)
    with Path("/proc/meminfo").open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return 16.0


def _db_sslmode() -> str:
    explicit = os.environ.get("SC_DB_SSLMODE", "").strip()
    if explicit:
        return explicit
    network_mode = os.environ.get("SC_PROVISION_NETWORK_MODE", "")
    if network_mode == "private_vnet":
        return "require"
    if network_mode == "private_vpc":
        return "prefer"
    return "prefer"


def _db_connect_kwargs() -> dict[str, str]:
    mode = _db_sslmode()
    if mode == "disable":
        return {}
    return {"sslmode": mode}


def _bootstrap_connect():
    import psycopg2

    return psycopg2.connect(
        host=_db_host(),
        port=_db_port(),
        user=BOOTSTRAP_USER,
        password=BOOTSTRAP_PASSWORD,
        dbname=BOOTSTRAP_DB,
        connect_timeout=10,
        **_db_connect_kwargs(),
    )


def _workload_admin_connect(*, dbname: str | None = None):
    import psycopg2

    return psycopg2.connect(
        host=_db_host(),
        port=_db_port(),
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=dbname or BOOTSTRAP_DB,
        connect_timeout=10,
        **_db_connect_kwargs(),
    )


def _ensure_workload_admin_role() -> None:
    """Create SC_DB_USER on GCP when bootstrap connects as postgres."""
    if _dbaas_vendor() != "gcp" or BOOTSTRAP_USER == PG_USER:
        return
    conn = _bootstrap_connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (PG_USER,))
            if not cur.fetchone():
                cur.execute(
                    f'CREATE USER "{PG_USER}" WITH PASSWORD %s CREATEDB CREATEROLE',
                    (PG_PASSWORD,),
                )
                cur.execute(f'GRANT cloudsqlsuperuser TO "{PG_USER}"')
            else:
                # cloudsqlsuperuser grants are not enough for CREATE DATABASE/ROLE on Cloud SQL.
                cur.execute(f'ALTER USER "{PG_USER}" CREATEDB CREATEROLE')
                cur.execute(f'GRANT cloudsqlsuperuser TO "{PG_USER}"')
    finally:
        conn.close()


def _bootstrap_managed_db() -> None:
    """Create the workload admin user and empty bench database (not managed by Pulumi)."""
    _ensure_workload_admin_role()

    if _dbaas_vendor() == "gcp" and BOOTSTRAP_USER != PG_USER:
        # postgres cannot SET ROLE scadmin, so scadmin must CREATE DATABASE directly.
        conn = _workload_admin_connect()
        create_sql = f'CREATE DATABASE "{PG_DB}"'
    else:
        conn = _bootstrap_connect()
        create_sql = (
            f'CREATE DATABASE "{PG_DB}" OWNER "{PG_USER}"'
            if BOOTSTRAP_USER != PG_USER
            else f'CREATE DATABASE "{PG_DB}"'
        )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DB,))
            if not cur.fetchone():
                cur.execute(create_sql)
    finally:
        conn.close()
    logging.info("Bootstrapped managed DB user=%s database=%s", PG_USER, PG_DB)


def _wait_bootstrap_ready() -> None:
    deadline = time.monotonic() + DB_WAIT_TIMEOUT_SEC
    last_err = None
    while time.monotonic() < deadline:
        try:
            conn = _bootstrap_connect()
            conn.close()
            logging.info("Managed DB bootstrap endpoint ready at %s:%s", _db_host(), _db_port())
            return
        except Exception as exc:
            last_err = exc
            time.sleep(10)
    raise TimeoutError(f"managed DB bootstrap endpoint not ready after {DB_WAIT_TIMEOUT_SEC}s: {last_err}")


def wait_db_ready() -> None:
    """Block until the managed Postgres endpoint accepts connections."""
    _wait_bootstrap_ready()
    _bootstrap_managed_db()
    import psycopg2

    host = _db_host()
    port = _db_port()
    deadline = time.monotonic() + DB_WAIT_TIMEOUT_SEC
    last_err = None
    while time.monotonic() < deadline:
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=PG_USER,
                password=PG_PASSWORD,
                dbname=PG_DB,
                connect_timeout=10,
                **_db_connect_kwargs(),
            )
            conn.close()
            logging.info("Managed DB ready at %s:%s db=%s user=%s", host, port, PG_DB, PG_USER)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(10)
    raise TimeoutError(f"managed DB not ready after {DB_WAIT_TIMEOUT_SEC}s: {last_err}")


def _sync_commit_session_settable() -> bool:
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=_db_host(),
            port=_db_port(),
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB,
            connect_timeout=10,
            **_db_connect_kwargs(),
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET synchronous_commit TO off")
            cur.execute("COMMIT")
        conn.close()
        return True
    except Exception:
        return False


def _durability_roles(_task) -> list[str]:
    return [PG_USER]


def _apply_durability(durability: str, roles: list[str], *, sync_settable: bool) -> None:
    """Set per-role synchronous_commit so benchmark clients honor SC_DURABILITY."""
    import psycopg2

    if durability == "async":
        if not sync_settable:
            raise RuntimeError(
                "async durability requires synchronous_commit=off; "
                "this managed DB does not allow relaxing it"
            )
        sync_value = "off"
    else:
        sync_value = None

    conn = psycopg2.connect(
        host=_db_host(),
        port=_db_port(),
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        connect_timeout=30,
        **_db_connect_kwargs(),
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        for role in roles:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if not cur.fetchone():
                continue
            if sync_value is None:
                cur.execute(f'ALTER ROLE "{role}" RESET synchronous_commit')
            else:
                cur.execute(f'ALTER ROLE "{role}" SET synchronous_commit TO {sync_value}')
    conn.close()


def _fix_public_schema(dbname: str, owner: str) -> None:
    import psycopg2

    conn = psycopg2.connect(
        host=_db_host(),
        port=_db_port(),
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=dbname,
        connect_timeout=30,
        **_db_connect_kwargs(),
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'ALTER SCHEMA public OWNER TO "{owner}"')
        cur.execute(f'GRANT ALL ON SCHEMA public TO "{owner}"')
    conn.close()


def _reset_benchmark_database(dbname: str, owner: str) -> None:
    """Drop and recreate a benchmark database; fix public schema for managed Postgres."""
    import psycopg2

    conn = psycopg2.connect(
        host=_db_host(),
        port=_db_port(),
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        connect_timeout=30,
        **_db_connect_kwargs(),
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (dbname,),
        )
        cur.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
        cur.execute(f'CREATE DATABASE "{dbname}" OWNER "{owner}"')
    conn.close()
    _fix_public_schema(dbname, owner)


def _ensure_benchbase_db() -> None:
    _reset_benchmark_database(BENCHBASE_DB, PG_USER)


def _provision_env() -> dict[str, str]:
    keys = (
        "SC_PROVISION_VENDOR_ID",
        "SC_PROVISION_NATIVE_ID",
        "SC_PROVISION_ENGINE_VERSION",
        "SC_PROVISION_HA_MODE",
        "SC_PROVISION_SKU_ID",
        "SC_PROVISION_CPU_COUNT",
        "SC_PROVISION_MEMORY_GIB",
        "SC_PROVISION_STORAGE_GIB",
        "SC_PROVISION_STORAGE_EDITION",
        "SC_PROVISION_STORAGE_TYPE",
        "SC_PROVISION_IOPS_TIER",
        "SC_PROVISION_DISK_IOPS",
        "SC_PROVISION_DISK_THROUGHPUT",
        "SC_PROVISION_CLIENT_INSTANCE",
        "SC_PROVISION_REGION",
        "SC_PROVISION_ZONE",
        "SC_PROVISION_NETWORK_MODE",
        "SC_PROVISION_SYNC_COMMIT_SETTABLE",
    )
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _benchmark_env(task, params, mem_gib: float, db_vcpus: int, client_vcpus: int) -> dict[str, str]:
    durability = getattr(task, "durability", "durable")
    profile_vus = concurrency_ladder(db_vcpus)
    env = {
        "SC_DB_HOST": _db_host(),
        "SC_DB_PORT": str(_db_port()),
        "SC_DB_USER": PG_USER,
        "SC_DB_PASSWORD": PG_PASSWORD,
        "SC_DB_NAME": BENCHBASE_DB,
        "SC_DB_VCPUS": str(db_vcpus),
        "SC_DB_MEM_GIB": str(mem_gib),
        "SC_CLIENT_VCPUS": str(client_vcpus),
        "SC_DURABILITY": durability,
        "SC_PROFILE": "1",
        "SC_PROFILE_VUS": ",".join(str(v) for v in profile_vus),
        "SC_RUN_VUS": str(params.run_vus),
        "SC_RUN_SECONDS": str(BENCHBASE_RUN_SECONDS),
        "SC_WARMUP_SECONDS": str(BENCHBASE_WARMUP_SECONDS),
        "SC_TOPOLOGY": "dbaas",
        "SC_DB_SSLMODE": _db_sslmode(),
        "SC_WORKLOAD": BENCHMARK,
        "SC_SCALEFACTOR": str(params.scale_units),
    }
    env.update(_provision_env())
    env.update(cdn_env_for_benchmark())
    return env


def _tracker_env(task) -> dict[str, str]:
    """Env vars for resource-tracker inside benchmark images (see companion.py)."""
    env = {
        "TRACKER_PROJECT_NAME": "inspector",
        "TRACKER_JOB_NAME": task.name,
        "TRACKER_EXTERNAL_RUN_ID": os.environ.get("GITHUB_RUN_ID", ""),
        "TRACKER_QUIET": "true",
    }
    for key in _TRACKER_FORWARD_ENV:
        if os.environ.get(key):
            env[key] = os.environ[key]
    if os.environ.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    return env


def _dbaas_repro_extra(
    task,
    *,
    mem_gib: float,
    db_vcpus: int,
    client_vcpus: int,
    params,
    sync_settable: bool,
) -> dict[str, Any]:
    """Top-level stdout fields useful for reproducing a DBaaS run."""
    profile_vus = concurrency_ladder(db_vcpus)
    return {
        "db_vcpus": db_vcpus,
        "client_vcpus": client_vcpus,
        "db_mem_gib": mem_gib,
        "profile_vus": profile_vus,
        "scalefactor": params.scale_units,
        "schema_gib": params.schema_gib,
        "benchmark_image": task.image,
        "sslmode": _db_sslmode(),
        "sync_commit_session_settable": sync_settable,
        "durability": getattr(task, "durability", "durable"),
    }


def run_dbaas_task(
    meta: Meta,
    task,
    data_dir: str | os.PathLike,
    gpu_count: float = 0.0,
) -> tuple[str | None, bytes, bytes]:
    sync_settable = _sync_commit_session_settable()
    os.environ["SC_PROVISION_SYNC_COMMIT_SETTABLE"] = "true" if sync_settable else "false"

    mem_gib = float(os.environ.get("MEM_GIB") or 0) or _mem_gib()
    db_vcpus = int(float(os.environ.get("SC_PROVISION_CPU_COUNT", os.environ.get("SC_DB_VCPUS", "4"))))
    client_vcpus = int(os.cpu_count() or 4)
    params = multi_vm_workload_params(db_vcpus, mem_gib)

    try:
        _ensure_benchbase_db()
    except Exception as exc:
        meta.error_msg = f"benchbase db setup failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    durability = getattr(task, "durability", "durable")
    try:
        _apply_durability(durability, _durability_roles(task), sync_settable=sync_settable)
    except Exception as exc:
        meta.error_msg = f"durability setup failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    env = _benchmark_env(task, params, mem_gib, db_vcpus, client_vcpus)
    env.update(_tracker_env(task))
    docker_opts = dict(DOCKER_OPTS)
    docker_opts["environment"] = env
    docker_opts["network_mode"] = "host"
    task_dir = os.path.join(data_dir, task.name)
    os.makedirs(task_dir, exist_ok=True)
    docker_opts = configure_resource_tracker_docker_opts(docker_opts, task_dir)
    c = None
    try:
        d = docker.from_env(timeout=1800)
        image = d.images.pull(task.image)
        image_ref = next(iter(image.attrs.get("RepoDigests") or []), task.image)
        env["TRACKER_CONTAINER_IMAGE"] = image_ref
        docker_opts["environment"] = env
        c = d.containers.run(task.image, task.command, **docker_opts)
        ts = time.time() + task.timeout.total_seconds()
        while time.time() < ts:
            time.sleep(0.1)
            c.reload()
            if c.status == "exited":
                break
        else:
            c.stop()
            meta.error_msg = f"Execution timed out after {task.timeout.total_seconds()}s"
            meta.end = datetime.now()
            meta.exit_code = 1
            return None, b"", meta.error_msg.encode()
        res = c.wait(timeout=60)
        meta.end = datetime.now()
        meta.exit_code = res["StatusCode"]
        stdout = c.logs(stdout=True, stderr=False)
        stderr = c.logs(stdout=False, stderr=True)
        if meta.exit_code != 0:
            meta.error_msg = stderr.decode("utf-8", errors="replace")[:500]
        postgres_repro = safe_collect_postgres_repro(
            host=_db_host(),
            port=_db_port(),
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=BENCHBASE_DB,
            connect_kwargs=_db_connect_kwargs(),
        )
        stdout = merge_postgres_into_stdout(
            stdout,
            postgres_repro,
            extra=_dbaas_repro_extra(
                task,
                mem_gib=mem_gib,
                db_vcpus=db_vcpus,
                client_vcpus=client_vcpus,
                params=params,
                sync_settable=sync_settable,
            ),
        )
        ver = task.image.rsplit(":", 1)[-1]
        return ver, stdout, stderr
    except Exception as exc:
        meta.error_msg = str(exc)
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()
    finally:
        if c is not None:
            container_remove(c)

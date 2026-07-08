"""Run Postgres benchmarks against a managed database endpoint."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import docker

from benchmark_tiers import (
    WORKLOADS,
    benchbase_scalefactor,
    multi_vm_profile_ladder,
    multi_vm_workload_params,
    wh_per_vu_min,
)
from lib import DOCKER_OPTS, Meta, container_remove

PG_USER = os.environ.get("SC_DB_USER", "scadmin")
PG_PASSWORD = os.environ.get("SC_DB_PASSWORD", "")
PG_DB = os.environ.get("SC_DB_NAME", "bench")
BENCHBASE_DB = "benchbase"
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


def wait_db_ready() -> None:
    """Block until the managed Postgres endpoint accepts connections."""
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
            logging.info("Managed DB ready at %s:%s", host, port)
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


def _reset_benchmark_database(dbname: str, owner: str, role_password: str | None = None) -> None:
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
        if role_password is not None:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (owner,))
            if not cur.fetchone():
                cur.execute(f'CREATE ROLE "{owner}" LOGIN PASSWORD %s', (role_password,))
        cur.execute(f'CREATE DATABASE "{dbname}" OWNER "{owner}"')
    conn.close()
    _fix_public_schema(dbname, owner)


def _ensure_benchbase_db() -> None:
    _reset_benchmark_database(BENCHBASE_DB, PG_USER)


def _ensure_hammerdb_prereqs(workload: str) -> None:
    """Reset HammerDB workload database so buildschema starts from a clean slate."""
    if workload == "tpcc":
        _reset_benchmark_database("tpcc", "tpcc", "tpcc")
    elif workload == "tpch":
        _reset_benchmark_database("tpch", "tpch", "tpch")
    else:
        raise RuntimeError(f"unsupported HammerDB workload: {workload}")


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
        "SC_PROVISION_IOPS_TIER",
        "SC_PROVISION_CLIENT_INSTANCE",
        "SC_PROVISION_REGION",
        "SC_PROVISION_ZONE",
        "SC_PROVISION_NETWORK_MODE",
        "SC_PROVISION_CACHE_TIER",
        "SC_PROVISION_SYNC_COMMIT_SETTABLE",
    )
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def _benchmark_env(task, params, mem_gib: float, db_vcpus: int, client_vcpus: int) -> dict[str, str]:
    wl = WORKLOADS.get(task.workload_proxy, {})
    build_vus = min(params.build_vus, client_vcpus)
    durability = getattr(task, "durability", "durable")
    profile_vus = multi_vm_profile_ladder(
        db_vcpus,
        params.scale_units,
        task.tool,
        task.workload_proxy,
        durability,
    )
    env = {
        "SC_DB_HOST": _db_host(),
        "SC_DB_PORT": str(_db_port()),
        "SC_DB_USER": PG_USER,
        "SC_DB_PASSWORD": PG_PASSWORD,
        "SC_DB_NAME": PG_DB if task.tool == "hammerdb" else BENCHBASE_DB,
        "SC_DB_VCPUS": str(db_vcpus),
        "SC_CLIENT_VCPUS": str(client_vcpus),
        "SC_CACHE_RATIO": str(task.cache_ratio),
        "SC_DURABILITY": durability,
        "SC_PROFILE": "1",
        "SC_PROFILE_VUS": ",".join(str(v) for v in profile_vus),
        "SC_BUILD_VUS": str(build_vus),
        "SC_RUN_VUS": str(params.run_vus),
        "SC_WAREHOUSES": str(params.scale_units),
        "SC_WH_PER_VU_MIN": str(wh_per_vu_min(db_vcpus)),
        "SC_TOPOLOGY": "dbaas",
        "SC_CACHE_TIER": task.cache_tier,
        "SC_DB_SSLMODE": _db_sslmode(),
    }
    env.update(_provision_env())
    if task.tool == "hammerdb":
        env["SC_WORKLOAD"] = wl.get("hammerdb", "tpcc")
    else:
        bench_name = wl.get("benchmark", "wikipedia")
        env["SC_WORKLOAD"] = bench_name
        env["SC_SCALEFACTOR"] = str(benchbase_scalefactor(bench_name, mem_gib, task.cache_ratio))
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
    params = multi_vm_workload_params(
        task.workload_proxy,
        task.tool,
        task.cache_ratio,
        db_vcpus,
        mem_gib,
        durability=getattr(task, "durability", "durable"),
    )

    if task.tool == "benchbase":
        try:
            _ensure_benchbase_db()
        except Exception as exc:
            meta.error_msg = f"benchbase db setup failed: {exc}"
            meta.end = datetime.now()
            meta.exit_code = 1
            return None, b"", str(exc).encode()
    elif task.tool == "hammerdb":
        try:
            wl = WORKLOADS.get(task.workload_proxy, {}).get("hammerdb", "tpcc")
            _ensure_hammerdb_prereqs(wl)
        except Exception as exc:
            meta.error_msg = f"hammerdb db setup failed: {exc}"
            meta.end = datetime.now()
            meta.exit_code = 1
            return None, b"", str(exc).encode()

    env = _benchmark_env(task, params, mem_gib, db_vcpus, client_vcpus)
    env.update(_tracker_env(task))
    docker_opts = dict(DOCKER_OPTS)
    docker_opts["environment"] = env
    docker_opts["network_mode"] = "host"
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

"""Server-side orchestration for multi-VM Postgres benchmarks."""

from __future__ import annotations

import base64
import logging
import os
import socket
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path

import docker

from benchmark_tiers import (
    WORKLOADS,
    benchbase_scalefactor,
    multi_vm_profile_ladder,
    multi_vm_workload_params,
    wh_per_vu_min,
)
from companion_protocol import BenchmarkResult, Ping, Pong, RunBenchmark, Shutdown
from lib import DOCKER_OPTS, Meta, container_remove
from resource_tracker import RESOURCE_TRACKER_OUTPUT_FILENAME

CONNECT_DEADLINE_SEC = int(os.environ.get("MP_CONNECT_DEADLINE_SEC", "600"))
CONNECT_RETRY_SEC = 5
PG_IMAGE = "ghcr.io/sparecores/benchmark-postgres-server:main"
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DB = "bench"
BENCHBASE_DB = "benchbase"
HAMMERDB_TPCC_DB = "tpcc"


class CompanionSession:
    def __init__(self) -> None:
        self._conn = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> None:
        if self._conn is not None:
            return
        host = _client_private_ip()
        port = _mp_port()
        authkey = base64.b64decode(os.environ["MP_AUTHKEY_B64"])
        deadline = time.monotonic() + CONNECT_DEADLINE_SEC
        last_err = None
        while time.monotonic() < deadline:
            try:
                conn = Client((host, port), authkey=authkey)
                conn.send(Ping())
                if isinstance(conn.recv(), Pong):
                    self._conn = conn
                    logging.info("Companion connected at %s:%s", host, port)
                    return
                conn.close()
            except Exception as exc:
                last_err = exc
                time.sleep(CONNECT_RETRY_SEC)
        raise TimeoutError(f"companion connect timeout: {last_err}")

    def shutdown(self, reason: str = "") -> None:
        if self._conn is None:
            return
        try:
            self._conn.send(Shutdown(reason=reason))
        except Exception:
            logging.exception("Failed to send Shutdown")
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None

    def run_benchmark(self, msg: RunBenchmark) -> BenchmarkResult:
        if self._conn is None:
            raise RuntimeError("companion not connected")
        self._conn.send(msg)
        reply = self._conn.recv()
        if not isinstance(reply, BenchmarkResult):
            raise RuntimeError(f"unexpected reply: {type(reply)}")
        return reply


_SESSION = CompanionSession()


def get_session() -> CompanionSession:
    return _SESSION


def _local_private_ip() -> str:
    env_ip = os.environ.get("DB_PRIVATE_IP", "").strip()
    if env_ip:
        return env_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _ip_placeholder(value: str) -> bool:
    return not value or "{" in value or "}" in value


def has_companion_client() -> bool:
    """True when this server was provisioned with a multi-VM benchmark client."""
    return not _ip_placeholder(os.environ.get("CLIENT_PRIVATE_IP", "").strip())


def _mp_port() -> int:
    return int(os.environ.get("MP_PORT", "18765"))


def _client_private_ip() -> str:
    client_ip = os.environ.get("CLIENT_PRIVATE_IP", "").strip()
    if _ip_placeholder(client_ip):
        raise RuntimeError(f"client_private_ip not configured (got {client_ip!r})")
    return client_ip


def _client_vcpus(db_vcpus: int) -> int:
    raw = os.environ.get("MULTI_VM_CLIENT_VCPUS", "").strip()
    if raw:
        return max(1, int(raw))
    return db_vcpus


def multi_vm_supported(vendor: str) -> bool:
    from sc_runner.resources import supported_vendors

    return vendor in supported_vendors


def pg_gucs(mem_gib: float, warehouses: int, durability: str = "durable", vcpus: int | None = None) -> list[str]:
    schema_gib = warehouses * 0.095
    sb_gb = max(1, min(int(mem_gib * 0.25), int(schema_gib * 1.05) + 1))
    ecs = min(int(mem_gib * 0.75), sb_gb * 4)
    ncpu = max(1, int(vcpus if vcpus is not None else os.cpu_count() or 4))
    mpw = min(ncpu, 128)
    # "async" removes the per-commit WAL fsync wait (synchronous_commit=off): the
    # OLTP score then reflects CPU/memory/lock scaling of the instance rather than
    # the provisioned disk's fsync latency, and is comparable across clouds. fsync
    # stays on, so the cluster is never at risk of corruption. "durable" keeps the
    # production-default synchronous_commit=on for the disclosed secondary metric.
    sync_commit = "off" if durability == "async" else "on"
    settings = [
        f"shared_buffers={sb_gb}GB",
        f"effective_cache_size={ecs}GB",
        "max_connections=400",
        f"max_parallel_workers={mpw}",
        f"max_worker_processes={mpw}",
        "max_parallel_workers_per_gather=2",
        f"synchronous_commit={sync_commit}",
        "wal_buffers=64MB",
        "max_wal_size=8GB",
        "min_wal_size=1GB",
        "checkpoint_completion_target=0.9",
        "random_page_cost=1.1",
        "effective_io_concurrency=128",
        "maintenance_work_mem=1GB",
    ]
    args: list[str] = []
    for setting in settings:
        args.extend(["-c", setting])
    return args


def _cleanup_stale_postgres(task) -> None:
    """Drop leftover host-network Postgres containers from prior benchmark tasks."""
    try:
        d = docker.from_env(timeout=120)
        keep = f"pg-{task.name}"
        for container in d.containers.list(all=True):
            name = container.name or ""
            if not name.startswith("pg-") or name == keep:
                continue
            logging.info("Removing stale postgres container %s", name)
            container_remove(container)
    except Exception:
        logging.exception("Stale postgres cleanup failed (non-fatal)")


def _wait_pg_port_free(port: int = 5432, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.5)


def _start_postgres(task, mem_gib: float, warehouses: int, vcpus: int) -> docker.models.containers.Container:
    _cleanup_stale_postgres(task)
    _wait_pg_port_free()
    d = docker.from_env(timeout=1800)
    gucs = pg_gucs(mem_gib, warehouses, durability=getattr(task, "durability", "durable"), vcpus=vcpus)
    cmd = [
        "postgres",
        "-c",
        "listen_addresses=*",
        *gucs,
    ]
    image = d.images.pull(PG_IMAGE)
    image_ref = next(iter(image.attrs.get("RepoDigests") or []), PG_IMAGE)
    tracker_env = {
        "TRACKER_PROJECT_NAME": "inspector",
        "TRACKER_JOB_NAME": f"{task.name}_postgres_server",
        "TRACKER_EXTERNAL_RUN_ID": os.environ.get("GITHUB_RUN_ID", ""),
        "TRACKER_CONTAINER_IMAGE": image_ref,
        "TRACKER_QUIET": "true",
        "SENTINEL_API_TOKEN": os.environ.get("SENTINEL_API_TOKEN", ""),
    }
    container = d.containers.run(
        PG_IMAGE,
        cmd,
        name=f"pg-{task.name}",
        environment={
            "POSTGRES_PASSWORD": PG_PASSWORD,
            "POSTGRES_USER": PG_USER,
            "POSTGRES_DB": PG_DB,
            **tracker_env,
        },
        **DOCKER_OPTS,
    )
    _wait_pg_ready(container)
    return container


def _wait_pg_ready(container, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        container.reload()
        if container.status != "running":
            logs = container.logs(tail=30).decode("utf-8", errors="replace")
            raise TimeoutError(
                f"postgres container exited before ready (status={container.status}): {logs[-1500:]}"
            )
        try:
            proc = container.exec_run(
                ["psql", "-U", PG_USER, "-d", PG_DB, "-tA", "-c", "SELECT 1"],
                environment={"PGPASSWORD": PG_PASSWORD},
            )
        except Exception as exc:
            last_err = exc
            time.sleep(2)
            continue
        if proc.exit_code == 0:
            return
        last_err = proc.output.decode("utf-8", errors="replace") if proc.output else proc.exit_code
        time.sleep(2)
    raise TimeoutError(f"postgres did not become ready: {last_err}")


def _ensure_database(container, name: str) -> None:
    proc = container.exec_run(
        [
            "psql",
            "-U",
            PG_USER,
            "-d",
            PG_DB,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            f"CREATE DATABASE {name}",
        ],
        environment={"PGPASSWORD": PG_PASSWORD},
    )
    if proc.exit_code == 0:
        return
    out = proc.output.decode("utf-8", errors="replace") if proc.output else ""
    if "already exists" in out.lower():
        return
    raise RuntimeError(f"CREATE DATABASE {name} failed: {out}")


def _verify_database(host: str, name: str) -> None:
    import psycopg2

    conn = psycopg2.connect(
        host=host,
        port=5432,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
        connect_timeout=10,
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if not cur.fetchone():
                raise RuntimeError(f"database {name!r} missing on {host}:5432 after CREATE DATABASE")
    finally:
        conn.close()


def _benchmark_env(task, db_host: str, params, mem_gib: float, db_vcpus: int, client_vcpus: int) -> dict[str, str]:
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
        "SC_DB_HOST": db_host,
        "SC_DB_PORT": "5432",
        "SC_DB_USER": PG_USER,
        "SC_DB_PASSWORD": PG_PASSWORD,
        "SC_DB_NAME": PG_DB,
        "SC_DB_VCPUS": str(db_vcpus),
        "SC_CLIENT_VCPUS": str(client_vcpus),
        "SC_SHIRT_SIZE": task.shirt_size,
        "SC_DURABILITY": durability,
        "SC_PROFILE": "1",
        "SC_PROFILE_VUS": ",".join(str(v) for v in profile_vus),
        "SC_BUILD_VUS": str(build_vus),
        "SC_RUN_VUS": str(params.run_vus),
        "SC_WAREHOUSES": str(params.scale_units),
        "SC_WH_PER_VU_MIN": str(wh_per_vu_min(db_vcpus)),
        "SC_TOPOLOGY": "multi_vm",
    }
    provisioned_gib = os.environ.get("PROVISIONED_DISK_GIB", "").strip()
    if provisioned_gib:
        env["SC_PROVISIONED_DISK_GIB"] = provisioned_gib
    for key, var in (
        ("SC_MULTI_VM_DB_DISK_TYPE", "MULTI_VM_DB_DISK_TYPE"),
        ("SC_MULTI_VM_DB_DISK_IOPS", "MULTI_VM_DB_DISK_IOPS"),
        ("SC_MULTI_VM_DB_DISK_THROUGHPUT", "MULTI_VM_DB_DISK_THROUGHPUT"),
    ):
        val = os.environ.get(var, "").strip()
        if val:
            env[key] = val
    if task.tool == "hammerdb":
        env["SC_WORKLOAD"] = wl.get("hammerdb", "tpcc")
    else:
        bench_name = wl.get("benchmark", "wikipedia")
        env["SC_WORKLOAD"] = bench_name
        env["SC_SCALEFACTOR"] = str(benchbase_scalefactor(bench_name, task.shirt_size))
    return env


def run_multi_vm_task(
    meta: Meta,
    task,
    data_dir: str | os.PathLike,
    gpu_count: float = 0.0,
) -> tuple[str | None, bytes, bytes]:
    session = get_session()
    pg_container = None
    try:
        session.connect()
    except Exception as exc:
        meta.error_msg = str(exc)
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    mem_gib = float(os.environ.get("MEM_GIB") or 0) or _mem_gib()
    vcpus = int(os.cpu_count() or 4)
    client_vcpus = _client_vcpus(vcpus)
    params = multi_vm_workload_params(
        task.workload_proxy,
        task.tool,
        task.shirt_size,
        vcpus,
        mem_gib,
        durability=getattr(task, "durability", "durable"),
    )
    db_host = _local_private_ip()
    pg_wh = params.scale_units

    try:
        pg_container = _start_postgres(task, mem_gib, pg_wh, vcpus)
        if task.tool == "benchbase":
            _ensure_database(pg_container, BENCHBASE_DB)
        elif task.tool == "hammerdb":
            _ensure_database(pg_container, HAMMERDB_TPCC_DB)
            _verify_database(db_host, HAMMERDB_TPCC_DB)
    except Exception as exc:
        meta.error_msg = f"postgres start failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    msg = RunBenchmark(
        task_name=task.name,
        image=task.image,
        env={
            **_benchmark_env(task, db_host, params, mem_gib, vcpus, client_vcpus),
            "SC_HAMMERDB_CLI_TIMEOUT": str(max(3600, int(task.timeout.total_seconds()) - 180)),
        },
        command=task.command or "",
        timeout_sec=int(task.timeout.total_seconds()),
        tracker_job_name=f"{task.name}_benchmark_client",
    )
    try:
        result = session.run_benchmark(msg)
    except Exception as exc:
        meta.error_msg = f"companion run failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()
    finally:
        if pg_container is not None:
            container_remove(pg_container)

    meta.end = datetime.now()
    meta.exit_code = result.exit_code
    meta.stdout_bytes = len(result.stdout.encode("utf-8"))
    meta.stderr_bytes = len(result.stderr.encode("utf-8"))
    if result.exit_code != 0 and not meta.error_msg:
        meta.error_msg = result.stderr[:500] if result.stderr else "benchmark failed"
    task_dir = os.path.join(data_dir, task.name)
    os.makedirs(task_dir, exist_ok=True)
    if result.resource_tracker_jsonl:
        tracker_path = os.path.join(task_dir, RESOURCE_TRACKER_OUTPUT_FILENAME)
        with open(tracker_path, "w", encoding="utf-8") as fh:
            fh.write(result.resource_tracker_jsonl)
    ver = task.image.rsplit(":", 1)[-1]
    return ver, result.stdout.encode("utf-8"), result.stderr.encode("utf-8")


def _mem_gib() -> float:
    with Path("/proc/meminfo").open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return 16.0


def shutdown_companion(reason: str) -> None:
    session = get_session()
    if session.connected:
        session.shutdown(reason=reason)
        return
    host = _client_private_ip()
    port = _mp_port()
    authkey = base64.b64decode(os.environ["MP_AUTHKEY_B64"])
    try:
        conn = Client((host, port), authkey=authkey)
        conn.send(Shutdown(reason=reason))
        conn.close()
        logging.info("Companion shutdown sent to %s:%s", host, port)
    except Exception:
        logging.exception("Companion shutdown failed for %s:%s", host, port)


def finalize_multi_vm(_data_dir: str | os.PathLike) -> None:
    if not has_companion_client():
        return
    shutdown_companion(reason="inspect complete")

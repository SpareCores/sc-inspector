"""Server-side orchestration for multi-VM Postgres benchmarks."""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import docker

from benchmark_tiers import (
    CONCURRENCY_LADDER_MAX,
    DB_RUN_SECONDS,
    DB_SETTLE_SECONDS,
    DB_WARMUP_SECONDS,
    IMPROVE_PCT,
    PGBENCH_GIB_PER_SCALE,
    PGBENCH_RO_CPU_SCALE,
    PGBENCH_RO_CPU_SCHEMA_GIB,
    concurrency_ladder,
    concurrency_profile_ro,
    concurrency_search_cap,
    max_connections_for_vcpus,
    pgbench_tpcb_max_clients,
    pgbench_tpcb_scale,
)
from companion_protocol import BenchmarkResult, Ping, Pong, RunBenchmark, Shutdown
from db_dataset_cache import cdn_env_for_benchmark
from lib import DB_DOCKER_OPTS, Meta, container_remove
from pg_repro import merge_postgres_into_stdout, safe_collect_postgres_repro
from pgtune_leopard import generate_for_host
from resource_tracker import RESOURCE_TRACKER_OUTPUT_FILENAME

# Must cover slow companion boots, but stay well below task timeouts so a dead
# client fails the multi-VM task instead of looking like an indefinite hang.
CONNECT_DEADLINE_SEC = int(os.environ.get("MP_CONNECT_DEADLINE_SEC", "1800"))
CONNECT_RETRY_SEC = 5
CONNECT_ATTEMPT_TIMEOUT_SEC = float(os.environ.get("MP_CONNECT_ATTEMPT_TIMEOUT_SEC", "5"))
PG_IMAGE = "ghcr.io/sparecores/benchmark-postgres-server:main"
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DB = "bench"
POSTGRES_LOG_FILENAME = "postgres.log"


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
        logging.info(
            "Connecting to companion at %s:%s (deadline %ss)",
            host,
            port,
            CONNECT_DEADLINE_SEC,
        )
        while time.monotonic() < deadline:
            try:
                # Fail each attempt quickly when the peer is down/powered off;
                # bare Client() can block on TCP for minutes and look hung.
                prev_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(CONNECT_ATTEMPT_TIMEOUT_SEC)
                try:
                    conn = Client((host, port), authkey=authkey)
                finally:
                    socket.setdefaulttimeout(prev_timeout)
                conn.send(Ping())
                if isinstance(conn.recv(), Pong):
                    self._conn = conn
                    logging.info("Companion connected at %s:%s", host, port)
                    return
                conn.close()
            except Exception as exc:
                last_err = exc
                time.sleep(CONNECT_RETRY_SEC)
        raise TimeoutError(f"companion connect timeout after {CONNECT_DEADLINE_SEC}s: {last_err}")

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


def pg_guc_settings(
    mem_gib: float,
    durability: str = "durable",
    *,
    vcpus: int | None = None,
    min_max_connections: int | None = None,
) -> tuple[dict[str, str], str]:
    """Return (name→value GUCs, pgtune share URL) for multi-VM Postgres.

    GUCs match https://pgtune.leopard.in.ua/ form defaults (dbType=web,
    hdType=ssd, dbVersion=18) with only RAM/CPU from the host. Durability
    overrides ``synchronous_commit``.
    """
    cpu = max(1, int(vcpus if vcpus is not None else os.cpu_count() or 4))
    result = generate_for_host(mem_gib=mem_gib, cpu_num=cpu)
    settings = dict(result.settings)
    settings["synchronous_commit"] = "off" if durability == "async" else "on"
    # Headroom for geometric search past nproc (pgbench clients).
    need = max_connections_for_vcpus(cpu)
    if min_max_connections is not None:
        need = max(need, int(min_max_connections))
    cur = int(str(settings.get("max_connections", "100")).split()[0])
    if need > cur:
        settings["max_connections"] = str(need)
    return settings, result.share_url


def pg_gucs(
    mem_gib: float,
    durability: str = "durable",
    *,
    vcpus: int | None = None,
    min_max_connections: int | None = None,
) -> list[str]:
    args: list[str] = []
    settings, _ = pg_guc_settings(
        mem_gib,
        durability=durability,
        vcpus=vcpus,
        min_max_connections=min_max_connections,
    )
    for name, value in settings.items():
        if name == "listen_addresses":
            continue  # set explicitly in _start_postgres
        args.extend(["-c", f"{name}={value}"])
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


def _start_postgres(
    task,
    mem_gib: float,
    durability: str,
    *,
    vcpus: int,
    min_max_connections: int | None = None,
) -> docker.models.containers.Container:
    _cleanup_stale_postgres(task)
    _wait_pg_port_free()
    d = docker.from_env(timeout=1800)
    gucs = pg_gucs(
        mem_gib,
        durability=durability,
        vcpus=vcpus,
        min_max_connections=min_max_connections,
    )
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
        **(getattr(task, "docker_opts", None) or DB_DOCKER_OPTS),
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


def _task_kind(task) -> str:
    """Classify multi-VM / DBaaS DB tasks for env + sizing."""
    family = (getattr(task, "benchmark_family", "") or "").lower()
    name = (getattr(task, "name", "") or "").lower()
    blob = f"{family} {name}"
    if "pgbench" in blob and "tpcb" in blob:
        return "pgbench_tpcb"
    if "pgbench" in blob:
        return "pgbench_ro"
    raise ValueError(f"unsupported DB benchmark task: {getattr(task, 'name', task)!r}")


def _profile_env(db_vcpus: int, *, kind: str) -> dict[str, str]:
    if kind == "pgbench_ro":
        anchors = concurrency_profile_ro(db_vcpus)
        return {
            "SC_PROFILE": "1",
            "SC_PROFILE_VUS": ",".join(str(v) for v in anchors),
            # Fixed profile only — no geometric ladder / adaptive search.
            "SC_PROFILE_SEARCH": "0",
            "SC_PROFILE_MAX_CLIENTS": str(anchors[-1]),
            "SC_PROFILE_HARD_MAX_CLIENTS": str(anchors[-1]),
            "SC_WARMUP_ONCE": "1",
            "SC_WARMUP_SECONDS": str(DB_WARMUP_SECONDS),
            "SC_SETTLE_SECONDS": str(DB_SETTLE_SECONDS),
            "SC_RUN_SECONDS": str(DB_RUN_SECONDS),
            "SC_RUN_VUS": str(anchors[-1]),
            "SC_CPU_SCALE": str(PGBENCH_RO_CPU_SCALE),
        }
    anchors = concurrency_ladder(db_vcpus)
    return {
        "SC_PROFILE": "1",
        "SC_PROFILE_VUS": ",".join(str(v) for v in anchors),
        "SC_PROFILE_SEARCH": "1",
        "SC_PROFILE_IMPROVE_PCT": str(IMPROVE_PCT),
        "SC_PROFILE_MAX_CLIENTS": str(concurrency_search_cap(db_vcpus)),
        "SC_PROFILE_HARD_MAX_CLIENTS": str(CONCURRENCY_LADDER_MAX),
        "SC_WARMUP_ONCE": "1",
        "SC_WARMUP_SECONDS": str(DB_WARMUP_SECONDS),
        "SC_SETTLE_SECONDS": str(DB_SETTLE_SECONDS),
        "SC_RUN_SECONDS": str(DB_RUN_SECONDS),
        "SC_RUN_VUS": str(anchors[-1]),
    }


def _benchmark_env(
    task,
    db_host: str,
    mem_gib: float,
    db_vcpus: int,
    client_vcpus: int,
    *,
    requested_gucs: dict[str, str] | None = None,
) -> dict[str, str]:
    durability = getattr(task, "durability", "durable")
    kind = _task_kind(task)
    env = {
        "SC_DB_HOST": db_host,
        "SC_DB_PORT": "5432",
        "SC_DB_USER": PG_USER,
        "SC_DB_PASSWORD": PG_PASSWORD,
        "SC_DB_VCPUS": str(db_vcpus),
        "SC_DB_MEM_GIB": str(mem_gib),
        "SC_CLIENT_VCPUS": str(client_vcpus),
        "SC_DURABILITY": durability,
        "SC_TOPOLOGY": "multi_vm",
        "SC_PG_IMAGE": PG_IMAGE,
        **_profile_env(db_vcpus, kind=kind),
    }
    if kind == "pgbench_ro":
        env.update(
            {
                "SC_WORKLOAD": "pgbench_ro",
                "SC_DB_NAME": PG_DB,
                "SC_PGBENCH_DB": "pgbench",
            }
        )
    else:
        scale = pgbench_tpcb_scale(mem_gib, db_vcpus)
        max_clients = pgbench_tpcb_max_clients(mem_gib, db_vcpus)
        env.update(
            {
                "SC_WORKLOAD": "pgbench_tpcb",
                "SC_SCALEFACTORS": str(scale),
                "SC_SCALEFACTOR": str(scale),
                "SC_PROFILE_MAX_CLIENTS": str(max_clients),
                "SC_PROFILE_HARD_MAX_CLIENTS": str(max_clients),
                "SC_DB_NAME": PG_DB,
                "SC_PGBENCH_DB": "pgbench",
            }
        )
    if requested_gucs:
        env["SC_PG_GUCS_REQUESTED"] = json.dumps(requested_gucs, sort_keys=True)
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
    env.update(cdn_env_for_benchmark())
    return env


def _multi_vm_repro_extra(
    task,
    *,
    mem_gib: float,
    db_vcpus: int,
    client_vcpus: int,
    pgtune_share_url: str = "",
) -> dict[str, Any]:
    """Top-level stdout fields useful for reproducing a multi-VM run."""
    kind = _task_kind(task)
    profile_vus = (
        concurrency_profile_ro(db_vcpus)
        if kind == "pgbench_ro"
        else concurrency_ladder(db_vcpus)
    )
    extra: dict[str, Any] = {
        "db_vcpus": db_vcpus,
        "client_vcpus": client_vcpus,
        "db_mem_gib": mem_gib,
        "profile_vus": profile_vus,
        "profile_search": kind != "pgbench_ro",
        "profile_max_clients": (
            profile_vus[-1]
            if kind == "pgbench_ro"
            else concurrency_search_cap(db_vcpus)
        ),
        "profile_hard_max_clients": (
            profile_vus[-1] if kind == "pgbench_ro" else CONCURRENCY_LADDER_MAX
        ),
        "pg_image": PG_IMAGE,
        "benchmark_image": task.image,
        "durability": getattr(task, "durability", "durable"),
        "workload_kind": kind,
    }
    if kind == "pgbench_ro":
        extra["cpu_scale"] = PGBENCH_RO_CPU_SCALE
        extra["schema_gib"] = PGBENCH_RO_CPU_SCHEMA_GIB
    else:
        scale = pgbench_tpcb_scale(mem_gib, db_vcpus)
        extra["scalefactor"] = scale
        extra["schema_gib"] = scale * PGBENCH_GIB_PER_SCALE
        extra["profile_max_clients"] = pgbench_tpcb_max_clients(mem_gib, db_vcpus)
        extra["profile_hard_max_clients"] = extra["profile_max_clients"]
    if pgtune_share_url:
        extra["pgtune_share_url"] = pgtune_share_url
    provisioned_gib = os.environ.get("PROVISIONED_DISK_GIB", "").strip()
    if provisioned_gib:
        extra["storage_gib"] = int(provisioned_gib)
    disk_type = os.environ.get("MULTI_VM_DB_DISK_TYPE", "").strip()
    if disk_type:
        extra["storage_type"] = disk_type
    disk_iops = os.environ.get("MULTI_VM_DB_DISK_IOPS", "").strip()
    if disk_iops:
        extra["disk_iops"] = int(disk_iops)
    disk_throughput = os.environ.get("MULTI_VM_DB_DISK_THROUGHPUT", "").strip()
    if disk_throughput:
        extra["disk_throughput_mb_s"] = int(disk_throughput)
    return extra


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
    durability = getattr(task, "durability", "durable")
    kind = _task_kind(task)
    db_host = _local_private_ip()
    # Server must allow every client the driver may open: RO peaks at 2·V;
    # TPC-B adaptive search uses CONCURRENCY_LADDER_MAX.
    min_max_connections = max_connections_for_vcpus(vcpus)
    if kind == "pgbench_tpcb":
        min_max_connections = max(
            min_max_connections,
            pgbench_tpcb_max_clients(mem_gib, vcpus) + 50,
        )
    requested_gucs, pgtune_share_url = pg_guc_settings(
        mem_gib,
        durability=durability,
        vcpus=vcpus,
        min_max_connections=min_max_connections,
    )

    try:
        pg_container = _start_postgres(
            task,
            mem_gib,
            durability,
            vcpus=vcpus,
            min_max_connections=min_max_connections,
        )
    except Exception as exc:
        meta.error_msg = f"postgres start failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    msg = RunBenchmark(
        task_name=task.name,
        image=task.image,
        env=_benchmark_env(
            task,
            db_host,
            mem_gib,
            vcpus,
            client_vcpus,
            requested_gucs=requested_gucs,
        ),
        command=task.command or "",
        timeout_sec=int(task.timeout.total_seconds()),
        tracker_job_name=f"{task.name}_benchmark_client",
    )
    postgres_repro: dict[str, Any] | None = None
    postgres_log = b""
    try:
        result = session.run_benchmark(msg)
        # Snapshot GUCs and server logs before tearing down the server container.
        postgres_repro = safe_collect_postgres_repro(
            host="127.0.0.1",
            port=5432,
            user=PG_USER,
            password=PG_PASSWORD,
            dbname=PG_DB,
            requested_gucs=requested_gucs,
        )
        if pg_container is not None:
            try:
                postgres_log = pg_container.logs(stdout=True, stderr=True)
            except Exception:
                logging.exception("Failed to collect postgres container logs")
    except Exception as exc:
        meta.error_msg = f"companion run failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        if pg_container is not None and not postgres_log:
            try:
                postgres_log = pg_container.logs(stdout=True, stderr=True)
            except Exception:
                pass
        if postgres_log:
            task_dir = os.path.join(data_dir, task.name)
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, POSTGRES_LOG_FILENAME), "wb") as fh:
                fh.write(postgres_log)
        return None, b"", str(exc).encode()
    finally:
        if pg_container is not None:
            container_remove(pg_container)

    stdout = merge_postgres_into_stdout(
        result.stdout.encode("utf-8"),
        postgres_repro,
        extra=_multi_vm_repro_extra(
            task,
            mem_gib=mem_gib,
            db_vcpus=vcpus,
            client_vcpus=client_vcpus,
            pgtune_share_url=pgtune_share_url,
        ),
    )
    meta.end = datetime.now()
    meta.exit_code = result.exit_code
    meta.stdout_bytes = len(stdout)
    meta.stderr_bytes = len(result.stderr.encode("utf-8"))
    if result.exit_code != 0 and not meta.error_msg:
        meta.error_msg = result.stderr[:500] if result.stderr else "benchmark failed"
    task_dir = os.path.join(data_dir, task.name)
    os.makedirs(task_dir, exist_ok=True)
    if result.resource_tracker_jsonl:
        tracker_path = os.path.join(task_dir, RESOURCE_TRACKER_OUTPUT_FILENAME)
        with open(tracker_path, "w", encoding="utf-8") as fh:
            fh.write(result.resource_tracker_jsonl)
    if postgres_log:
        with open(os.path.join(task_dir, POSTGRES_LOG_FILENAME), "wb") as fh:
            fh.write(postgres_log)
    ver = task.image.rsplit(":", 1)[-1]
    return ver, stdout, result.stderr.encode("utf-8")


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

"""Server-side orchestration for multi-VM Postgres benchmarks."""

from __future__ import annotations

import base64
import json
import logging
import os
import platform
import socket
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import docker

from benchmark_tiers import WORKLOADS, workload_for_cache_tier
from companion_protocol import BenchmarkResult, Ping, Pong, RunBenchmark, Shutdown
from lib import DOCKER_OPTS, Meta, container_remove

CONNECT_DEADLINE_SEC = int(os.environ.get("MP_CONNECT_DEADLINE_SEC", "600"))
CONNECT_RETRY_SEC = 5
PG_IMAGE = "postgres:18"
PG_USER = "postgres"
PG_PASSWORD = "postgres"
PG_DB = "bench"


class CompanionSession:
    def __init__(self) -> None:
        self._conn = None
        self._infra: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def infra(self, data_dir: str | os.PathLike) -> dict[str, Any]:
        if self._infra is None:
            self._infra = load_or_write_infra(data_dir)
        return self._infra

    def connect(self, data_dir: str | os.PathLike) -> None:
        if self._conn is not None:
            return
        infra = self.infra(data_dir)
        host = infra["client_private_ip"]
        port = int(infra.get("mp_port", 18765))
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


def load_or_write_infra(data_dir: str | os.PathLike) -> dict[str, Any]:
    path = Path(data_dir) / "infra.json"
    infra: dict[str, Any] = {}
    if path.is_file():
        infra = json.loads(path.read_text(encoding="utf-8"))

    client_ip = os.environ.get("CLIENT_PRIVATE_IP", "").strip()
    db_ip = _local_private_ip()
    runtime = {
        "topology": "multi_vm",
        "vendor": os.environ.get("VENDOR", ""),
        "db_instance": os.environ.get("INSTANCE", ""),
        "client_instance": os.environ.get("MULTI_VM_CLIENT_INSTANCE", ""),
        "db_cpu_architecture": platform.machine(),
        "client_cpu_architecture": os.environ.get("MULTI_VM_CLIENT_CPU_ARCH", ""),
        "region": os.environ.get("REGION", ""),
        "zone": os.environ.get("ZONE", ""),
        "db_private_ip": db_ip,
        "client_private_ip": client_ip,
        "mp_port": int(os.environ.get("MP_PORT", "18765")),
        "provisioned_disk_gib": int(os.environ.get("PROVISIONED_DISK_GIB", "128")),
        "client_disk_gib": int(os.environ.get("CLIENT_DISK_GIB", "30")),
        "network_mode": "private_vpc",
    }
    # Per-deployment private IPs can change between runs; always refresh from live host/env.
    infra.update(runtime)
    if _ip_placeholder(infra.get("client_private_ip", "")):
        raise RuntimeError(
            f"client_private_ip not configured (got {infra.get('client_private_ip')!r})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(infra, indent=2) + "\n", encoding="utf-8")
    return infra


def multi_vm_supported(vendor: str) -> bool:
    from sc_runner.resources import supported_vendors

    return vendor in supported_vendors


def pg_gucs(mem_gib: float, warehouses: int) -> list[str]:
    schema_gib = warehouses * 0.095
    sb_gb = max(1, min(int(mem_gib * 0.25), int(schema_gib * 1.05) + 1))
    ecs = min(int(mem_gib * 0.75), sb_gb * 4)
    mpw = min(int(os.cpu_count() or 4), 32)
    settings = [
        f"shared_buffers={sb_gb}GB",
        f"effective_cache_size={ecs}GB",
        "max_connections=400",
        f"max_parallel_workers={mpw}",
        f"max_worker_processes={mpw}",
        "max_parallel_workers_per_gather=2",
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


def _start_postgres(task, mem_gib: float, warehouses: int) -> docker.models.containers.Container:
    d = docker.from_env(timeout=1800)
    gucs = pg_gucs(mem_gib, warehouses)
    cmd = [
        "postgres",
        "-c",
        "listen_addresses=*",
        *gucs,
    ]
    tracker_env = {
        "TRACKER_PROJECT_NAME": "inspector",
        "TRACKER_JOB_NAME": f"{task.name}_postgres_server",
        "TRACKER_EXTERNAL_RUN_ID": os.environ.get("GITHUB_RUN_ID", ""),
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


def _benchmark_env(task, db_host: str, workload) -> dict[str, str]:
    wl = WORKLOADS.get(task.workload_proxy, {})
    env = {
        "SC_DB_HOST": db_host,
        "SC_DB_PORT": "5432",
        "SC_DB_USER": PG_USER,
        "SC_DB_PASSWORD": PG_PASSWORD,
        "SC_DB_NAME": PG_DB,
        "SC_CACHE_RATIO": str(task.cache_ratio),
        "SC_PROFILE": "1",
        "SC_BUILD_VUS": str(workload.build_vus),
        "SC_RUN_VUS": str(workload.run_vus),
        "SC_WAREHOUSES": str(workload.warehouses),
    }
    if task.tool == "hammerdb":
        env["SC_WORKLOAD"] = wl.get("hammerdb", "tpcc")
    else:
        env["SC_WORKLOAD"] = wl.get("benchmark", "wikipedia")
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
        session.connect(data_dir)
    except Exception as exc:
        meta.error_msg = str(exc)
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    mem_gib = float(os.environ.get("MEM_GIB") or 0) or _mem_gib()
    workload = workload_for_cache_tier(task.cache_ratio, int(os.cpu_count() or 4), mem_gib)
    db_host = session.infra(data_dir)["db_private_ip"]

    try:
        pg_container = _start_postgres(task, mem_gib, workload.warehouses)
    except Exception as exc:
        meta.error_msg = f"postgres start failed: {exc}"
        meta.end = datetime.now()
        meta.exit_code = 1
        return None, b"", str(exc).encode()

    msg = RunBenchmark(
        task_name=task.name,
        image=task.image,
        env=_benchmark_env(task, db_host, workload),
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

    task_dir = Path(data_dir) / task.name
    task_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(result.metrics_json)
    metrics.setdefault("topology", "multi_vm")
    metrics.setdefault("client_rtt_ms", metrics.get("client_rtt_ms"))
    metrics.setdefault("peak_concurrency", metrics.get("peak_concurrency"))
    (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if result.stdout:
        (task_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        (task_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    meta.end = datetime.now()
    meta.exit_code = result.exit_code
    meta.stdout_bytes = len(result.stdout.encode("utf-8"))
    meta.stderr_bytes = len(result.stderr.encode("utf-8"))
    if result.exit_code != 0 and not meta.error_msg:
        meta.error_msg = result.stderr[:500] if result.stderr else "benchmark failed"
    ver = task.image.rsplit(":", 1)[-1]
    return ver, result.stdout.encode("utf-8"), result.stderr.encode("utf-8")


def _mem_gib() -> float:
    with Path("/proc/meminfo").open(encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return 16.0


def shutdown_companion(data_dir: str | os.PathLike, reason: str) -> None:
    session = get_session()
    if session.connected:
        session.shutdown(reason=reason)
        return
    infra = load_or_write_infra(data_dir)
    host = infra["client_private_ip"]
    port = int(infra.get("mp_port", 18765))
    authkey = base64.b64decode(os.environ["MP_AUTHKEY_B64"])
    try:
        conn = Client((host, port), authkey=authkey)
        conn.send(Shutdown(reason=reason))
        conn.close()
        logging.info("Companion shutdown sent to %s:%s", host, port)
    except Exception:
        logging.exception("Companion shutdown failed for %s:%s", host, port)


def finalize_multi_vm(data_dir: str | os.PathLike) -> None:
    shutdown_companion(data_dir, reason="inspect complete")

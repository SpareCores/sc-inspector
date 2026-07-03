"""Benchmark client daemon for multi-VM Postgres benchmarks."""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import docker

from companion_protocol import BenchmarkResult, Ping, Pong, RunBenchmark, Shutdown
from lib import DOCKER_OPTS

CONNECT_ACCEPT_DEADLINE_SEC = int(os.environ.get("MP_ACCEPT_DEADLINE_SEC", "600"))
NICE_ENTRYPOINT = ["nice", "-n", "-20"]
DEFAULT_BENCHMARK_COMMAND = [
    "/usr/local/bin/resource-tracker",
    "--",
    "python3",
    "/benchmark.py",
]


def _authkey() -> bytes:
    raw = os.environ.get("MP_AUTHKEY_B64", "")
    if not raw:
        raise RuntimeError("MP_AUTHKEY_B64 not set")
    return base64.b64decode(raw)


def _mp_port() -> int:
    return int(os.environ.get("MP_PORT", "18765"))


def _poweroff(reason: str) -> None:
    logging.info("Companion shutting down: %s", reason)
    # user_data.sh powers off the host after this container exits.
    sys.exit(0)


def _benchmark_output_dirs() -> tuple[str, Path]:
    """Return host bind path for nested containers and in-container metrics path."""
    host_dir = os.environ.get("HOST_BENCHMARK_OUTPUT_DIR", "/tmp/benchmark-output")
    mount_dir = Path(os.environ.get("BENCHMARK_OUTPUT_MOUNT", "/benchmark-output"))
    if not mount_dir.is_dir():
        mount_dir = Path(host_dir)
    return host_dir, mount_dir


def _benchmark_command(msg: RunBenchmark) -> tuple[list[str], list[str]]:
    """Return docker (entrypoint, command) with highest scheduling priority."""
    if isinstance(msg.command, list):
        return NICE_ENTRYPOINT, msg.command
    if msg.command:
        return NICE_ENTRYPOINT, ["bash", "-lc", msg.command]
    return NICE_ENTRYPOINT, DEFAULT_BENCHMARK_COMMAND


def _docker_run(msg: RunBenchmark) -> BenchmarkResult:
    env = dict(os.environ)
    env.update(msg.env)
    env.setdefault("TRACKER_PROJECT_NAME", "inspector")
    env.setdefault("TRACKER_JOB_NAME", msg.tracker_job_name)
    env.setdefault("TRACKER_EXTERNAL_RUN_ID", os.environ.get("GITHUB_RUN_ID", ""))

    host_out, metrics_dir = _benchmark_output_dirs()
    metrics_path = metrics_dir / "metrics.json"
    if metrics_path.is_file():
        metrics_path.unlink()
    entrypoint, command = _benchmark_command(msg)
    container = None
    try:
        d = docker.from_env(timeout=msg.timeout_sec + 120)
        try:
            image = d.images.pull(msg.image)
            env["TRACKER_CONTAINER_IMAGE"] = next(
                iter(image.attrs.get("RepoDigests") or []),
                msg.image,
            )
        except Exception:
            logging.exception("Failed to pull %s", msg.image)
            env["TRACKER_CONTAINER_IMAGE"] = msg.image

        run_kwargs = {
            **DOCKER_OPTS,
            "environment": env,
            "volumes": {host_out: {"bind": "/output", "mode": "rw"}},
            "entrypoint": entrypoint,
        }
        container = d.containers.run(msg.image, command, **run_kwargs)

        deadline = time.monotonic() + msg.timeout_sec
        while time.monotonic() < deadline:
            container.reload()
            if container.status == "exited":
                break
            time.sleep(0.5)
        else:
            container.stop(timeout=10)

        res = container.wait(timeout=60)
        exit_code = int(res.get("StatusCode", 1))
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
    except Exception as exc:
        logging.exception("Benchmark container failed for %s", msg.task_name)
        return BenchmarkResult(exit_code=1, stdout="", stderr=str(exc))
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                logging.exception("Failed to remove benchmark container")

    metrics: dict = {}
    if exit_code == 0:
        if stdout.strip():
            try:
                metrics = json.loads(stdout)
            except Exception:
                logging.exception("Failed to parse benchmark stdout JSON")
        if not metrics and metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                logging.exception("Failed to parse metrics.json")
    return BenchmarkResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        metrics_json=metrics,
    )


def _handle(conn) -> None:
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError) as exc:
            _poweroff(f"connection lost: {exc}")
        if isinstance(msg, Ping):
            conn.send(Pong())
        elif isinstance(msg, RunBenchmark):
            logging.info("RunBenchmark %s", msg.task_name)
            try:
                result = _docker_run(msg)
            except Exception as exc:
                logging.exception("RunBenchmark %s failed", msg.task_name)
                result = BenchmarkResult(exit_code=1, stderr=str(exc))
            conn.send(result)
        elif isinstance(msg, Shutdown):
            logging.info("Shutdown: %s", msg.reason)
            conn.close()
            _poweroff("shutdown requested")
        else:
            logging.warning("Unknown message type: %s", type(msg))


def run_companion(vendor: str, instance: str, listen_port: int | None = None) -> None:
    port = listen_port or _mp_port()
    authkey = _authkey()
    deadline = time.monotonic() + CONNECT_ACCEPT_DEADLINE_SEC
    logging.info("Companion listening on 0.0.0.0:%s for %s/%s", port, vendor, instance)
    with Listener(("0.0.0.0", port), authkey=authkey) as listener:
        listener._listener._socket.settimeout(1.0)
        while time.monotonic() < deadline:
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            except Exception:
                time.sleep(1)
                continue
            peer = getattr(conn, "name", None) or "unknown"
            logging.info("Accepted connection from %s", peer)
            _handle(conn)
    _poweroff("accept deadline elapsed with no persistent session")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_companion(os.environ.get("VENDOR", ""), os.environ.get("INSTANCE", ""))

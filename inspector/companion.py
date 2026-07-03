"""Benchmark client daemon for multi-VM Postgres benchmarks."""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

from companion_protocol import BenchmarkResult, Ping, Pong, RunBenchmark, Shutdown

CONNECT_ACCEPT_DEADLINE_SEC = int(os.environ.get("MP_ACCEPT_DEADLINE_SEC", "600"))
RUN_WITH_TRACKER = Path("/usr/local/bin/run-with-resource-tracker.sh")


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


def _docker_run(msg: RunBenchmark) -> BenchmarkResult:
    env = dict(os.environ)
    env.update(msg.env)
    env.setdefault("TRACKER_PROJECT_NAME", "inspector")
    env.setdefault("TRACKER_JOB_NAME", msg.tracker_job_name)
    env.setdefault("TRACKER_EXTERNAL_RUN_ID", os.environ.get("GITHUB_RUN_ID", ""))
    cmd = ["docker", "run", "--rm", "--network=host", "--privileged"]
    for key, value in env.items():
        if value is not None:
            cmd.extend(["-e", f"{key}={value}"])
    out_dir = Path("/tmp/benchmark-output")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["-v", f"{out_dir}:/output"])
    image = msg.image
    if isinstance(msg.command, list):
        cmd.append(image)
        cmd.extend(msg.command)
    elif msg.command:
        cmd.extend([image, "bash", "-lc", msg.command])
    else:
        cmd.append(image)

    tracker_cmd = cmd
    if RUN_WITH_TRACKER.is_file():
        tracker_cmd = [str(RUN_WITH_TRACKER), *cmd]

    proc = subprocess.run(
        tracker_cmd,
        capture_output=True,
        text=True,
        timeout=msg.timeout_sec,
    )
    metrics: dict = {}
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Failed to parse metrics.json")
    return BenchmarkResult(
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
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
            result = _docker_run(msg)
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

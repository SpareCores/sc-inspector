"""Multiprocessing control-plane messages for multi-VM Postgres benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Ping:
    pass


@dataclass
class Pong:
    pass


@dataclass
class RunBenchmark:
    task_name: str
    image: str
    env: dict[str, str]
    command: str | list
    timeout_sec: int
    tracker_job_name: str


@dataclass
class BenchmarkResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    metrics_json: dict[str, Any] = field(default_factory=dict)
    resource_tracker_jsonl: str = ""


@dataclass
class Shutdown:
    reason: str = ""

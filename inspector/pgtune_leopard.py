#!/usr/bin/env python3
"""PGTune settings matching https://pgtune.leopard.in.ua/ (le0pard/pgtune).

Port of the selectors in
``src/features/configuration/configurationSlice.js`` and the formatting in
``src/common/components/configurationView/index.jsx``.

Site form defaults (when hardware is filled in and Generate is clicked without
changing the other dropdowns):

  dbVersion=18, osType=linux, dbType=web, totalMemoryUnit=GB,
  connectionNum=<empty/auto>, hdType=ssd, dbSize=mid_ram

Pass host RAM (GB) and CPU count; leave the rest at those defaults unless
overridden.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlencode

SIZE_UNIT = {
    "KB": 1024,
    "MB": 1048576,
    "GB": 1073741824,
    "TB": 1099511627776,
}

# Matches FORM_DEFAULTS / initialState on the website.
DEFAULT_DB_VERSION = 18
DEFAULT_OS_TYPE = "linux"
DEFAULT_DB_TYPE = "web"
DEFAULT_HD_TYPE = "ssd"
DEFAULT_DB_SIZE = "mid_ram"
DEFAULT_MEMORY_UNIT = "GB"

SITE_BASE = "https://pgtune.leopard.in.ua/"


@dataclass(frozen=True)
class PgTuneInput:
    total_memory: int
    cpu_num: int
    db_version: float = DEFAULT_DB_VERSION
    os_type: str = DEFAULT_OS_TYPE
    db_type: str = DEFAULT_DB_TYPE
    total_memory_unit: str = DEFAULT_MEMORY_UNIT
    connection_num: int | None = None
    hd_type: str = DEFAULT_HD_TYPE
    db_size: str = DEFAULT_DB_SIZE

    def share_url(self) -> str:
        params = {
            "dbVersion": str(int(self.db_version) if float(self.db_version).is_integer() else self.db_version),
            "osType": self.os_type,
            "dbType": self.db_type,
            "totalMemory": str(self.total_memory),
            "totalMemoryUnit": self.total_memory_unit,
            "cpuNum": str(self.cpu_num),
            "hdType": self.hd_type,
            "dbSize": self.db_size,
        }
        if self.connection_num is not None:
            params["connectionNum"] = str(self.connection_num)
        return f"{SITE_BASE}?{urlencode(params)}"


@dataclass
class PgTuneResult:
    inputs: PgTuneInput
    settings: dict[str, str]  # name -> formatted value (e.g. "16GB")
    ini: str
    share_url: str
    warnings: list[str] = field(default_factory=list)

    def docker_c_args(self) -> list[str]:
        out: list[str] = []
        for key, val in self.settings.items():
            out.extend(["-c", f"{key}={val}"])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": SITE_BASE,
            "github": "https://github.com/le0pard/pgtune",
            "share_url": self.share_url,
            "inputs": asdict(self.inputs),
            "settings": self.settings,
            "warnings": self.warnings,
            "ini": self.ini,
        }


def _fmt_kb(value_kb: float | int) -> str:
    value = int(value_kb)
    kb_per_gb = SIZE_UNIT["GB"] // SIZE_UNIT["KB"]
    kb_per_mb = SIZE_UNIT["MB"] // SIZE_UNIT["KB"]
    if value % kb_per_gb == 0:
        return f"{value // kb_per_gb}GB"
    if value % kb_per_mb == 0:
        return f"{value // kb_per_mb}MB"
    return f"{value}kB"


def generate(inp: PgTuneInput) -> PgTuneResult:
    """Return config identical to clicking Generate on pgtune.leopard.in.ua."""
    mem_bytes = inp.total_memory * SIZE_UNIT[inp.total_memory_unit]
    mem_kb = mem_bytes / SIZE_UNIT["KB"]
    db, os_, hd, dbsz = inp.db_type, inp.os_type, inp.hd_type, inp.db_size
    ver = float(inp.db_version)
    cpu = inp.cpu_num

    # max_connections
    if inp.connection_num is not None:
        max_conn = inp.connection_num
    else:
        max_conn = {"web": 200, "oltp": 300, "dw": 40, "desktop": 20, "mixed": 100}[db]

    # shared_buffers
    shared = {
        "web": math.floor(mem_kb / 4),
        "oltp": math.floor(mem_kb / 4),
        "dw": math.floor(mem_kb / 4),
        "desktop": math.floor(mem_kb / 16),
        "mixed": math.floor(mem_kb / 4),
    }[db]
    if ver < 10 and os_ == "windows":
        shared = min(shared, (512 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"])

    # huge_pages
    if os_ == "mac":
        huge_pages = "off"
    else:
        huge_pages = "try" if shared >= (2 * SIZE_UNIT["GB"]) // SIZE_UNIT["KB"] else "off"

    # effective_cache_size
    ecs = {
        "web": math.floor((mem_kb * 3) / 4),
        "oltp": math.floor((mem_kb * 3) / 4),
        "dw": math.floor((mem_kb * 3) / 4),
        "desktop": math.floor(mem_kb / 4),
        "mixed": math.floor((mem_kb * 3) / 4),
    }[db]

    # maintenance_work_mem
    maint = {
        "web": math.floor(mem_kb / 16),
        "oltp": math.floor(mem_kb / 16),
        "dw": math.floor(mem_kb / 8),
        "desktop": math.floor(mem_kb / 16),
        "mixed": math.floor(mem_kb / 16),
    }[db]
    memory_limit = (8 * SIZE_UNIT["GB"]) // SIZE_UNIT["KB"]
    if os_ == "windows" and ver <= 17:
        memory_limit = (2 * SIZE_UNIT["GB"]) // SIZE_UNIT["KB"]
    if maint >= memory_limit:
        if os_ == "windows" and ver <= 17:
            maint = memory_limit - (1 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"]
        else:
            maint = memory_limit

    # WAL sizes
    min_wal = {
        "web": (1024 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "oltp": (2048 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "dw": (4096 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "desktop": (100 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "mixed": (1024 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
    }[db]
    max_wal = {
        "web": (4096 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "oltp": (8192 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "dw": (16384 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "desktop": (2048 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
        "mixed": (4096 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"],
    }[db]

    checkpoint_completion_target = 0.9

    # wal_buffers
    wal_buffers = math.floor((3 * shared) / 100)
    max_wal_buffer = (16 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"]
    if wal_buffers > max_wal_buffer:
        wal_buffers = max_wal_buffer
    near = (14 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"]
    if near < wal_buffers < max_wal_buffer:
        wal_buffers = max_wal_buffer
    if wal_buffers < 32:
        wal_buffers = 32

    default_statistics_target = {"web": 100, "oltp": 100, "dw": 500, "desktop": 100, "mixed": 100}[db]

    if dbsz == "less_ram":
        random_page_cost = 1.1
    elif hd == "hdd":
        random_page_cost = 4
    elif db == "dw":
        random_page_cost = 4
    else:
        random_page_cost = 1.1

    effective_io_concurrency = None
    if os_ == "linux":
        effective_io_concurrency = {"hdd": 2, "ssd": 200, "san": 300, "nvme": 1000}[hd]

    parallel: list[tuple[str, int]] = []
    if cpu and cpu >= 4:
        workers_per_gather = math.ceil(cpu / 2)
        if db != "dw" and workers_per_gather > 4:
            workers_per_gather = 4
        parallel.append(("max_worker_processes", cpu))
        parallel.append(("max_parallel_workers_per_gather", workers_per_gather))
        if ver >= 10:
            parallel.append(("max_parallel_workers", cpu))
        if ver >= 11:
            pmaint = math.ceil(cpu / 2)
            if pmaint > 4:
                pmaint = 4
            parallel.append(("max_parallel_maintenance_workers", pmaint))

    parallel_for_work_mem = 1
    for k, v in parallel:
        if k == "max_worker_processes" and v > 0:
            parallel_for_work_mem = v
            break
    work_mem = (mem_kb - shared) / ((max_conn + parallel_for_work_mem) * 3)
    work_mem = {
        "web": math.floor(work_mem),
        "oltp": math.floor(work_mem),
        "dw": math.floor(work_mem / 2),
        "desktop": math.floor(work_mem / 6),
        "mixed": math.floor(work_mem / 2),
    }[db]
    if dbsz == "less_ram":
        work_mem = math.floor(work_mem * 1.3)
    elif dbsz == "greater_ram":
        work_mem = math.floor(work_mem * 0.9)
    min_work = (4 * SIZE_UNIT["MB"]) // SIZE_UNIT["KB"]
    if work_mem < min_work:
        work_mem = min_work

    jit = None
    if ver >= 12 and db in ("web", "oltp", "mixed"):
        jit = "off"

    wal_compression = None
    if ver >= 15:
        wal_compression = "lz4"
    elif ver >= 10:
        wal_compression = "on"

    autovacuum_max_workers = None
    if cpu:
        if cpu >= 32:
            autovacuum_max_workers = 5
        elif cpu >= 16:
            autovacuum_max_workers = 4

    autovacuum_work_mem = None
    threshold = (2 * SIZE_UNIT["GB"]) // SIZE_UNIT["KB"]
    if maint >= threshold:
        autovacuum_work_mem = threshold

    io_method = None
    io_workers = None
    if ver >= 18:
        io_method = "io_uring" if os_ == "linux" else "worker"
        if io_method != "io_uring" and cpu:
            io_workers_val = min(32, max(3, math.floor(cpu / 4)))
            if io_workers_val > 3:
                io_workers = io_workers_val

    warnings: list[str] = []
    if mem_bytes < 256 * SIZE_UNIT["MB"]:
        warnings.append("WARNING: this tool not being designed for low memory systems")
    elif mem_bytes > 100 * SIZE_UNIT["GB"]:
        warnings.append("WARNING: this tool not being designed for high memory systems")

    settings: dict[str, str] = {
        "max_connections": str(max_conn),
        "shared_buffers": _fmt_kb(shared),
        "effective_cache_size": _fmt_kb(ecs),
        "maintenance_work_mem": _fmt_kb(maint),
        "checkpoint_completion_target": str(checkpoint_completion_target),
        "wal_buffers": _fmt_kb(wal_buffers),
        "default_statistics_target": str(default_statistics_target),
        "random_page_cost": str(random_page_cost),
        "work_mem": _fmt_kb(work_mem),
        "huge_pages": huge_pages,
        "min_wal_size": _fmt_kb(min_wal),
        "max_wal_size": _fmt_kb(max_wal),
    }
    if effective_io_concurrency is not None:
        settings["effective_io_concurrency"] = str(effective_io_concurrency)
    for k, v in parallel:
        settings[k] = str(v)
    if jit is not None:
        settings["jit"] = jit
    if wal_compression is not None:
        settings["wal_compression"] = wal_compression
    if autovacuum_max_workers is not None:
        settings["autovacuum_max_workers"] = str(autovacuum_max_workers)
    if autovacuum_work_mem is not None:
        settings["autovacuum_work_mem"] = _fmt_kb(autovacuum_work_mem)
    if io_method is not None:
        settings["io_method"] = io_method
    if io_workers is not None:
        settings["io_workers"] = str(io_workers)
    if db == "desktop":
        settings["wal_level"] = "minimal"
        settings["max_wal_senders"] = "0"

    # INI in the same order as the website view.
    header = [
        f"# DB Version: {int(ver) if ver == int(ver) else ver}",
        f"# OS Type: {os_}",
        f"# DB Type: {db}",
        f"# Total Memory (RAM): {inp.total_memory} {inp.total_memory_unit}",
        f"# CPUs num: {cpu}",
        f"# Data Storage: {hd}",
        f"# DB Size vs RAM: {dbsz}",
        f"# Source: {SITE_BASE}",
        "",
    ]
    for w in warnings:
        header.append(f"# {w}")
    if warnings:
        header.append("")

    order = [
        "max_connections",
        "shared_buffers",
        "effective_cache_size",
        "maintenance_work_mem",
        "checkpoint_completion_target",
        "wal_buffers",
        "default_statistics_target",
        "random_page_cost",
        "effective_io_concurrency",
        "work_mem",
        "huge_pages",
        "min_wal_size",
        "max_wal_size",
        "max_worker_processes",
        "max_parallel_workers_per_gather",
        "max_parallel_workers",
        "max_parallel_maintenance_workers",
        "wal_level",
        "max_wal_senders",
        "jit",
        "wal_compression",
        "autovacuum_max_workers",
        "autovacuum_work_mem",
        "io_method",
        "io_workers",
    ]
    body = [f"{k} = {settings[k]}" for k in order if k in settings]
    ini = "\n".join(header + body) + "\n"
    # Keep settings dict insertion order matching body for docker args.
    ordered = {k: settings[k] for k in order if k in settings}

    return PgTuneResult(
        inputs=inp,
        settings=ordered,
        ini=ini,
        share_url=inp.share_url(),
        warnings=warnings,
    )


def generate_for_host(
    *,
    mem_gib: float,
    cpu_num: int,
    db_type: str = DEFAULT_DB_TYPE,
) -> PgTuneResult:
    """Website defaults + detected hardware (memory rounded down to whole GB)."""
    return generate(
        PgTuneInput(
            total_memory=max(1, int(math.floor(mem_gib))),
            cpu_num=cpu_num,
            db_type=db_type,
        )
    )


if __name__ == "__main__":
    import json
    import sys

    mem = float(sys.argv[1]) if len(sys.argv) > 1 else 64
    cpus = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    result = generate_for_host(mem_gib=mem, cpu_num=cpus)
    print(result.ini)
    print("# share_url:", result.share_url, file=sys.stderr)
    print(json.dumps(result.to_dict(), indent=2), file=sys.stderr)

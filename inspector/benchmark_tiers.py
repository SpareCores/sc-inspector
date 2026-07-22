"""Workload sizing and concurrency ladders for Postgres benchmarks.

Schema size scales with instance RAM (~1/4, capped at 16 GiB) so the working
set fits in Postgres caches (at/under typical ``shared_buffers`` from
pgtune web/SSD defaults). Concurrency is fixed to ``{1, ncpus/2, ncpus}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DISK_SCHEMA_RATIO = 2.0
BUILD_VU_CAP = 64
CLIENT_MIN_VCPUS = 4
CLIENT_ABSOLUTE_MAX_VCPUS = 2048

SCHEMA_RAM_FRAC = 0.25
SCHEMA_MAX_GIB = 16.0
MIN_MEM_GIB = 2.0

# Calibrated in sc-db-benchmark-tmp RESULTS: 2 min warmup + 5 min measure
# is enough for SKU ranking; longer windows mainly shrink variance.
BENCHBASE_RUN_SECONDS = 300
BENCHBASE_WARMUP_SECONDS = 120

# BenchBase wikipedia: measured on Postgres 18 (sf=100 -> 14.8 GiB).
GIB_PER_SCALE_UNIT = 14.803 / 100

WORKLOAD_PROXY = "read_heavy"
BENCHMARK = "wikipedia"


def db_disk_options(vendor: str, mem_gib: float) -> dict[str, Any]:
    """Provider-native disk options for a multi-VM DB host (shared with DBaaS)."""
    from db_storage import db_storage_plan

    return db_storage_plan(vendor, mem_gib).multi_vm_disk_opts()


@dataclass(frozen=True)
class MultiVmWorkloadParams:
    """Sizing passed to postgres + benchmark containers."""

    build_vus: int
    run_vus: int
    scale_units: int
    schema_gib: float


@dataclass(frozen=True)
class ClientRequirements:
    """Absolute mins for companion VM."""

    min_vcpus: int
    min_memory_gib: float = 2.0


def target_schema_gib(mem_gib: float) -> float:
    """On-disk schema target: ~1/4 RAM, capped at 16 GiB.

    With pgtune web/SSD defaults (``shared_buffers`` ≈ 25% RAM), this
    keeps the working set at or under shared buffers so the timed run stays
    cache-resident rather than storage-bound.
    """
    return min(max(float(mem_gib), 0.0) * SCHEMA_RAM_FRAC, SCHEMA_MAX_GIB)


def scale_units_for_mem(mem_gib: float) -> int:
    return max(1, int(round(target_schema_gib(mem_gib) / GIB_PER_SCALE_UNIT)))


def schema_gib_for_scale_units(scale_units: int) -> float:
    return int(scale_units) * GIB_PER_SCALE_UNIT


def schema_disk_gib(mem_gib: float) -> float:
    return target_schema_gib(mem_gib) * DISK_SCHEMA_RATIO


def mem_feasible(mem_gib: float) -> bool:
    return float(mem_gib) >= MIN_MEM_GIB


def concurrency_ladder(vcpus: int) -> list[int]:
    """Fixed rungs: concurrency 1, half cores, and full cores."""
    vcpus = max(1, int(vcpus))
    rungs = [1]
    if vcpus >= 2:
        rungs.append(max(1, vcpus // 2))
        rungs.append(vcpus)
    return sorted(set(rungs))


def client_max_vcpus(db_vcpus: int) -> int:
    return min(CLIENT_ABSOLUTE_MAX_VCPUS, max(1, int(db_vcpus)))


def companion_client_vcpus(build_vus: int, db_vcpus: int) -> int:
    db_vcpus = max(1, int(db_vcpus))
    cap = client_max_vcpus(db_vcpus)
    min_vcpus = min(CLIENT_MIN_VCPUS, db_vcpus)
    db_floor = max(min_vcpus, (db_vcpus + 1) // 2)
    build_need = (int(build_vus) + 3) // 4
    return min(cap, max(db_floor, build_need))


def multi_vm_workload_params(vcpus: int, mem_gib: float) -> MultiVmWorkloadParams:
    vcpus = max(1, int(vcpus))
    scale_units = scale_units_for_mem(mem_gib)
    ladder = concurrency_ladder(vcpus)
    return MultiVmWorkloadParams(
        build_vus=min(vcpus, BUILD_VU_CAP),
        run_vus=ladder[-1],
        scale_units=scale_units,
        schema_gib=schema_gib_for_scale_units(scale_units),
    )


def client_req(db_srv) -> ClientRequirements:
    build_vus = min(int(db_srv.vcpus), BUILD_VU_CAP)
    return ClientRequirements(
        min_vcpus=companion_client_vcpus(build_vus, db_srv.vcpus),
        min_memory_gib=2.0,
    )


def merge_client_requirements(reqs: list[ClientRequirements]) -> ClientRequirements:
    if not reqs:
        return ClientRequirements(min_vcpus=2, min_memory_gib=2.0)
    return ClientRequirements(
        min_vcpus=max(r.min_vcpus for r in reqs),
        min_memory_gib=max(r.min_memory_gib for r in reqs),
    )

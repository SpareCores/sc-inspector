"""Cache-tier workload sizing and client requirements for Postgres multi-VM benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from disk import DISK_USABLE_FRAC

CACHE_TIERS = [1.0, 0.3]  # C100, C30
BUFFER_FRAC = 0.25
WH_SIZE_GIB = 0.095
WH_PER_VU_MIN = 5
WH_PER_VU_MIN_LARGE = 20
DISK_SCHEMA_RATIO = 2.0
MAX_RUN_VUS = 500
BUILD_VU_CAP = 64
OS_HEADROOM_GIB = 2.0
# Companion driver VM: threaded VUs do not need 1 vCPU each.
CLIENT_MIN_VCPUS = 4
CLIENT_VUS_PER_VCPU = 4
# HammerDB clients mostly wait on the remote DB; keep companions small.
HAMMERDB_CLIENT_MAX_VCPUS = 8
# BenchBase light mixes (twitter, ycsb, …) can be client-bound on fast DBs; scale higher.
BENCHBASE_CLIENT_MAX_VCPUS = 16

WORKLOADS: dict[str, dict[str, Any]] = {
    "oltp_mixed": {"tool": "hammerdb", "hammerdb": "tpcc", "tiers": [1.0, 0.3]},
    "read_heavy": {"tool": "benchbase", "benchmark": "wikipedia", "tiers": [1.0]},
    "crud_simple": {"tool": "benchbase", "benchmark": "ycsb", "tiers": [1.0]},
    "olap": {"tool": "hammerdb", "hammerdb": "tpch", "tiers": [1.0]},
}

# BenchBase wikipedia: ~8 GiB at scalefactor 50 (scratch calibration on F16-class hosts).
WIKIPEDIA_GIB_PER_SF = 8.0 / 50
# BenchBase ycsb: scalefactor * 1000 rows; default row layout is ~1 KiB.
YCSB_ROW_KIB = 1.0


@dataclass(frozen=True)
class Workload:
    cache_ratio: float
    vcpus: int
    mem_gib: float
    schema_gib: float
    warehouses: int
    run_vus: int
    build_vus: int
    disk_gib: float
    cpu_bound: bool = False
    ram_bound: bool = False


@dataclass(frozen=True)
class MultiVmWorkloadParams:
    """Sizing passed to multi-VM postgres + benchmark containers."""

    build_vus: int
    run_vus: int
    scale_units: int
    schema_gib: float


@dataclass(frozen=True)
class ClientRequirements:
    """Absolute mins for companion VM — no memory_pct of DB host."""

    min_vcpus: int
    min_memory_gib: float = 2.0


def wh_per_vu_min(vcpus: int) -> int:
    return WH_PER_VU_MIN_LARGE if vcpus >= 16 else WH_PER_VU_MIN


def profile_points(vcpus: int) -> list[int]:
    """Concurrency ladder for adaptive profiling."""
    if vcpus <= 2:
        return [1, vcpus]
    if vcpus <= 8:
        return sorted({1, 2, 4, vcpus})
    if vcpus <= 32:
        return sorted({1, 4, 8, 16, min(vcpus, 24)})
    return sorted({1, 8, 16, 32, min(vcpus, 48)})


def profile_vu_upper_bound(vcpus: int) -> int:
    return max(profile_points(vcpus))


def client_vcpus_for_peak_vus(peak_vus: int, *, max_vcpus: int) -> int:
    """Companion CPU floor from peak VUs/terminals (~CLIENT_VUS_PER_VCPU per core), capped."""
    need = (peak_vus + CLIENT_VUS_PER_VCPU - 1) // CLIENT_VUS_PER_VCPU
    return max(CLIENT_MIN_VCPUS, min(max_vcpus, need))


def workload_for_cache_tier(
    cache_ratio: float,
    vcpus: int,
    mem_gib: float,
    peak_vu: int | None = None,
) -> Workload:
    buffer_gib = BUFFER_FRAC * mem_gib
    schema_gib = buffer_gib / cache_ratio
    wh_min = wh_per_vu_min(vcpus)
    warehouses = max(wh_min, int(schema_gib / WH_SIZE_GIB))
    warehouses = min(warehouses, 100_000)
    max_vu_by_wh = max(1, warehouses // wh_min)
    run_vus = peak_vu if peak_vu else max(1, min(vcpus, MAX_RUN_VUS, max_vu_by_wh))
    run_vus = min(run_vus, max_vu_by_wh)
    build_vus = min(vcpus, warehouses, BUILD_VU_CAP)
    disk_gib = schema_gib * DISK_SCHEMA_RATIO
    ram_bound = run_vus < min(vcpus, MAX_RUN_VUS)
    return Workload(
        cache_ratio=cache_ratio,
        vcpus=vcpus,
        mem_gib=mem_gib,
        schema_gib=schema_gib,
        warehouses=warehouses,
        run_vus=run_vus,
        build_vus=build_vus,
        disk_gib=disk_gib,
        cpu_bound=False,
        ram_bound=ram_bound,
    )


def cache_tier_feasible(
    cache_ratio: float,
    host_vcpus: float,
    host_mem_gib: float,
    host_disk_gib: float,
) -> bool:
    w = workload_for_cache_tier(cache_ratio, int(host_vcpus), host_mem_gib)
    if host_vcpus < 1 or host_mem_gib < 3:
        return False
    if w.schema_gib < 0.5:
        return False
    if BUFFER_FRAC * host_mem_gib + w.run_vus * 0.05 + OS_HEADROOM_GIB > host_mem_gib * 0.92:
        return False
    if w.disk_gib > host_disk_gib * DISK_USABLE_FRAC:
        return False
    return w.warehouses >= wh_per_vu_min(int(host_vcpus)) * w.run_vus


def benchbase_scalefactor(workload: str, mem_gib: float, cache_ratio: float = 1.0) -> int:
    schema_gib = BUFFER_FRAC * mem_gib / cache_ratio
    if workload == "wikipedia":
        return max(10, min(200, int(mem_gib / 2.5)))
    if workload == "ycsb":
        # BenchBase YCSB: scalefactor * 1000 rows (see sample_ycsb_config.xml).
        return max(100, min(500_000, int(schema_gib * 1024)))
    raise ValueError(workload)


def benchbase_schema_gib(benchmark: str, mem_gib: float, cache_ratio: float) -> float:
    sf = benchbase_scalefactor(benchmark, mem_gib, cache_ratio)
    if benchmark == "wikipedia":
        return sf * WIKIPEDIA_GIB_PER_SF
    if benchmark == "ycsb":
        return (sf * 1000) * YCSB_ROW_KIB / (1024 * 1024)
    raise ValueError(benchmark)


def benchbase_disk_gib_required(workload_proxy: str, cache_ratio: float, mem_gib: float) -> float:
    benchmark = WORKLOADS[workload_proxy]["benchmark"]
    return benchbase_schema_gib(benchmark, mem_gib, cache_ratio) * DISK_SCHEMA_RATIO


def benchbase_cache_tier_feasible(
    workload_proxy: str,
    cache_ratio: float,
    host_vcpus: float,
    host_mem_gib: float,
    host_disk_gib: float,
) -> bool:
    benchmark = WORKLOADS[workload_proxy]["benchmark"]
    schema_gib = benchbase_schema_gib(benchmark, host_mem_gib, cache_ratio)
    disk_gib = schema_gib * DISK_SCHEMA_RATIO
    peak_vus = profile_vu_upper_bound(int(host_vcpus))
    if host_vcpus < 1 or host_mem_gib < 3:
        return False
    if schema_gib < 0.5:
        return False
    if BUFFER_FRAC * host_mem_gib + peak_vus * 0.05 + OS_HEADROOM_GIB > host_mem_gib * 0.92:
        return False
    if disk_gib > host_disk_gib * DISK_USABLE_FRAC:
        return False
    return True


def hammerdb_disk_gib_required(
    hammerdb_workload: str,
    cache_ratio: float,
    host_vcpus: int,
    host_mem_gib: float,
) -> float:
    if hammerdb_workload == "tpch":
        schema_gib = BUFFER_FRAC * host_mem_gib / cache_ratio
        return schema_gib * DISK_SCHEMA_RATIO
    return workload_for_cache_tier(cache_ratio, host_vcpus, host_mem_gib).disk_gib


def hammerdb_cache_tier_feasible(
    hammerdb_workload: str,
    cache_ratio: float,
    host_vcpus: float,
    host_mem_gib: float,
    host_disk_gib: float,
) -> bool:
    if hammerdb_workload == "tpch":
        schema_gib = BUFFER_FRAC * host_mem_gib / cache_ratio
        disk_gib = schema_gib * DISK_SCHEMA_RATIO
        if host_vcpus < 1 or host_mem_gib < 3:
            return False
        return disk_gib <= host_disk_gib * DISK_USABLE_FRAC
    return cache_tier_feasible(cache_ratio, host_vcpus, host_mem_gib, host_disk_gib)


def multi_vm_workload_params(
    workload_proxy: str,
    tool: str,
    cache_ratio: float,
    vcpus: int,
    mem_gib: float,
) -> MultiVmWorkloadParams:
    wl = WORKLOADS[workload_proxy]
    if tool == "hammerdb":
        hammer = wl["hammerdb"]
        if hammer == "tpch":
            schema_gib = BUFFER_FRAC * mem_gib / cache_ratio
            scale_units = max(1, min(300, int(schema_gib)))
            run_vus = min(vcpus, MAX_RUN_VUS)
            build_vus = min(vcpus, BUILD_VU_CAP, scale_units)
            return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)
        w = workload_for_cache_tier(cache_ratio, vcpus, mem_gib)
        return MultiVmWorkloadParams(w.build_vus, w.run_vus, w.warehouses, w.schema_gib)
    benchmark = wl["benchmark"]
    schema_gib = benchbase_schema_gib(benchmark, mem_gib, cache_ratio)
    scale_units = benchbase_scalefactor(benchmark, mem_gib, cache_ratio)
    peak_vus = profile_vu_upper_bound(vcpus)
    run_vus = min(peak_vus, MAX_RUN_VUS)
    build_vus = min(vcpus, BUILD_VU_CAP)
    return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)


def hammerdb_client_req(db_srv, cache_ratio: float) -> ClientRequirements:
    mem_gib = db_srv.memory_amount / 1024
    w = workload_for_cache_tier(cache_ratio, db_srv.vcpus, mem_gib)
    peak_vus = profile_vu_upper_bound(db_srv.vcpus)
    min_vcpus = max(
        client_vcpus_for_peak_vus(peak_vus, max_vcpus=HAMMERDB_CLIENT_MAX_VCPUS),
        client_vcpus_for_peak_vus(w.build_vus, max_vcpus=HAMMERDB_CLIENT_MAX_VCPUS),
    )
    return ClientRequirements(
        min_vcpus=min_vcpus,
        min_memory_gib=2.0,
    )


def benchbase_client_req(db_srv, workload: str, cache_ratio: float) -> ClientRequirements:
    peak_vus = profile_vu_upper_bound(db_srv.vcpus)
    # Light BenchBase workloads may need more driver CPU than HammerDB on large DB hosts.
    max_vcpus = min(BENCHBASE_CLIENT_MAX_VCPUS, max(8, int(db_srv.vcpus) // 2))
    return ClientRequirements(
        min_vcpus=client_vcpus_for_peak_vus(peak_vus, max_vcpus=max_vcpus),
        min_memory_gib=2.0,
    )


def merge_client_requirements(reqs: list[ClientRequirements]) -> ClientRequirements:
    if not reqs:
        return ClientRequirements(min_vcpus=2, min_memory_gib=2.0)
    return ClientRequirements(
        min_vcpus=max(r.min_vcpus for r in reqs),
        min_memory_gib=max(r.min_memory_gib for r in reqs),
    )

"""Cache-tier workload sizing and client requirements for Postgres multi-VM benchmarks."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from disk import DISK_USABLE_FRAC

CACHE_TIERS = [1.0, 0.3]  # C100, C30
BUFFER_FRAC = 0.25
WH_SIZE_GIB = 0.095
WH_PER_VU_MIN = 5
WH_PER_VU_MIN_LARGE = 20
DISK_SCHEMA_RATIO = 2.0
# Async OLTP/YCSB profiling targets up to 2x DB vCPUs when warehouses allow.
ASYNC_PROFILE_VU_FACTOR = 2
# Safety valve only — normal async cap is 2x vCPUs, not this constant.
ASYNC_PROFILE_VU_ABSOLUTE_MAX = 4096
BUILD_VU_CAP = 64
OS_HEADROOM_GIB = 2.0
# Durable (disk-bound) profiling: extra VUs only queue on fsync, so cap the ladder low.
DURABLE_PROFILE_VCPU_CAP = 16
# Companion driver VM on a separate host from Postgres.
CLIENT_MIN_VCPUS = 4
CLIENT_ABSOLUTE_MAX_VCPUS = 2048
# Multi-VM: Postgres is the bottleneck before HammerDB/BenchBase drivers saturate.
# F16 measurement: ~6.5 client cores for both async and durable OLTP at 16 VUs.
# Size to ~½ DB vCPUs with a small bump for parallel schema-build loaders.
TPCH_SCALE_FACTORS = (1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
TPCH_GIB_PER_SF = 1.0

WORKLOADS: dict[str, dict[str, Any]] = {
    "oltp_mixed": {"tool": "hammerdb", "hammerdb": "tpcc", "tiers": [1.0, 0.3]},
    "read_heavy": {"tool": "benchbase", "benchmark": "wikipedia", "tiers": [1.0]},
    "crud_simple": {"tool": "benchbase", "benchmark": "ycsb", "tiers": [1.0]},
    "olap": {"tool": "hammerdb", "hammerdb": "tpch", "tiers": [1.0]},
}

# Storage tier for the multi-VM DB host. These benchmarks are commit-fsync bound, so
# the DB host needs low-latency SSD-backed storage with enough provisioned IOPS/throughput
# that the disclosed durable-commit metric is not throttled by the provider baseline.
# Values are provider-native and passed straight through to sc-runner's generic per-VM
# disk knobs; every override lives here on the sc-inspector side. Set the disk type to an
# empty string to fall back to the provider default storage.
DB_DISK_TIERS: dict[str, dict[str, Any]] = {
    "azure": {"disk_type": "Premium_LRS"},
    "gcp": {"disk_type": "pd-ssd"},
    "aws": {"disk_type": "gp3", "disk_iops": 16000, "disk_throughput": 1000},
}


def db_disk_options(vendor: str) -> dict[str, Any]:
    """Provider-native disk options for a multi-VM DB host, overridable via env vars.

    Returns a dict with optional ``disk_type`` / ``disk_iops`` / ``disk_throughput``
    keys. An empty (or absent) ``disk_type`` means: use the provider default storage.
    """
    opts = dict(DB_DISK_TIERS.get(vendor, {}))
    disk_type = os.environ.get("MULTI_VM_DB_DISK_TYPE")
    if disk_type is not None:
        opts["disk_type"] = disk_type or None
    disk_iops = os.environ.get("MULTI_VM_DB_DISK_IOPS")
    if disk_iops:
        opts["disk_iops"] = int(disk_iops)
    disk_throughput = os.environ.get("MULTI_VM_DB_DISK_THROUGHPUT")
    if disk_throughput:
        opts["disk_throughput"] = int(disk_throughput)
    if not opts.get("disk_type"):
        return {}
    # PremiumV2 without explicit knobs still beats Premium_LRS; default to P30-class
    # (5k IOPS / 200 MB/s) when MULTI_VM_DB_DISK_TYPE=PremiumV2_LRS is set alone.
    if opts.get("disk_type") in {"PremiumV2_LRS", "UltraSSD_LRS"}:
        opts.setdefault("disk_iops", 5000)
        opts.setdefault("disk_throughput", 200)
    return opts


# BenchBase: min scale units per terminal (matches benchmark-benchbase-postgres/benchmark.py).
BENCHBASE_UNITS_PER_VU_MIN = 5
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


def async_peak_vu_cap(vcpus: int) -> int:
    """Upper async profiling VU count for a DB host (2x vCPUs, warehouse permitting)."""
    vcpus = max(1, int(vcpus))
    return min(ASYNC_PROFILE_VU_ABSOLUTE_MAX, vcpus * ASYNC_PROFILE_VU_FACTOR)


def durable_peak_vu_cap(vcpus: int) -> int:
    """Upper durable profiling VU count — disk-bound, does not scale past min(vcpus, 16)."""
    return min(max(1, int(vcpus)), DURABLE_PROFILE_VCPU_CAP)


def peak_vu_cap(vcpus: int, durability: str = "async") -> int:
    if durability == "async":
        return async_peak_vu_cap(vcpus)
    return durable_peak_vu_cap(vcpus)


def tpch_scale_factor(cache_ratio: float, mem_gib: float) -> int:
    """Largest HammerDB TPC-H scale factor that fits the cache-tier schema budget."""
    schema_gib = BUFFER_FRAC * mem_gib / max(cache_ratio, 0.05)
    target = schema_gib / TPCH_GIB_PER_SF
    allowed = [sf for sf in TPCH_SCALE_FACTORS if sf <= max(target, 1)]
    return max(allowed) if allowed else TPCH_SCALE_FACTORS[0]


def _vcpu_ladder_rungs(vcpus: int) -> list[int]:
    """Base concurrency rungs, geometrically (2x) spaced up to the DB vCPU count.

    A doubling ladder keeps the low-concurrency scaling curve dense while covering
    the full range without a giant gap on large hosts — e.g. 800 vCPUs yields
    1, 2, 4, …, 512, 800 (~11 rungs, ~log2(vcpus)) instead of jumping 32 -> 800.
    """
    vcpus = max(1, int(vcpus))
    rungs = [1]
    v = 2
    while v < vcpus:
        rungs.append(v)
        v *= 2
    rungs.append(vcpus)
    return sorted(set(rungs))


def _async_oversubscribe_rungs(vcpus: int) -> list[int]:
    """Extra rungs above 1:1 VU/vCPU for CPU-bound (async) OLTP profiling."""
    vcpus = max(1, int(vcpus))
    rungs: list[int] = []
    for mult in (1.5, 2.0):
        vu = int(vcpus * mult)
        if vu > vcpus:
            rungs.append(vu)
    return rungs


def max_profile_vus_by_warehouses(warehouses: int, vcpus: int) -> int:
    """Upper VU count allowed by warehouse count for even-enough TPC-C spread."""
    return max(1, int(warehouses) // wh_per_vu_min(vcpus))


def max_profile_vus_by_scalefactor(scalefactor: int) -> int:
    """Upper terminal count allowed by BenchBase scale factor."""
    return max(1, int(scalefactor) // BENCHBASE_UNITS_PER_VU_MIN)


def profile_units_cap(
    tool: str,
    *,
    hammerdb: str = "",
    benchmark: str = "",
    scale_units: int,
    vcpus: int,
) -> int:
    """Workload-specific upper bound on profiling concurrency before vCPU/durability caps."""
    scale_units = max(1, int(scale_units))
    vcpus = max(1, int(vcpus))
    if tool == "benchbase":
        return max_profile_vus_by_scalefactor(scale_units)
    if hammerdb == "tpch":
        return max(1, scale_units // WH_PER_VU_MIN)
    return max_profile_vus_by_warehouses(scale_units, vcpus)


def multi_vm_profile_ladder(
    vcpus: int,
    scale_units: int,
    tool: str,
    workload_proxy: str,
    durability: str = "durable",
) -> list[int]:
    wl = WORKLOADS[workload_proxy]
    if tool == "hammerdb":
        units_cap = profile_units_cap(
            "hammerdb",
            hammerdb=wl.get("hammerdb", "tpcc"),
            scale_units=scale_units,
            vcpus=vcpus,
        )
    else:
        units_cap = profile_units_cap(
            "benchbase",
            benchmark=wl.get("benchmark", "wikipedia"),
            scale_units=scale_units,
            vcpus=vcpus,
        )
    return concurrency_ladder(vcpus, units_cap, durability)


def concurrency_ladder(
    vcpus: int,
    max_by_warehouses: int,
    durability: str = "durable",
) -> list[int]:
    """Adaptive profiling ladder, capped by warehouses and durability mode.

    ``async`` (CPU-bound): base rungs plus 1.5x/2x vCPU oversubscription when
    warehouses allow — finds peak NOPM when the DB still has unused cores.

    ``durable`` (disk-bound): base rungs only, capped at min(vcpus, 16) — extra
    VUs only pile up behind commit fsyncs and waste benchmark time.
    """
    vcpus = max(1, int(vcpus))
    max_by_warehouses = max(1, int(max_by_warehouses))
    rungs = list(_vcpu_ladder_rungs(vcpus))
    if durability == "async":
        rungs.extend(_async_oversubscribe_rungs(vcpus))
        cap = min(max_by_warehouses, async_peak_vu_cap(vcpus))
    else:
        cap = min(max_by_warehouses, durable_peak_vu_cap(vcpus))
    # Clamp (not drop) rungs above the cap so the ladder always reaches the ceiling.
    return sorted({max(1, min(r, cap)) for r in rungs})


def profile_points(vcpus: int) -> list[int]:
    """Concurrency ladder for adaptive profiling (BenchBase / legacy callers)."""
    return _vcpu_ladder_rungs(vcpus)


def profile_vu_upper_bound(
    vcpus: int,
    durability: str = "async",
    max_by_warehouses: int | None = None,
) -> int:
    wh_cap = (
        max_by_warehouses
        if max_by_warehouses is not None
        else peak_vu_cap(vcpus, durability)
    )
    ladder = concurrency_ladder(vcpus, wh_cap, durability)
    return max(ladder) if ladder else 1


def client_max_vcpus(db_vcpus: int) -> int:
    """Companion CPU ceiling (never above the DB host)."""
    return min(CLIENT_ABSOLUTE_MAX_VCPUS, max(1, int(db_vcpus)))


def companion_client_vcpus(build_vus: int, db_vcpus: int) -> int:
    """Companion CPU floor from DB host size and parallel build loaders."""
    db_vcpus = max(1, int(db_vcpus))
    cap = client_max_vcpus(db_vcpus)
    min_vcpus = min(CLIENT_MIN_VCPUS, db_vcpus)
    db_floor = max(min_vcpus, (db_vcpus + 1) // 2)
    build_need = (int(build_vus) + 3) // 4
    return min(cap, max(db_floor, build_need))


def workload_for_cache_tier(
    cache_ratio: float,
    vcpus: int,
    mem_gib: float,
    peak_vu: int | None = None,
    durability: str = "durable",
) -> Workload:
    buffer_gib = BUFFER_FRAC * mem_gib
    schema_gib = buffer_gib / cache_ratio
    wh_min = wh_per_vu_min(vcpus)
    warehouses = max(wh_min, int(schema_gib / WH_SIZE_GIB))
    warehouses = min(warehouses, 100_000)
    max_vu_by_wh = max_profile_vus_by_warehouses(warehouses, vcpus)
    if peak_vu is not None:
        run_vus = peak_vu
    elif durability == "async":
        run_vus = min(async_peak_vu_cap(vcpus), max_vu_by_wh)
    else:
        run_vus = max(1, min(durable_peak_vu_cap(vcpus), max_vu_by_wh))
    run_vus = min(run_vus, max_vu_by_wh)
    build_vus = min(vcpus, warehouses, BUILD_VU_CAP)
    disk_gib = schema_gib * DISK_SCHEMA_RATIO
    ram_bound = run_vus < min(vcpus, peak_vu_cap(vcpus, durability))
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
    durability: str = "async",
) -> bool:
    w = workload_for_cache_tier(cache_ratio, int(host_vcpus), host_mem_gib, durability=durability)
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
    durability: str = "durable",
) -> bool:
    benchmark = WORKLOADS[workload_proxy]["benchmark"]
    schema_gib = benchbase_schema_gib(benchmark, host_mem_gib, cache_ratio)
    disk_gib = schema_gib * DISK_SCHEMA_RATIO
    scale_units = benchbase_scalefactor(benchmark, host_mem_gib, cache_ratio)
    units_cap = max_profile_vus_by_scalefactor(scale_units)
    peak_vus = profile_vu_upper_bound(int(host_vcpus), durability, units_cap)
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
        sf = tpch_scale_factor(cache_ratio, host_mem_gib)
        return sf * TPCH_GIB_PER_SF * DISK_SCHEMA_RATIO
    return workload_for_cache_tier(cache_ratio, host_vcpus, host_mem_gib).disk_gib


def hammerdb_cache_tier_feasible(
    hammerdb_workload: str,
    cache_ratio: float,
    host_vcpus: float,
    host_mem_gib: float,
    host_disk_gib: float,
    durability: str = "async",
) -> bool:
    if hammerdb_workload == "tpch":
        sf = tpch_scale_factor(cache_ratio, host_mem_gib)
        disk_gib = sf * TPCH_GIB_PER_SF * DISK_SCHEMA_RATIO
        if host_vcpus < 1 or host_mem_gib < 3:
            return False
        return disk_gib <= host_disk_gib * DISK_USABLE_FRAC
    return cache_tier_feasible(
        cache_ratio, host_vcpus, host_mem_gib, host_disk_gib, durability=durability
    )


def multi_vm_workload_params(
    workload_proxy: str,
    tool: str,
    cache_ratio: float,
    vcpus: int,
    mem_gib: float,
    durability: str = "durable",
) -> MultiVmWorkloadParams:
    wl = WORKLOADS[workload_proxy]
    if tool == "hammerdb":
        hammer = wl["hammerdb"]
        if hammer == "tpch":
            scale_units = tpch_scale_factor(cache_ratio, mem_gib)
            schema_gib = scale_units * TPCH_GIB_PER_SF
            run_vus = min(vcpus, async_peak_vu_cap(vcpus), max(1, scale_units // WH_PER_VU_MIN))
            build_vus = min(vcpus, BUILD_VU_CAP)
            return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)
        w = workload_for_cache_tier(cache_ratio, vcpus, mem_gib, durability=durability)
        return MultiVmWorkloadParams(w.build_vus, w.run_vus, w.warehouses, w.schema_gib)
    benchmark = wl["benchmark"]
    schema_gib = benchbase_schema_gib(benchmark, mem_gib, cache_ratio)
    scale_units = benchbase_scalefactor(benchmark, mem_gib, cache_ratio)
    units_cap = max_profile_vus_by_scalefactor(scale_units)
    peak_vus = profile_vu_upper_bound(vcpus, durability, units_cap)
    run_vus = min(peak_vus, peak_vu_cap(vcpus, durability))
    build_vus = min(vcpus, BUILD_VU_CAP)
    return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)


def hammerdb_client_req(db_srv, cache_ratio: float) -> ClientRequirements:
    mem_gib = db_srv.memory_amount / 1024
    w = workload_for_cache_tier(cache_ratio, db_srv.vcpus, mem_gib)
    return ClientRequirements(
        min_vcpus=companion_client_vcpus(w.build_vus, db_srv.vcpus),
        min_memory_gib=2.0,
    )


def benchbase_client_req(db_srv, workload: str, cache_ratio: float) -> ClientRequirements:
    del workload, cache_ratio
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

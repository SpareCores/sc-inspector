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
MAX_RUN_VUS = 500
BUILD_VU_CAP = 64
OS_HEADROOM_GIB = 2.0
# Durable (disk-bound) profiling: extra VUs only queue on fsync, so cap the ladder low.
DURABLE_PROFILE_VCPU_CAP = 16
# Companion driver VM: threaded VUs do not need 1 vCPU each.
CLIENT_MIN_VCPUS = 4
CLIENT_VUS_PER_VCPU = 4
# HammerDB TPROC-C stored-procs: ~2 VUs per client vCPU before the Tcl driver
# saturates (F16 eval: 16 VUs on a 4-vCPU client pegged 3.6/4 CPUs). The heaviest
# client phases are schema build (parallel bulk load) and cache-resident C100 runs,
# so we size for the busiest VU/terminal count and keep HEADROOM above the observed
# saturation point — the DB, not the driver, must be the bottleneck at the top of
# the profile ladder.
HAMMERDB_CLIENT_VUS_PER_VCPU = 2
HAMMERDB_CLIENT_HEADROOM = 1.5
HAMMERDB_CLIENT_MAX_VCPUS = 32
# BenchBase light mixes (twitter, ycsb, …) can be client-bound on fast DBs; scale higher.
BENCHBASE_CLIENT_MAX_VCPUS = 16
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
    return opts


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


def tpch_scale_factor(cache_ratio: float, mem_gib: float) -> int:
    """Largest HammerDB TPC-H scale factor that fits the cache-tier schema budget."""
    schema_gib = BUFFER_FRAC * mem_gib / max(cache_ratio, 0.05)
    target = schema_gib / TPCH_GIB_PER_SF
    allowed = [sf for sf in TPCH_SCALE_FACTORS if sf <= max(target, 1)]
    return max(allowed) if allowed else TPCH_SCALE_FACTORS[0]


def _vcpu_ladder_rungs(vcpus: int) -> list[int]:
    """Base concurrency rungs scaled to DB vCPU count (1:1 region and below)."""
    vcpus = max(1, int(vcpus))
    if vcpus <= 2:
        return [1, vcpus]
    if vcpus <= 8:
        return [1, 2, 4, vcpus]
    if vcpus <= 32:
        return [1, 4, 8, 16, vcpus]
    return [1, 8, 16, 32, vcpus]


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
        cap = min(max_by_warehouses, MAX_RUN_VUS, max(vcpus, vcpus * 2))
    else:
        cap = min(max_by_warehouses, vcpus, DURABLE_PROFILE_VCPU_CAP)
    return sorted({max(1, min(r, cap)) for r in rungs if r <= cap})


def profile_points(vcpus: int) -> list[int]:
    """Concurrency ladder for adaptive profiling (BenchBase / legacy callers)."""
    return _vcpu_ladder_rungs(vcpus)


def profile_vu_upper_bound(
    vcpus: int,
    durability: str = "async",
    max_by_warehouses: int | None = None,
) -> int:
    wh_cap = max_by_warehouses if max_by_warehouses is not None else MAX_RUN_VUS
    ladder = concurrency_ladder(vcpus, wh_cap, durability)
    return max(ladder) if ladder else 1


def client_vcpus_for_peak_vus(
    peak_vus: int,
    *,
    max_vcpus: int,
    vus_per_vcpu: int = CLIENT_VUS_PER_VCPU,
) -> int:
    """Companion CPU floor from peak VUs/terminals, capped."""
    need = (peak_vus + vus_per_vcpu - 1) // vus_per_vcpu
    return max(CLIENT_MIN_VCPUS, min(max_vcpus, need))


def hammerdb_client_max_vcpus(db_vcpus: int, durability: str = "async") -> int:
    """HammerDB companion CPU ceiling.

    Durable runs rarely need more than a 1:1 match with the DB host. Async runs
    may profile up to 2x DB vCPUs in VUs, so allow ~1.5x client vCPUs for driver
    headroom (HAMMERDB_CLIENT_VUS_PER_VCPU saturation ratio).
    """
    db_vcpus = max(CLIENT_MIN_VCPUS, int(db_vcpus))
    if durability == "durable":
        return min(HAMMERDB_CLIENT_MAX_VCPUS, db_vcpus)
    driver_cap = max(db_vcpus, (db_vcpus * 3 + 1) // 2)
    return min(HAMMERDB_CLIENT_MAX_VCPUS, driver_cap)


def hammerdb_client_vcpus(
    peak_vus: int,
    build_vus: int,
    db_vcpus: int,
    durability: str = "async",
) -> int:
    cap = hammerdb_client_max_vcpus(db_vcpus, durability)
    # Size for the busiest client phase (peak run VUs or parallel build loaders) and
    # provision HEADROOM above the driver's saturation ratio so it stays below its
    # limit while the DB reaches peak concurrency.
    driver_vus = max(peak_vus, build_vus)
    need = math.ceil(driver_vus * HAMMERDB_CLIENT_HEADROOM / HAMMERDB_CLIENT_VUS_PER_VCPU)
    return max(CLIENT_MIN_VCPUS, min(cap, need))


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
        run_vus = min(MAX_RUN_VUS, max_vu_by_wh, max(vcpus, vcpus * 2))
    else:
        run_vus = max(1, min(vcpus, DURABLE_PROFILE_VCPU_CAP, MAX_RUN_VUS, max_vu_by_wh))
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
    peak_vus = profile_vu_upper_bound(int(host_vcpus), durability)
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
            run_vus = min(vcpus, MAX_RUN_VUS, max(1, scale_units // WH_PER_VU_MIN))
            build_vus = min(vcpus, BUILD_VU_CAP)
            return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)
        w = workload_for_cache_tier(cache_ratio, vcpus, mem_gib, durability=durability)
        return MultiVmWorkloadParams(w.build_vus, w.run_vus, w.warehouses, w.schema_gib)
    benchmark = wl["benchmark"]
    schema_gib = benchbase_schema_gib(benchmark, mem_gib, cache_ratio)
    scale_units = benchbase_scalefactor(benchmark, mem_gib, cache_ratio)
    peak_vus = profile_vu_upper_bound(vcpus, durability)
    run_vus = min(peak_vus, MAX_RUN_VUS)
    build_vus = min(vcpus, BUILD_VU_CAP)
    return MultiVmWorkloadParams(build_vus, run_vus, scale_units, schema_gib)


def hammerdb_client_req(
    db_srv,
    cache_ratio: float,
    durability: str = "async",
) -> ClientRequirements:
    mem_gib = db_srv.memory_amount / 1024
    w = workload_for_cache_tier(cache_ratio, db_srv.vcpus, mem_gib, durability=durability)
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, db_srv.vcpus)
    peak_vus = profile_vu_upper_bound(db_srv.vcpus, durability, wh_cap)
    return ClientRequirements(
        min_vcpus=hammerdb_client_vcpus(peak_vus, w.build_vus, db_srv.vcpus, durability=durability),
        min_memory_gib=2.0,
    )


def benchbase_client_req(
    db_srv,
    workload: str,
    cache_ratio: float,
    durability: str = "durable",
) -> ClientRequirements:
    peak_vus = profile_vu_upper_bound(db_srv.vcpus, durability)
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

"""Fixed shirt-size workload tiers and client requirements for Postgres multi-VM benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

WH_SIZE_GIB = 0.095
WH_PER_VU_MIN = 5
WH_PER_VU_MIN_LARGE = 20
DISK_SCHEMA_RATIO = 2.0
ASYNC_PROFILE_VU_FACTOR = 2
ASYNC_PROFILE_VU_ABSOLUTE_MAX = 4096
BUILD_VU_CAP = 64
OS_HEADROOM_GIB = 2.0
DURABLE_PROFILE_VCPU_CAP = 16
CLIENT_MIN_VCPUS = 4
CLIENT_ABSOLUTE_MAX_VCPUS = 2048
BUFFER_FRAC = 0.25

# Fixed shirt-size tiers: warehouse count, schema size, and minimum RAM.
# Each instance runs the tier(s) where its RAM falls within [min_ram_gib, max_ram_gib].
# Schema targets (HammerDB TPROC-C): XS ~1 GiB, S ~10 GiB, M ~100 GiB.
SHIRT_SIZES: dict[str, dict[str, Any]] = {
    "XS": {"warehouses": 11, "schema_gib": 11 * WH_SIZE_GIB, "min_ram_gib": 2, "max_ram_gib": float("inf")},
    "S": {"warehouses": 105, "schema_gib": 105 * WH_SIZE_GIB, "min_ram_gib": 8, "max_ram_gib": float("inf")},
    "M": {"warehouses": 1047, "schema_gib": 1047 * WH_SIZE_GIB, "min_ram_gib": 32, "max_ram_gib": float("inf")},
}

# Fixed BenchBase scale factors per shirt size (matched to the same schema targets).
BENCHBASE_SHIRT_SIZES: dict[str, dict[str, Any]] = {
    "XS": {"wikipedia_sf": 7, "ycsb_sf": 882},
    "S": {"wikipedia_sf": 68, "ycsb_sf": 8819},
    "M": {"wikipedia_sf": 676, "ycsb_sf": 88190},
}

WIKIPEDIA_GIB_PER_SF = 14.803 / 100  # measured on Postgres 18 (sf=100 -> 14.8 GiB)
YCSB_ROW_KIB = 1.189  # measured on Postgres 18 (sf=1049 -> 1.19 GiB)
BENCHBASE_UNITS_PER_VU_MIN = 5

WORKLOADS: dict[str, dict[str, Any]] = {
    "oltp_mixed": {"tool": "hammerdb", "hammerdb": "tpcc"},
    "read_heavy": {"tool": "benchbase", "benchmark": "wikipedia"},
    "crud_simple": {"tool": "benchbase", "benchmark": "ycsb"},
}

DB_DISK_TIERS: dict[str, dict[str, Any]] = {
    "azure": {"disk_type": "Premium_LRS"},
    "gcp": {"disk_type": "pd-ssd"},
    "aws": {"disk_type": "gp3", "disk_iops": 16000, "disk_throughput": 1000},
}


def db_disk_options(vendor: str) -> dict[str, Any]:
    """Provider-native disk options for a multi-VM DB host, overridable via env vars."""
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
    if opts.get("disk_type") in {"PremiumV2_LRS", "UltraSSD_LRS"}:
        opts.setdefault("disk_iops", 5000)
        opts.setdefault("disk_throughput", 200)
    return opts


@dataclass(frozen=True)
class MultiVmWorkloadParams:
    """Sizing passed to multi-VM postgres + benchmark containers."""
    build_vus: int
    run_vus: int
    scale_units: int
    schema_gib: float


@dataclass(frozen=True)
class ClientRequirements:
    """Absolute mins for companion VM."""
    min_vcpus: int
    min_memory_gib: float = 2.0


def wh_per_vu_min(vcpus: int) -> int:
    return WH_PER_VU_MIN_LARGE if vcpus >= 16 else WH_PER_VU_MIN


def async_peak_vu_cap(vcpus: int) -> int:
    vcpus = max(1, int(vcpus))
    return min(ASYNC_PROFILE_VU_ABSOLUTE_MAX, vcpus * ASYNC_PROFILE_VU_FACTOR)


def durable_peak_vu_cap(vcpus: int) -> int:
    return min(max(1, int(vcpus)), DURABLE_PROFILE_VCPU_CAP)


def peak_vu_cap(vcpus: int, durability: str = "async") -> int:
    if durability == "async":
        return async_peak_vu_cap(vcpus)
    return durable_peak_vu_cap(vcpus)


def shirt_size_feasible(size: str, mem_gib: float) -> bool:
    """True when the instance has enough RAM for the given shirt size."""
    tier = SHIRT_SIZES.get(size)
    if tier is None:
        return False
    return tier["min_ram_gib"] <= mem_gib <= tier["max_ram_gib"]


def shirt_size_disk_gib(size: str) -> float:
    """Disk required for a shirt-size tier (schema * DISK_SCHEMA_RATIO)."""
    tier = SHIRT_SIZES[size]
    return tier["schema_gib"] * DISK_SCHEMA_RATIO


def benchbase_shirt_size_schema_gib(benchmark: str, size: str) -> float:
    """Schema size for a BenchBase workload at a given shirt size."""
    bb = BENCHBASE_SHIRT_SIZES[size]
    if benchmark == "wikipedia":
        return bb["wikipedia_sf"] * WIKIPEDIA_GIB_PER_SF
    if benchmark == "ycsb":
        return (bb["ycsb_sf"] * 1000) * YCSB_ROW_KIB / (1024 * 1024)
    raise ValueError(benchmark)


def benchbase_shirt_size_disk_gib(workload_proxy: str, size: str) -> float:
    benchmark = WORKLOADS[workload_proxy]["benchmark"]
    return benchbase_shirt_size_schema_gib(benchmark, size) * DISK_SCHEMA_RATIO


def benchbase_scalefactor(benchmark: str, size: str) -> int:
    """Return the fixed BenchBase scale factor for a shirt size."""
    bb = BENCHBASE_SHIRT_SIZES[size]
    if benchmark == "wikipedia":
        return bb["wikipedia_sf"]
    if benchmark == "ycsb":
        return bb["ycsb_sf"]
    raise ValueError(benchmark)


def _vcpu_ladder_rungs(vcpus: int) -> list[int]:
    vcpus = max(1, int(vcpus))
    rungs = [1]
    v = 2
    while v < vcpus:
        rungs.append(v)
        v *= 2
    rungs.append(vcpus)
    return sorted(set(rungs))


def _async_oversubscribe_rungs(vcpus: int) -> list[int]:
    vcpus = max(1, int(vcpus))
    rungs: list[int] = []
    for mult in (1.5, 2.0):
        vu = int(vcpus * mult)
        if vu > vcpus:
            rungs.append(vu)
    return rungs


def max_profile_vus_by_warehouses(warehouses: int, vcpus: int) -> int:
    return max(1, int(warehouses) // wh_per_vu_min(vcpus))


def max_profile_vus_by_scalefactor(scalefactor: int) -> int:
    return max(1, int(scalefactor) // BENCHBASE_UNITS_PER_VU_MIN)


def profile_units_cap(
    tool: str,
    *,
    benchmark: str = "",
    scale_units: int,
    vcpus: int,
) -> int:
    scale_units = max(1, int(scale_units))
    vcpus = max(1, int(vcpus))
    if tool == "benchbase":
        return max_profile_vus_by_scalefactor(scale_units)
    return max_profile_vus_by_warehouses(scale_units, vcpus)


def concurrency_ladder(
    vcpus: int,
    max_by_warehouses: int,
    durability: str = "durable",
) -> list[int]:
    vcpus = max(1, int(vcpus))
    max_by_warehouses = max(1, int(max_by_warehouses))
    rungs = list(_vcpu_ladder_rungs(vcpus))
    if durability == "async":
        rungs.extend(_async_oversubscribe_rungs(vcpus))
        cap = min(max_by_warehouses, async_peak_vu_cap(vcpus))
    else:
        cap = min(max_by_warehouses, durable_peak_vu_cap(vcpus))
    return sorted({max(1, min(r, cap)) for r in rungs})


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


def multi_vm_profile_ladder(
    vcpus: int,
    scale_units: int,
    tool: str,
    workload_proxy: str,
    durability: str = "durable",
) -> list[int]:
    wl = WORKLOADS[workload_proxy]
    if tool == "benchbase":
        units_cap = profile_units_cap(
            "benchbase",
            benchmark=wl.get("benchmark", "wikipedia"),
            scale_units=scale_units,
            vcpus=vcpus,
        )
    else:
        units_cap = profile_units_cap(
            "hammerdb",
            scale_units=scale_units,
            vcpus=vcpus,
        )
    return concurrency_ladder(vcpus, units_cap, durability)


def client_max_vcpus(db_vcpus: int) -> int:
    return min(CLIENT_ABSOLUTE_MAX_VCPUS, max(1, int(db_vcpus)))


def companion_client_vcpus(build_vus: int, db_vcpus: int) -> int:
    db_vcpus = max(1, int(db_vcpus))
    cap = client_max_vcpus(db_vcpus)
    min_vcpus = min(CLIENT_MIN_VCPUS, db_vcpus)
    db_floor = max(min_vcpus, (db_vcpus + 1) // 2)
    build_need = (int(build_vus) + 3) // 4
    return min(cap, max(db_floor, build_need))


def multi_vm_workload_params(
    workload_proxy: str,
    tool: str,
    shirt_size: str,
    vcpus: int,
    mem_gib: float,
    durability: str = "durable",
) -> MultiVmWorkloadParams:
    """Compute fixed workload parameters from shirt size."""
    wl = WORKLOADS[workload_proxy]
    if tool == "hammerdb":
        tier = SHIRT_SIZES[shirt_size]
        warehouses = tier["warehouses"]
        schema_gib = tier["schema_gib"]
        max_vu_by_wh = max_profile_vus_by_warehouses(warehouses, vcpus)
        if durability == "async":
            run_vus = min(async_peak_vu_cap(vcpus), max_vu_by_wh)
        else:
            run_vus = max(1, min(durable_peak_vu_cap(vcpus), max_vu_by_wh))
        build_vus = min(vcpus, warehouses, BUILD_VU_CAP)
        return MultiVmWorkloadParams(build_vus, run_vus, warehouses, schema_gib)
    benchmark = wl["benchmark"]
    sf = benchbase_scalefactor(benchmark, shirt_size)
    schema_gib = benchbase_shirt_size_schema_gib(benchmark, shirt_size)
    units_cap = max_profile_vus_by_scalefactor(sf)
    peak_vus = profile_vu_upper_bound(vcpus, durability, units_cap)
    run_vus = min(peak_vus, peak_vu_cap(vcpus, durability))
    build_vus = min(vcpus, BUILD_VU_CAP)
    return MultiVmWorkloadParams(build_vus, run_vus, sf, schema_gib)


def hammerdb_client_req(db_srv, shirt_size: str) -> ClientRequirements:
    tier = SHIRT_SIZES[shirt_size]
    build_vus = min(int(db_srv.vcpus), tier["warehouses"], BUILD_VU_CAP)
    return ClientRequirements(
        min_vcpus=companion_client_vcpus(build_vus, db_srv.vcpus),
        min_memory_gib=2.0,
    )


def benchbase_client_req(db_srv) -> ClientRequirements:
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

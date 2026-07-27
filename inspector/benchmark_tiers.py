"""Workload sizing and concurrency ladders for Postgres pgbench.

Concurrency uses a static geometric ladder (powers of two + midpoints) so
machines can be compared at shared client counts. Per host we always measure
anchors ``{1, rung(V/4), rung(V/2), rung(V)}``, then walk up the ladder while
TPM improves by ≥ ``IMPROVE_PCT`` vs the best so far (driver-side early stop).

Schema sizing (keep variations few):
  * Discrete GiB rungs ``SCHEMA_SIZE_GIB`` = (1, 4, 16, 64). 64 GiB covers the
    concurrency ladder max (3072 clients ≈ 45 GiB of pgbench data).
  * Cache budget = ¼ RAM (≈ ``shared_buffers``); never pick a rung above it.
  * Disk / storage planning uses the largest cache-fitting rung.
  * pgbench RO — fixed ~1 GiB (``PGBENCH_RO_SCALE``); no lock contention.
  * pgbench TPC-B — single target: smallest rung whose scale covers
    ``concurrency_search_cap(V)`` (pgbench ``-s >= -c``), else the largest
    cache-fitting rung; clients still capped at scale as a safety net.
"""

from __future__ import annotations

from dataclasses import dataclass

DISK_SCHEMA_RATIO = 2.0
BUILD_VU_CAP = 64
CLIENT_MIN_VCPUS = 4
CLIENT_ABSOLUTE_MAX_VCPUS = 2048

SCHEMA_RAM_FRAC = 0.25
# Few fixed sizes for cross-SKU compare. 64 GiB ≥ gib(ladder_max clients).
SCHEMA_SIZE_GIB: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
MIN_MEM_GIB = 2.0

# Calibrated: 2 min warmup + 5 min measure is enough for SKU ranking;
# longer windows mainly shrink variance (see README-db.md design decisions).
DB_RUN_SECONDS = 300
DB_WARMUP_SECONDS = 120
# After the first warmup, later rungs only need a short settle (connection storm).
DB_SETTLE_SECONDS = 60

# Keep searching while a rung beats the best TPM by at least this fraction.
IMPROVE_PCT = 5.0
# Cap upward search at rung(SEARCH_VCPU_MULT * V) (and ladder max).
SEARCH_VCPU_MULT = 4

# Shared geometric ladder (Gergely): powers of two + midpoints.
GEOMETRIC_CONCURRENCY_LADDER: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
)

# pgbench: scale 65 → ~980 MB (fixed RO dataset).
PGBENCH_RO_SCALE = 65
PGBENCH_GIB_PER_SCALE = 0.9567 / 65


@dataclass(frozen=True)
class ClientRequirements:
    """Absolute mins for companion VM."""

    min_vcpus: int
    min_memory_gib: float = 2.0


def target_schema_gib(mem_gib: float) -> float:
    """Largest discrete schema rung that fits in ~shared_buffers (¼ RAM).

    Used for disk/storage planning so provisioned volume covers the biggest
    cache-resident dump we might load on this host.
    """
    budget = max(float(mem_gib), 0.0) * SCHEMA_RAM_FRAC
    fitting = [g for g in SCHEMA_SIZE_GIB if g <= budget + 1e-9]
    if fitting:
        return fitting[-1]
    return max(0.0625, budget)


def mem_feasible(mem_gib: float) -> bool:
    return float(mem_gib) >= MIN_MEM_GIB


def rung(x: float) -> int:
    """Snap a concurrency target onto ``GEOMETRIC_CONCURRENCY_LADDER``."""
    target = max(1.0, float(x))
    return min(GEOMETRIC_CONCURRENCY_LADDER, key=lambda v: (abs(v - target), v))


def concurrency_ladder(vcpus: int) -> list[int]:
    """Always-measured anchors: 1, V/4, V/2, V (snapped), deduped ascending."""
    v = max(1, int(vcpus))
    return sorted({1, rung(v / 4), rung(v / 2), rung(v)})


def concurrency_search_cap(vcpus: int) -> int:
    """Upper bound for upward search (inclusive)."""
    v = max(1, int(vcpus))
    return min(GEOMETRIC_CONCURRENCY_LADDER[-1], rung(SEARCH_VCPU_MULT * v))


def max_connections_for_vcpus(vcpus: int) -> int:
    """Postgres ``max_connections`` headroom for the highest search rung."""
    return concurrency_search_cap(vcpus) + 50


def pgbench_tpcb_scale(mem_gib: float, vcpus: int) -> int:
    """TPC-B ``-s`` for this host: cover ``search_cap(V)``, stay cache-resident.

    pgbench docs require ``-s >= max -c``. Pick the smallest ``SCHEMA_SIZE_GIB``
    rung that covers the search cap when RAM allows; otherwise the largest
    rung that still fits under ¼ RAM.
    """
    budget = max(float(mem_gib), 0.0) * SCHEMA_RAM_FRAC
    fitting = [g for g in SCHEMA_SIZE_GIB if g <= budget + 1e-9] or [
        max(0.0625, budget)
    ]
    need_gib = concurrency_search_cap(vcpus) * PGBENCH_GIB_PER_SCALE
    covering = [g for g in fitting if g + 1e-9 >= need_gib]
    gib = covering[0] if covering else fitting[-1]
    return max(1, int(round(gib / PGBENCH_GIB_PER_SCALE)))


def companion_client_vcpus(build_vus: int, db_vcpus: int) -> int:
    db_vcpus = max(1, int(db_vcpus))
    cap = min(CLIENT_ABSOLUTE_MAX_VCPUS, db_vcpus)
    min_vcpus = min(CLIENT_MIN_VCPUS, db_vcpus)
    db_floor = max(min_vcpus, (db_vcpus + 1) // 2)
    build_need = (int(build_vus) + 3) // 4
    # High client counts multiplex well (run10: ~540 clients on ~47 busy cores).
    search_need = (concurrency_search_cap(db_vcpus) + 9) // 10
    return min(cap, max(db_floor, build_need, search_need))


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

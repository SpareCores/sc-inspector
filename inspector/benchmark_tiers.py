"""Workload sizing and concurrency profiles for Postgres pgbench.

Read-only (cached CPU-heavy script):
  * Fixed concurrency profile ``{1, V/2, V, 2·V}`` — single-thread rank,
    half-machine, full vCPU saturate, and 2× oversubscribe (HT / SMT).
  * No geometric ladder and no adaptive upward search (service time ≫ RTT,
    so oversubscribe past ~2·V does not raise TPM).
  * Schema is the fixed ``ro_cpu_*`` dataset (~170 MB), not pgbench ``-s``.
  * Work multiplier ``PGBENCH_RO_CPU_SCALE`` (pgbench ``-D scale=N``).

TPC-B (tpcb-like, unchanged):
  * Geometric ladder anchors ``{1, V/4, V/2, V}`` + upward search while TPM
    improves by ≥ ``IMPROVE_PCT``.
  * Schema: smallest ``SCHEMA_SIZE_GIB`` rung covering ``search_cap(V)``,
    else largest cache-fitting rung; clients capped at scale (``-s >= -c``).
"""

from __future__ import annotations

from dataclasses import dataclass

DISK_SCHEMA_RATIO = 2.0
BUILD_VU_CAP = 64
CLIENT_MIN_VCPUS = 4
CLIENT_ABSOLUTE_MAX_VCPUS = 2048
# Companion picker search ceiling: min_vcpus * this. Heavy RO leaves the
# client mostly idle (see ClientRequirements docstring), so there's no need
# to search past a modest multiple of the computed floor.
CLIENT_MAX_VCPUS_MULTIPLIER = 2
# Heavy RO: client is mostly idle waiting on the server (N-128 DBaaS @ c=256/j=32
# used ≪4 of 64 vCPUs). 20 clients/vCPU is still a conservative fan-in budget.
CLIENTS_PER_CLIENT_VCPU = 20

SCHEMA_RAM_FRAC = 0.25
SCHEMA_SIZE_GIB: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
MIN_MEM_GIB = 2.0

# Calibrated: 2 min warmup + 5 min measure is enough for SKU ranking.
DB_RUN_SECONDS = 300
DB_WARMUP_SECONDS = 120
# After the first warmup, later rungs only need a short settle.
DB_SETTLE_SECONDS = 60

# TPC-B only: keep searching while a rung beats the best TPM by ≥ this %.
IMPROVE_PCT = 5.0
SEARCH_VCPU_MULT = 4

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
CONCURRENCY_LADDER_MAX = GEOMETRIC_CONCURRENCY_LADDER[-1]

# Legacy pgbench -i scale (TPC-B sizing helper).
PGBENCH_GIB_PER_SCALE = 0.9567 / 65
# Heavy RO: fixed schema; cpu_scale multiplies LIMIT/slice widths in the script.
PGBENCH_RO_CPU_SCALE = 1
PGBENCH_RO_CPU_SCHEMA_GIB = 0.17
# Cap pgbench -j on the client (many threads + deep work hurts more than helps).
PGBENCH_RO_MAX_JOBS = 32


@dataclass(frozen=True)
class ClientRequirements:
    """Sizing bounds for the companion VM.

    max_vcpus keeps the picker from wandering into oversized/exotic SKUs: the
    heavy-RO workload leaves the client mostly idle (a 64-vCPU client used
    <4 vCPUs driving a 128-vCPU-class DB host), so ranking by raw single-core
    stress-ng score with no ceiling can land on the single fastest chip in the
    whole catalog — e.g. a 48-vCPU bare-metal box — for a 4-vCPU primary.
    None means unbounded (kept for callers that don't set it explicitly).
    """

    min_vcpus: int
    min_memory_gib: float = 2.0
    max_vcpus: int | None = None


# DBaaS client sizing: unlike multi-VM (server on a companion VM we control),
# the managed DB does all the heavy RO work server-side, so the client driving
# pgbench against it barely does anything regardless of the DB's own size (see
# ClientRequirements docstring above). No need to scale with the DB SKU at
# all — a small, fixed, non-burstable instance is enough for any remote DB.
DBAAS_CLIENT_MIN_VCPUS = 2
DBAAS_CLIENT_MAX_VCPUS = 4
DBAAS_CLIENT_MIN_MEMORY_GIB = 4.0
DBAAS_CLIENT_MAX_MEMORY_GIB = 8.0
# Price ceiling isn't a fixed number here — companion_picker computes a Tukey
# IQR fence (Q3 + 1.5*IQR) per vendor/region each round instead, so it only
# excludes genuine outliers (e.g. a small-but-exotic accelerator SKU) rather
# than an arbitrary top slice of an otherwise reasonable, tightly-priced pool.


@dataclass(frozen=True)
class DbaasClientRequirements:
    """Fixed sizing bounds for the DBaaS pgbench client — deliberately not
    ClientRequirements: independent of DB size, and ranked by price (not
    single-core score). Bounded above on memory; the price ceiling is
    computed dynamically per vendor/region (see companion_picker).
    """

    min_vcpus: int = DBAAS_CLIENT_MIN_VCPUS
    max_vcpus: int = DBAAS_CLIENT_MAX_VCPUS
    min_memory_gib: float = DBAAS_CLIENT_MIN_MEMORY_GIB
    max_memory_gib: float = DBAAS_CLIENT_MAX_MEMORY_GIB


def dbaas_client_req() -> DbaasClientRequirements:
    """DBaaS client requirements: fixed, not derived from the DB target."""
    return DbaasClientRequirements()


def target_schema_gib(mem_gib: float) -> float:
    """Largest discrete schema rung that fits in ~shared_buffers (¼ RAM)."""
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


def concurrency_profile_ro(vcpus: int) -> list[int]:
    """RO fixed profile: 1, V/2, V, 2·V (deduped, ascending)."""
    v = max(1, int(vcpus))
    return sorted({1, max(1, v // 2), v, 2 * v})


def concurrency_ladder(vcpus: int) -> list[int]:
    """TPC-B always-measured anchors: 1, V/4, V/2, V (snapped), deduped."""
    v = max(1, int(vcpus))
    return sorted({1, rung(v / 4), rung(v / 2), rung(v)})


def concurrency_search_cap(vcpus: int) -> int:
    """TPC-B upper bound for upward search (inclusive)."""
    v = max(1, int(vcpus))
    return min(CONCURRENCY_LADDER_MAX, rung(SEARCH_VCPU_MULT * v))


def max_connections_for_vcpus(vcpus: int) -> int:
    """Postgres ``max_connections`` floor for multi-VM pgbench.

    Must cover TPC-B ladder max (adaptive search) plus a small admin reserve.
    ``vcpus`` kept for API symmetry.
    """
    _ = vcpus
    return CONCURRENCY_LADDER_MAX + 50


def pgbench_tpcb_scale(mem_gib: float, vcpus: int) -> int:
    """TPC-B ``-s`` for this host: cover ``search_cap(V)``, stay cache-resident."""
    budget = max(float(mem_gib), 0.0) * SCHEMA_RAM_FRAC
    fitting = [g for g in SCHEMA_SIZE_GIB if g <= budget + 1e-9] or [
        max(0.0625, budget)
    ]
    need_gib = concurrency_search_cap(vcpus) * PGBENCH_GIB_PER_SCALE
    covering = [g for g in fitting if g + 1e-9 >= need_gib]
    gib = covering[0] if covering else fitting[-1]
    return max(1, int(round(gib / PGBENCH_GIB_PER_SCALE)))


def pgbench_tpcb_max_clients(mem_gib: float, vcpus: int) -> int:
    """TPC-B concurrency cap derived from the cache-resident max scale."""
    return min(pgbench_tpcb_scale(mem_gib, vcpus), CONCURRENCY_LADDER_MAX)


def companion_design_clients(db_vcpus: int) -> int:
    """Concurrency the companion must drive without becoming the bottleneck.

    Sized for active workloads: heavy RO peaks at ``2·V``. TPC-B is disabled;
    if re-enabled, also cover ``concurrency_search_cap(V)``.
    """
    return concurrency_profile_ro(db_vcpus)[-1]


def companion_client_vcpus(build_vus: int, db_vcpus: int) -> int:
    """Minimum companion vCPUs so remote pgbench is not client-bound.

    Heavy RO is server/RTT-bound on the client side — do not mirror ``≈V/2``
    of the DB SKU. Size from max clients ÷ ``CLIENTS_PER_CLIENT_VCPU``, floored
    at ``CLIENT_MIN_VCPUS`` (and never above the DB vCPU count).
    """
    _ = build_vus  # reserved if build-heavy / TPC-B workloads return
    db_vcpus = max(1, int(db_vcpus))
    cap = min(CLIENT_ABSOLUTE_MAX_VCPUS, db_vcpus)
    min_vcpus = min(CLIENT_MIN_VCPUS, db_vcpus)
    design_c = companion_design_clients(db_vcpus)
    drive_need = (design_c + CLIENTS_PER_CLIENT_VCPU - 1) // CLIENTS_PER_CLIENT_VCPU
    return min(cap, max(min_vcpus, drive_need))


def client_req(db_srv) -> ClientRequirements:
    build_vus = min(int(db_srv.vcpus), BUILD_VU_CAP)
    min_vcpus = companion_client_vcpus(build_vus, db_srv.vcpus)
    return ClientRequirements(
        min_vcpus=min_vcpus,
        min_memory_gib=2.0,
        max_vcpus=min(CLIENT_ABSOLUTE_MAX_VCPUS, min_vcpus * CLIENT_MAX_VCPUS_MULTIPLIER),
    )


def merge_client_requirements(reqs: list[ClientRequirements]) -> ClientRequirements:
    if not reqs:
        return ClientRequirements(min_vcpus=2, min_memory_gib=2.0, max_vcpus=2 * CLIENT_MAX_VCPUS_MULTIPLIER)
    max_vcpus_values = [r.max_vcpus for r in reqs if r.max_vcpus is not None]
    return ClientRequirements(
        min_vcpus=max(r.min_vcpus for r in reqs),
        min_memory_gib=max(r.min_memory_gib for r in reqs),
        max_vcpus=max(max_vcpus_values) if len(max_vcpus_values) == len(reqs) else None,
    )

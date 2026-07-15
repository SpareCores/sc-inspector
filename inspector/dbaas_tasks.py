"""DBaaS Postgres benchmark task definitions (mirror of multi-VM Postgres in tasks.py).

DBaaS retains both async and durable durability modes: storage is a bundled,
differentiating factor of managed DB offerings, so fsync-bound scores are
meaningful for customers comparing DBaaS SKUs.

Fixed shirt-size tiers (XS / S / M) — same warehouse/scale-factor counts as
multi-VM so scores are comparable across topologies.
"""

from datetime import timedelta

from lib import DbaasDbTask

DBAAS_ROLLOUT = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
}

# ---------------------------------------------------------------------------
# Tier XS (11 warehouses / ~1 GiB schema)
# ---------------------------------------------------------------------------

hammerdb_postgres_dbaas_oltp_mixed_xs_async = DbaasDbTask(
    parallel=False,
    priority=0.9,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="XS",
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=30),
)

hammerdb_postgres_dbaas_oltp_mixed_xs_durable = DbaasDbTask(
    parallel=False,
    priority=0.91,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="XS",
    durability="durable",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=30),
)

benchbase_postgres_dbaas_read_heavy_xs = DbaasDbTask(
    parallel=False,
    priority=0.92,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="read_heavy",
    shirt_size="XS",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=30),
)

benchbase_postgres_dbaas_crud_simple_xs = DbaasDbTask(
    parallel=False,
    priority=0.93,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="crud_simple",
    shirt_size="XS",
    durability="async",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=30),
)

# ---------------------------------------------------------------------------
# Tier S (105 warehouses / ~10 GiB schema)
# ---------------------------------------------------------------------------

hammerdb_postgres_dbaas_oltp_mixed_s_async = DbaasDbTask(
    parallel=False,
    priority=1.0,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="S",
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=60),
)

hammerdb_postgres_dbaas_oltp_mixed_s_durable = DbaasDbTask(
    parallel=False,
    priority=1.01,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="S",
    durability="durable",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=60),
)

benchbase_postgres_dbaas_read_heavy_s = DbaasDbTask(
    parallel=False,
    priority=1.02,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="read_heavy",
    shirt_size="S",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

benchbase_postgres_dbaas_crud_simple_s = DbaasDbTask(
    parallel=False,
    priority=1.03,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="crud_simple",
    shirt_size="S",
    durability="async",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

# ---------------------------------------------------------------------------
# Tier M (1047 warehouses / ~100 GiB schema)
# ---------------------------------------------------------------------------

hammerdb_postgres_dbaas_oltp_mixed_m_async = DbaasDbTask(
    parallel=False,
    priority=1.1,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="M",
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

hammerdb_postgres_dbaas_oltp_mixed_m_durable = DbaasDbTask(
    parallel=False,
    priority=1.11,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="M",
    durability="durable",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

benchbase_postgres_dbaas_read_heavy_m = DbaasDbTask(
    parallel=False,
    priority=1.12,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="read_heavy",
    shirt_size="M",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=120),
)

benchbase_postgres_dbaas_crud_simple_m = DbaasDbTask(
    parallel=False,
    priority=1.13,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="crud_simple",
    shirt_size="M",
    durability="async",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=120),
)

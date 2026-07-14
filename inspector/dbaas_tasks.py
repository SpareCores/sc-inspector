"""DBaaS Postgres benchmark task definitions (mirror of multi-VM Postgres in tasks.py).

DBaaS retains both async and durable durability modes: storage is a bundled,
differentiating factor of managed DB offerings, so fsync-bound scores are
meaningful for customers comparing DBaaS SKUs.

Fixed shirt-size tiers (S / M / L) — same warehouse/scale-factor counts as
multi-VM so scores are comparable across topologies.
"""

from datetime import timedelta

from lib import DbaasDbTask

DBAAS_ROLLOUT = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
}

# ---------------------------------------------------------------------------
# Tier S (100 warehouses / ~10 GiB schema) — 16-128 GiB RAM
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
# Tier M (300 warehouses / ~29 GiB schema) — 32-512 GiB RAM
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
    timeout=timedelta(minutes=60),
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
    timeout=timedelta(minutes=60),
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
    timeout=timedelta(minutes=60),
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
    timeout=timedelta(minutes=60),
)

# ---------------------------------------------------------------------------
# Tier L (1000 warehouses / ~95 GiB schema) — 128 GiB+ RAM
# ---------------------------------------------------------------------------

hammerdb_postgres_dbaas_oltp_mixed_l_async = DbaasDbTask(
    parallel=False,
    priority=1.2,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="L",
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

hammerdb_postgres_dbaas_oltp_mixed_l_durable = DbaasDbTask(
    parallel=False,
    priority=1.21,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    shirt_size="L",
    durability="durable",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

benchbase_postgres_dbaas_read_heavy_l = DbaasDbTask(
    parallel=False,
    priority=1.22,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="read_heavy",
    shirt_size="L",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

benchbase_postgres_dbaas_crud_simple_l = DbaasDbTask(
    parallel=False,
    priority=1.23,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="crud_simple",
    shirt_size="L",
    durability="async",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

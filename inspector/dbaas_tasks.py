"""DBaaS Postgres benchmark task definitions (mirror of multi-VM Postgres in tasks.py)."""

from datetime import timedelta

from lib import DbaasDbTask

# Initial rollout allowlist — expand as more managed SKUs are validated per vendor.
DBAAS_ROLLOUT = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
}

# DBaaS priority band 1 (1.0, 1.1, …). Async tasks skip provisioning on targets where
# synchronous_commit cannot be relaxed per ManagedDbTarget.sync_commit_session_settable.

hammerdb_postgres_dbaas_oltp_mixed_c100 = DbaasDbTask(
    parallel=False,
    priority=1.0,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    cache_tier="c100",
    cache_ratio=1.0,
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=60),
)

hammerdb_postgres_dbaas_oltp_mixed_c30 = DbaasDbTask(
    parallel=False,
    priority=1.1,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    cache_tier="c30",
    cache_ratio=0.3,
    durability="async",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

hammerdb_postgres_dbaas_oltp_mixed_durable_c100 = DbaasDbTask(
    parallel=False,
    priority=1.15,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="oltp_mixed",
    cache_tier="c100",
    cache_ratio=1.0,
    durability="durable",
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=60),
)

benchbase_postgres_dbaas_read_heavy_c100 = DbaasDbTask(
    parallel=False,
    priority=1.2,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="read_heavy",
    cache_tier="c100",
    cache_ratio=1.0,
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

benchbase_postgres_dbaas_crud_simple_c100 = DbaasDbTask(
    parallel=False,
    priority=1.3,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    tool="benchbase",
    workload_proxy="crud_simple",
    cache_tier="c100",
    cache_ratio=1.0,
    durability="async",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=60),
)

hammerdb_postgres_dbaas_olap_c100 = DbaasDbTask(
    parallel=False,
    priority=1.4,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="hammerdb_postgres_dbaas",
    tool="hammerdb",
    workload_proxy="olap",
    cache_tier="c100",
    cache_ratio=1.0,
    image="ghcr.io/sparecores/benchmark-hammerdb-postgres:main",
    timeout=timedelta(minutes=120),
)

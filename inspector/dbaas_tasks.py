"""DBaaS Postgres benchmark task definitions (mirror of multi-VM in tasks.py).

BenchBase Wikipedia only. Schema scales with managed-instance RAM
(~1/4, max 16 GiB). Concurrency: 1, ncpus/2, ncpus. Timed rungs: 5 minutes.

Both async (synchronous_commit=off) and durable (on) are measured.
"""

from datetime import timedelta

from lib import DbaasDbTask

DBAAS_ROLLOUT = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
}

_COMMON = dict(
    parallel=False,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    workload_proxy="read_heavy",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=90),
)

benchbase_postgres_dbaas_read_heavy_async = DbaasDbTask(
    **_COMMON,
    priority=1.0,
    durability="async",
)

benchbase_postgres_dbaas_read_heavy_durable = DbaasDbTask(
    **_COMMON,
    priority=1.01,
    durability="durable",
)

"""DBaaS Postgres benchmark task definitions (mirror of multi-VM in tasks.py).

BenchBase Wikipedia only. Schema scales with managed-instance RAM
(~1/4, max 16 GiB). Concurrency: 1, ncpus/2, ncpus. Timed rungs: 5 minutes.

Wikipedia is read-heavy → durable only for now. Async task kept commented so
DbaasDbTask(durability="async") can be re-enabled later.
"""

from datetime import timedelta

from lib import DB_DOCKER_OPTS, DbaasDbTask

# GCP DBaaS mirrors the multi-VM matrix where Cloud SQL shapes allow
# (Enterprise Plus fixed 8 GiB/vCPU on perf-optimized; see README-db.md):
#   A) more cores — N-8 (8c/64G) vs N-16 (16c/128G); RAM scales with cores
#   B) same cores, different RAM — N-8 (64G) vs memory-optimized-N-8 (256G)
#   C) cross-topology peers — N-8 ↔ n2/c2d-highmem-8; N-16 ↔ n2-highmem-16
DBAAS_ROLLOUT = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-8/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
    ("gcp", "db-memory-optimized-N-8/postgres/18/standalone"),
}

_COMMON = dict(
    parallel=False,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="benchbase_postgres_dbaas",
    workload_proxy="read_heavy",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    timeout=timedelta(minutes=90),
    # BenchBase client: many terminals need elevated nofile (see DB_DOCKER_OPTS)
    docker_opts=DB_DOCKER_OPTS,
)

# benchbase_postgres_dbaas_read_heavy_async = DbaasDbTask(
#     **_COMMON,
#     priority=1.0,
#     durability="async",
# )

benchbase_postgres_dbaas_read_heavy_durable = DbaasDbTask(
    **_COMMON,
    priority=1.01,
    durability="durable",
)

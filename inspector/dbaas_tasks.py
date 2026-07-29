"""DBaaS Postgres benchmark task definitions (mirror of multi-VM in tasks.py).

Concurrency: geometric anchors {1, V/4, V/2, V} + upward search while TPM
improves ≥5%. Warmup once, then short settle; 5 min measure windows.

Workloads (pgbench only for now):
  * RO (-S), fixed ~1 GiB — durable
  * TPC-B (tpcb-like), V-selected size ≤ cache budget — async
"""

from datetime import timedelta

from lib import DB_DOCKER_OPTS, DbaasDbTask

# GCP DBaaS mirrors the multi-VM matrix where Cloud SQL shapes allow
# (Enterprise Plus fixed 8 GiB/vCPU on perf-optimized; see README-db.md):
#   A) more cores — N-8 (8c/64G) vs N-16 (16c/128G); RAM scales with cores
#   B) same cores, different RAM — N-8 (64G) vs memory-optimized-N-8 (256G)
#   C) cross-topology peers — N-8 ↔ n2/c2d-highmem-8; N-16 ↔ n2-highmem-16
DBAAS_ROLLOUT = {
    # ("azure", "Standard_E16ds_v5/postgres/18/standalone"),
    # ("gcp", "db-perf-optimized-N-8/postgres/18/standalone"),
    # ("gcp", "db-perf-optimized-N-16/postgres/18/standalone"),
    ("gcp", "db-perf-optimized-N-128/postgres/18/standalone"),
    # ("gcp", "db-memory-optimized-N-8/postgres/18/standalone"),
}

_PGBENCH_RO = dict(
    parallel=False,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="pgbench_postgres_dbaas",
    workload_proxy="read_heavy",
    image="ghcr.io/sparecores/benchmark-pgbench-postgres:main",
    timeout=timedelta(minutes=120),
    docker_opts=DB_DOCKER_OPTS,
)

pgbench_postgres_dbaas_ro_durable = DbaasDbTask(
    **_PGBENCH_RO,
    priority=1.02,
    durability="durable",
)

_PGBENCH_TPCB = dict(
    parallel=False,
    dbaas_only=DBAAS_ROLLOUT,
    benchmark_family="pgbench_postgres_dbaas_tpcb",
    workload_proxy="write_heavy",
    image="ghcr.io/sparecores/benchmark-pgbench-postgres:main",
    timeout=timedelta(minutes=180),
    docker_opts=DB_DOCKER_OPTS,
)

pgbench_postgres_dbaas_tpcb_async = DbaasDbTask(
    **_PGBENCH_TPCB,
    priority=1.03,
    durability="async",
)

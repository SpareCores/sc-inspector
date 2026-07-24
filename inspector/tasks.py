import os
from datetime import timedelta

import parse
import psutil
import transform
from lib import DOCKER_OPTS, DB_DOCKER_OPTS, DockerTask, MultiVmDbTask, Task, VllmDockerTask

# vLLM CPU uses multiprocessing; default Docker /dev/shm (64 MiB) is too small.
VLLM_DOCKER_OPTS = DOCKER_OPTS | {
    "shm_size": 4 * 1024**3,
}

def tracker_docker_opts(job_name: str, **extra_env: str | None) -> dict:
    return dict(
        environment={
            "TRACKER_PROJECT_NAME": "inspector",
            "TRACKER_JOB_NAME": job_name,
            "TRACKER_EXTERNAL_RUN_ID": os.environ.get("GITHUB_RUN_ID"),
            "SENTINEL_API_TOKEN": os.environ.get("SENTINEL_API_TOKEN"),
            "HF_TOKEN": os.environ.get("HF_TOKEN"),
            **extra_env,
        }
    )


GPU_EXCLUDE = {
    ("aws", "g3.4xlarge"),
    ("aws", "g3.8xlarge"),
    ("aws", "g3.16xlarge"),
    ("aws", "g4dn.metal"),
    ("aws", "p2.8xlarge"),
    ("aws", "p2.xlarge"),
    ("aws", "p4d.24xlarge"),
    ("aws", "g3s.xlarge"),
    ("gcp", "a2-megagpu-16g"),
}

# get the amount of available memory
mem_bytes = psutil.virtual_memory().available


# Records timing/* checkpoints (user_data mount, machine boot, inspector); see sc-inspector-data README.
timing = Task(
    parallel=True,
    priority=0,
    command="",
    timeout=timedelta(minutes=1),
    start_with_instance=True,
    always_run=True,
)

dmidecode = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/dmidecode:main",
    parse_output=[parse.dmidecode],
    version_command="dmidecode --version",
    command="dmidecode",
    timeout=timedelta(minutes=5),
)

lshw = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lshw -version",
    command="lshw -json",
    timeout=timedelta(minutes=5),
)

lsblk = DockerTask(
    parallel=False,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -O -Jdb",
    timeout=timedelta(minutes=1),
    start_with_instance=True,
)

lsblk_discard = DockerTask(
    parallel=False,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -DJdb",
    timeout=timedelta(minutes=1),
    start_with_instance=True,
)

lsblk_topo = DockerTask(
    parallel=False,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -tJdb",
    timeout=timedelta(minutes=1),
    start_with_instance=True,
)

lscpu = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="bash -c \"lscpu --version | awk '{print $4}'\"",
    # pretty print JSON output
    command="bash -c 'lscpu -JB | jq'",
    timeout=timedelta(minutes=5),
)

lstopo = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lstopo --version",
    command="lstopo --of xml",
    parse_output=[parse.lstopo],
    start_with_instance=True,
    timeout=timedelta(seconds=30),
)

nvidia_smi = DockerTask(
    parallel=True,
    priority=0,
    # we have to adapt to the oldest supported CUDA version
    image="nvidia/cuda:11.4.3-base-ubuntu20.04",
    gpu=True,
    servers_exclude=GPU_EXCLUDE,
    version_command="bash -c \"nvidia-smi -h | head -1 | egrep -o 'v[0-9.]+'\"",
    command="nvidia-smi -q -x",
    precheck_command="lshw -C display -json | jq -r '.[].vendor'",
    precheck_regex="nvidia",
    timeout=timedelta(minutes=10),
)

virtualization = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/virtualization:main",
    command="/usr/local/bin/check_virt.sh",
    timeout=timedelta(minutes=5),
)

# Azure: keep prior validation SKUs. GCP: comparison matrix (see README-db.md):
#   A) same CPU+RAM, different cores — n2-highmem-8 (8c/64G) vs n2-standard-16 (16c/64G)
#   B) same CPU+cores, different RAM — n2-standard-8 (8c/32G) vs n2-highmem-8 (8c/64G)
#   C) same shape, Intel vs AMD     — n2-highmem-8 vs c2d-highmem-8 (both 8c/64G)
#   + n2-highmem-16 (16c/128G) peers Cloud SQL db-perf-optimized-N-16
POSTGRES_MULTI_ROLLOUT = {
    ("azure", "Standard_F16ams_v6"),
    ("azure", "Standard_E16ds_v5"),
    ("azure", "Standard_E8ds_v5"),
    ("gcp", "n2-standard-8"),
    # ("gcp", "n2-highmem-8"),
    # ("gcp", "n2-standard-16"),
    # ("gcp", "n2-highmem-16"),
    # ("gcp", "c2d-highmem-8"),
}

# ---------------------------------------------------------------------------
# Multi-VM Postgres — BenchBase Wikipedia (RAM-scaled working set)
#
# Schema ≈ min(25% RAM, 16 GiB) so it fits under pgtune shared_buffers.
# Concurrency ladder: 1, ncpus/2, ncpus. Each timed rung is 5 minutes.
# Wikipedia is read-heavy → durable only for now. Async task kept below
# (commented) so MultiVmDbTask(durability="async") can be re-enabled later.
# ---------------------------------------------------------------------------

_POSTGRES_MULTI_COMMON = dict(
    parallel=False,
    servers_only=POSTGRES_MULTI_ROLLOUT,
    benchmark_family="benchbase_postgres_multi",
    workload_proxy="read_heavy",
    image="ghcr.io/sparecores/benchmark-benchbase-postgres:main",
    # load + up to 3×(2 min warmup + 5 min run) ≈ ~40+ min; keep headroom
    timeout=timedelta(minutes=90),
    # High nofile / memlock / seccomp — see DB_DOCKER_OPTS in lib.py
    docker_opts=DB_DOCKER_OPTS,
)

# benchbase_postgres_multi_read_heavy_async = MultiVmDbTask(
#     **_POSTGRES_MULTI_COMMON,
#     priority=1.0,
#     durability="async",
# )

benchbase_postgres_multi_read_heavy_durable = MultiVmDbTask(
    **_POSTGRES_MULTI_COMMON,
    priority=1.01,
    durability="durable",
)

# We use this benchmark to determine the "SCore" of a given instance. This should represent the relative
# performance of it, which can be used to compare the "speed" of measured machines.
# After running all stress-ng `--cpu-method`s on a few selected instances (x86_64 and ARM, with and without
# HyperThreading), the `div16` method seemed to be best in terms of scalability.
# Other methods showed either lower or higher scalability than expected. For example on a 2 vCPU machine with
#     HyperThreading `div16` showed approx. 1x performance when running with `--cpu 2`, other methods showed up to 1.8x.
# Also, when running on HT machines with many cores, other methods gave either very low (like ~50x single core performance
# on a 64 core machine) or very high (like ~130x on a 64/128 core/thread machine) scalability.
stressngfull = DockerTask(
    parallel=False,
    priority=2,
    image="ghcr.io/sparecores/stress-ng:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("stressngfull"),
    version_docker_opts={},
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command=r"""-c 'for ncpu in $(awk -f /usr/local/bin/count.awk $(nproc)); do echo -n "$ncpu,"; nice -n -20 stress-ng --metrics --cpu $ncpu --cpu-method div16 -t 10 | egrep "metrc.*cpu" | awk "{print \$9}"; done'""",
    timeout=timedelta(minutes=30),
)

# Self-contained stress-ng benchmarks (no network, no disk writes). JSON output.
# Runner runs each stressor with --metrics, parses bogo ops/s, emits one JSON; failed stressors → None.
stressng_benchmarks = DockerTask(
    parallel=False,
    priority=3,
    image="ghcr.io/sparecores/stress-ng:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("stressng_benchmarks"),
    version_docker_opts={},
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command="-c 'nice -n -20 python3 /usr/local/bin/run_stressng_benchmarks.py'",
    timeout=timedelta(minutes=20),
    start_with_instance=True,
)

openssl = DockerTask(
    parallel=False,
    priority=4,
    image="ghcr.io/sparecores/benchmark:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("openssl"),
    parse_output=[parse.openssl],
    version_command="bash -c \"openssl version | awk '{print $2}'\"",
    command="openssl.sh",
    timeout=timedelta(hours=1),
)

geekbench = DockerTask(
    parallel=False,
    priority=5,
    image="ghcr.io/sparecores/benchmark:main",
    version_command="bash -c \"/usr/local/geekbench-$(uname -m)/geekbench6 --version | awk '{print $2}'\"",
    docker_opts=DOCKER_OPTS
    | tracker_docker_opts(
        "geekbench",
        BENCHMARK_SECRETS_PASSPHRASE=os.environ.get("BENCHMARK_SECRETS_PASSPHRASE"),
    )
    | dict(
        mem_limit=int(mem_bytes * 0.85),
        memswap_limit=int(mem_bytes * 0.85),
        mem_swappiness=0,
    ),
    # geekbench struggles and give truncated results with less than 2GB of memory
    minimum_memory=2.1,
    transform_output=[transform.raw, transform.fetch_geekbench_results],
    command="nice -n -20 geekbench.sh",
    timeout=timedelta(hours=1),
)

compression_text = DockerTask(
    parallel=False,
    priority=6,
    minimum_memory=1,
    timeout=timedelta(hours=1),
    image="ghcr.io/sparecores/benchmark:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("compression_text"),
    command="nice -n -20 python /usr/local/bin/compress.py",
)

membench = DockerTask(
    parallel=False,
    priority=7,
    timeout=timedelta(minutes=40),
    image="ghcr.io/sparecores/membench:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("membench"),
    # run for 30 minutes max
    command="-Hv -t 1800",
    start_with_instance=True,
    version_command="-V",
    minimum_memory=0.9,
    servers_exclude={
        ("ovh", "r3-128"),
    },
)

bw_mem = DockerTask(
    parallel=False,
    priority=8,
    timeout=timedelta(hours=1),
    image="ghcr.io/sparecores/benchmark:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("bw_mem"),
    command="bw_mem.sh",
    # These machines either crash or hang when running this benchmark
    servers_exclude={
        ("ovh", "a10-180"),
        ("ovh", "l4-360"),
    },
)

static_web = DockerTask(
    parallel=False,
    priority=9,
    image="ghcr.io/sparecores/benchmark-web:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("static_web"),
    version_command="bash -c \"(binserve --version; wrk -v) | egrep -o '(binserve|wrk) [0-9.]+'\"",
    command="nice -n -20 python /usr/local/bin/benchmark.py",
    timeout=timedelta(minutes=30),
)

redis = DockerTask(
    parallel=False,
    priority=10,
    image="ghcr.io/sparecores/benchmark-redis:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("redis"),
    version_command="redis-server -v",
    version_docker_opts=dict(entrypoint=""),
    command="nice -n -20 python /usr/local/bin/benchmark.py",
    timeout=timedelta(minutes=10),
)

nvbandwidth = DockerTask(
    parallel=False,
    priority=11,
    image="ghcr.io/sparecores/nvbandwidth:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("nvbandwidth"),
    gpu=True,
    servers_exclude=GPU_EXCLUDE,
    version_command="bash -c \"nvbandwidth --help | head -1 | egrep -o 'v[0-9.]+'\"",
    command="nvbandwidth -j",
    precheck_command="lshw -C display -json | jq -r '.[].vendor'",
    precheck_regex="nvidia",
    timeout=timedelta(minutes=30),
)

passmark = DockerTask(
    parallel=False,
    # might be slow on some machines
    timeout=timedelta(hours=1),
    priority=12,
    image="ghcr.io/sparecores/benchmark-passmark:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("passmark"),
    command=None,
)

llm = DockerTask(
    parallel=False,
    # might be slow when testing large models
    timeout=timedelta(hours=1.5),
    minimum_memory=1,
    priority=13,
    image="ghcr.io/sparecores/benchmark-llm:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("llm"),
    command=None,
    version_command="--version",
)

# Unified vLLM: probe GPU → Hub CPU → AVX2 CPU images, then GuideLLM benchmark on first probe success.
# Output under data/.../vllm/ (no fallback after benchmark starts).

vllm = VllmDockerTask(
    parallel=False,
    timeout=timedelta(hours=3),
    minimum_memory=4,
    priority=14,
    images=[
        "ghcr.io/sparecores/benchmark-vllm-gpu:main",
        "ghcr.io/sparecores/benchmark-vllm-cpu:main",
        "ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main",
    ],
    docker_opts=VLLM_DOCKER_OPTS | tracker_docker_opts("vllm"),
    command=None,
    version_command="--version",
    start_with_instance=True,
)

# An extended version of the multicore StressNg task: running
# stress-ng for an increasing number of seconds per minute, then
# sleeping until the start of the next minute, repeated 1440 times,
# so running for a full day.
# The load is increased from 5 seconds per minute (in the 1st hour)
# to 55 seconds per minute (from the 11th hour) linearly.
stressnglongrun = DockerTask(
    servers_only={
        ("aws", "t4g.medium"),
        ("aws", "c7g.large"),
        ("gcp", "e2-medium"),
        ("gcp", "c2d-highcpu-2"),
        ("hcloud", "cx21"),
        ("hcloud", "cx22"),
        ("hcloud", "cax11"),
        ("hcloud", "ccx13"),
    },
    parallel=False,
    timeout=timedelta(hours=26),
    image="ghcr.io/sparecores/stress-ng-longrun:main",
    docker_opts=DOCKER_OPTS | tracker_docker_opts("stressnglongrun"),
    version_docker_opts={},
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command="-c \"nice -n -20 sh -c 'for i in $(seq 1 1440); do SPM=$(($(($i / 60 + 1)) * 5)); SPM=$(( $SPM > 55 ? 55 : $SPM )); stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t $SPM -Y /dev/stderr; sleep $((60 - $(date +%-S) )); done'\"",
    parse_output=[parse.stressnglongrun],
)

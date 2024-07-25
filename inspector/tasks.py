from datetime import timedelta
from lib import DockerTask, DOCKER_OPTS
import os
import parse
import psutil
import transform

STRESSNG_TAG = "b7c7a5877501679a3b0a67d877e6274a801d1e4e"  # V0.17.08

# get the amount of available memory
mem_bytes = psutil.virtual_memory().available


dmidecode = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/dmidecode:main",
    parse_output=[parse.dmidecode],
    version_command="dmidecode --version",
    command="dmidecode",
)

lshw = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lshw -version",
    command="lshw -json",
)

lscpu = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="bash -c \"lscpu --version | awk '{print $4}'\"",
    # pretty print JSON output
    command="bash -c 'lscpu -JB | jq'",
)

nvidia_smi = DockerTask(
    parallel=True,
    priority=0,
    image="nvidia/cuda:12.4.1-base-ubuntu22.04",
    gpu=True,
    version_command="bash -c \"nvidia-smi -h | head -1 | egrep -o 'v[0-9.]+'\"",
    command="nvidia-smi -q -x",
)

# We use this benchmark to determine the "SCore" of a given instance. This should represent the relative
# performance of it, which can be used to compare the "speed" of measured machines.
# After running all stress-ng `--cpu-method`s on a few selected instances (x86_64 and ARM, with and without
# HyperThreading), the `div16` method seemed to be best in terms of scalability.
# Other methods showed either lower or higher scalability than expected. For example on a 2 vCPU machine with
#     HyperThreading `div16` showed approx. 1x performance when running with `--cpu 2`, other methods showed up to 1.8x.
# Also, when running on HT machines with many cores, other methods gave either very low (like ~50x single core performance
# on a 64 core machine) or very high (like ~130x on a 64/128 core/thread machine) scalability.
stressng = DockerTask(
    parallel=False,
    priority=1,
    image=f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}",
    docker_opts=DOCKER_OPTS | dict(entrypoint="sh"),
    version_docker_opts=dict(entrypoint="sh"),
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command="-c \"nice -n -20 stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t 20 -Y /dev/stderr\"",
)

stressngsinglecore = DockerTask(
    parallel=False,
    priority=2,
    image=f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}",
    docker_opts=DOCKER_OPTS | dict(entrypoint="sh"),
    version_docker_opts=dict(entrypoint="sh"),
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command="-c \"nice -n -20 stress-ng --metrics --cpu 1 --cpu-method div16 -t 20 -Y /dev/stderr\"",
)

# An extended version of the multicore StressNg task: running
# stress-ng for an increasing number of seconds per minute, then
# sleeping until the start of the next minute, repeated 1440 times,
# so running for a full day.
# The load is increased from 5 seconds per minute (in the 1st hour)
# to 55 seconds per minute (from the 11th hour) linearly.
stressnglongrung = DockerTask(
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
    image=f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}",
    docker_opts=DOCKER_OPTS | dict(entrypoint="sh"),
    version_docker_opts=dict(entrypoint="sh"),
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command="-c \"nice -n -20 sh -c 'for i in $(seq 1 1440); do SPM=$(($(($i / 60 + 1)) * 5)); SPM=$(( $SPM > 55 ? 55 : $SPM )); stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t $SPM -Y /dev/stderr; sleep $((60 - $(date +%-S) )); done'\"",
    parse_output=[parse.stressnglongrun],
)

openssl = DockerTask(
    parallel=False,
    priority=3,
    image="ghcr.io/sparecores/benchmark:main",
    parse_output=[parse.openssl],
    version_command="bash -c \"openssl version | awk '{print $2}'\"",
    command="openssl.sh",
)

geekbench = DockerTask(
    parallel=False,
    priority=4,
    image="ghcr.io/sparecores/benchmark:main",
    version_command="bash -c \"/usr/local/geekbench-$(uname -m)/geekbench6 --version | awk '{print $2}'\"",
    docker_opts=DOCKER_OPTS | dict(
        environment={"BENCHMARK_SECRETS_PASSPHRASE": os.environ.get("BENCHMARK_SECRETS_PASSPHRASE")},
        mem_limit=int(mem_bytes * 0.85),
        memswap_limit=int(mem_bytes * 0.85),
        mem_swappiness=0,
    ),
    minimum_memory=1.3,
    transform_output=[transform.raw, transform.fetch_geekbench_results],
    command="nice -n -20 geekbench.sh",
)

compression_text = DockerTask(
    parallel=False,
    priority=5,
    image="ghcr.io/sparecores/benchmark:main",
    command="nice -n -20 python /usr/local/bin/compress.py"
)

bw_mem = DockerTask(
    parallel=False,
    priority=6,
    image="ghcr.io/sparecores/benchmark:main",
    command="bw_mem.sh",
    docker_opts=DOCKER_OPTS | dict(
        mem_limit=int(mem_bytes * 0.85),
        memswap_limit=int(mem_bytes * 0.85),
        mem_swappiness=0,
    ),
)

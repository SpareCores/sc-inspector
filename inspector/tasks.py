import os
from datetime import timedelta

import parse
import psutil
import transform
from lib import DOCKER_OPTS, DockerTask

STRESSNG_TAG = "b7c7a5877501679a3b0a67d877e6274a801d1e4e"  # V0.17.08
GPU_EXCLUDE = {
    ("aws", "g3.16xlarge"),
    ("aws", "g4dn.metal"),
    ("aws", "p2.8xlarge"),
    ("aws", "p2.xlarge"),
    ("aws", "p4d.24xlarge"),
    ("aws", "g6f.2xlarge"),
    ("aws", "g6f.4xlarge"),
    ("aws", "g6f.xlarge"),
    ("aws", "gr6f.4xlarge"),
    ("gcp", "a2-megagpu-16g"),
}

class DynamicServerSet:
    """
    A set-like object that allows dynamic inclusion of all instances for a specific vendor
    while maintaining a static list for other vendors.
    
    Compatible with tuple-based lookups: (vendor, instance_type) in set
    """
    def __init__(self, static_servers, enable_vendors=None):
        """
        :param static_servers: Set of (vendor, instance_type) tuples for static inclusion
        :param enable_vendors: Set of vendor names that should match all instances
        """
        self.static_servers = static_servers
        self.enable_vendors = enable_vendors or set()
    
    def __contains__(self, item):
        """
        Check if (vendor, instance_type) is in the set.
        Returns True for all instances of enabled vendors, or if in static_servers.
        """
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        vendor, instance = item
        # If vendor is in enable_vendors, allow all instances
        if vendor in self.enable_vendors:
            return True
        # Otherwise, check static list
        return item in self.static_servers
    
    def __bool__(self):
        """Return True if there are any static servers or enabled vendors"""
        return bool(self.static_servers) or bool(self.enable_vendors)
    
    def __iter__(self):
        """Iterate over static servers"""
        return iter(self.static_servers)
    
    def __len__(self):
        """Return length of static servers (for compatibility)"""
        return len(self.static_servers)


RUN_NEW_TASKS_ON_SERVERS = DynamicServerSet(
    static_servers={
        # ("aws", "r8a.48xlarge"),
        # ("aws", "r8g.48xlarge"),
        ("aws", "r8a.4xlarge"),
        # ("aws", "t3.nano"),
        # ("aws", "t3a.nano"),
        # ("aws", "i7ie.48xlarge"),
        # ("aws", "i7ie.metal-48xl"),
        # ("azure", "M416s_9_v2"),
        # ("gcp", "c3d-highcpu-360"),
        # AWS x8i instances
        ("aws", "x8i.12xlarge"),
        ("aws", "x8i.16xlarge"),
        ("aws", "x8i.24xlarge"),
        ("aws", "x8i.2xlarge"),
        ("aws", "x8i.32xlarge"),
        ("aws", "x8i.48xlarge"),
        ("aws", "x8i.4xlarge"),
        ("aws", "x8i.64xlarge"),
        ("aws", "x8i.8xlarge"),
        ("aws", "x8i.96xlarge"),
        ("aws", "x8i.large"),
        ("aws", "x8i.metal-48xl"),
        ("aws", "x8i.metal-96xl"),
        ("aws", "x8i.xlarge"),
        # AWS x2i* instances
        ("aws", "x2idn.16xlarge"),
        ("aws", "x2idn.24xlarge"),
        ("aws", "x2idn.32xlarge"),
        ("aws", "x2idn.metal"),
        ("aws", "x2iedn.16xlarge"),
        ("aws", "x2iedn.24xlarge"),
        ("aws", "x2iedn.2xlarge"),
        ("aws", "x2iedn.32xlarge"),
        ("aws", "x2iedn.4xlarge"),
        ("aws", "x2iedn.8xlarge"),
        ("aws", "x2iedn.metal"),
        ("aws", "x2iedn.xlarge"),
        ("aws", "x2iezn.12xlarge"),
        ("aws", "x2iezn.2xlarge"),
        ("aws", "x2iezn.4xlarge"),
        ("aws", "x2iezn.6xlarge"),
        ("aws", "x2iezn.8xlarge"),
        ("aws", "x2iezn.metal"),
        # AWS GPU instances (excluding those in GPU_EXCLUDE)
        ("aws", "dl1.24xlarge"),
        ("aws", "g3.4xlarge"),
        ("aws", "g3.8xlarge"),
        ("aws", "g3s.xlarge"),
        ("aws", "g4ad.16xlarge"),
        ("aws", "g4ad.2xlarge"),
        ("aws", "g4ad.4xlarge"),
        ("aws", "g4ad.8xlarge"),
        ("aws", "g4ad.xlarge"),
        ("aws", "g4dn.12xlarge"),
        ("aws", "g4dn.16xlarge"),
        ("aws", "g4dn.2xlarge"),
        ("aws", "g4dn.4xlarge"),
        ("aws", "g4dn.8xlarge"),
        ("aws", "g4dn.xlarge"),
        ("aws", "g5.12xlarge"),
        ("aws", "g5.16xlarge"),
        ("aws", "g5.24xlarge"),
        ("aws", "g5.2xlarge"),
        ("aws", "g5.48xlarge"),
        ("aws", "g5.4xlarge"),
        ("aws", "g5.8xlarge"),
        ("aws", "g5.xlarge"),
        ("aws", "g5g.16xlarge"),
        ("aws", "g5g.2xlarge"),
        ("aws", "g5g.4xlarge"),
        ("aws", "g5g.8xlarge"),
        ("aws", "g5g.metal"),
        ("aws", "g5g.xlarge"),
        ("aws", "g6.12xlarge"),
        ("aws", "g6.16xlarge"),
        ("aws", "g6.24xlarge"),
        ("aws", "g6.2xlarge"),
        ("aws", "g6.48xlarge"),
        ("aws", "g6.4xlarge"),
        ("aws", "g6.8xlarge"),
        ("aws", "g6.xlarge"),
        ("aws", "g6e.12xlarge"),
        ("aws", "g6e.16xlarge"),
        ("aws", "g6e.24xlarge"),
        ("aws", "g6e.2xlarge"),
        ("aws", "g6e.48xlarge"),
        ("aws", "g6e.4xlarge"),
        ("aws", "g6e.8xlarge"),
        ("aws", "g6e.xlarge"),
        ("aws", "g7e.12xlarge"),
        ("aws", "g7e.24xlarge"),
        ("aws", "g7e.2xlarge"),
        ("aws", "g7e.48xlarge"),
        ("aws", "g7e.4xlarge"),
        ("aws", "g7e.8xlarge"),
        ("aws", "gr6.4xlarge"),
        ("aws", "gr6.8xlarge"),
        ("aws", "p2.16xlarge"),
        ("aws", "p3.16xlarge"),
        ("aws", "p3.2xlarge"),
        ("aws", "p3.8xlarge"),
        ("aws", "p3dn.24xlarge"),
        ("aws", "p4de.24xlarge"),
        ("aws", "p5.48xlarge"),
        ("aws", "p5.4xlarge"),
        ("aws", "p5e.48xlarge"),
        ("aws", "p5en.48xlarge"),
        ("aws", "p6-b200.48xlarge"),
        ("aws", "p6-b300.48xlarge"),
    },
    enable_vendors={"alicloud"},
)

# get the amount of available memory
mem_bytes = psutil.virtual_memory().available


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
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -Jdb",
    timeout=timedelta(minutes=1),
    servers_only=RUN_NEW_TASKS_ON_SERVERS,
)

lsblk_discard = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -DJdb",
    timeout=timedelta(minutes=1),
    servers_only=RUN_NEW_TASKS_ON_SERVERS,
)

lsblk_topo = DockerTask(
    parallel=True,
    priority=0,
    image="ghcr.io/sparecores/hwinfo:main",
    version_command="lsblk -V",
    command="lsblk -tJdb",
    timeout=timedelta(minutes=1),
    servers_only=RUN_NEW_TASKS_ON_SERVERS,
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
    servers_only=RUN_NEW_TASKS_ON_SERVERS,
    timeout=timedelta(seconds=30),
)

nvidia_smi = DockerTask(
    parallel=True,
    priority=0,
    image="nvidia/cuda:12.4.1-base-ubuntu22.04",
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
    priority=1,
    image="ghcr.io/sparecores/stress-ng:main",
    docker_opts=DOCKER_OPTS | dict(entrypoint="sh"),
    version_docker_opts=dict(entrypoint="sh"),
    version_command="-c \"stress-ng --version | awk '{print $3}'\"",
    command=r"""-c 'for ncpu in $(awk -f /usr/local/bin/count.awk $(nproc)); do echo -n "$ncpu,"; nice -n -20 stress-ng --metrics --cpu $ncpu --cpu-method div16 -t 10 | egrep "metrc.*cpu" | awk "{print \$9}"; done'""",
    timeout=timedelta(minutes=30),
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
    timeout=timedelta(hours=1),
)

geekbench = DockerTask(
    parallel=False,
    priority=4,
    image="ghcr.io/sparecores/benchmark:main",
    version_command="bash -c \"/usr/local/geekbench-$(uname -m)/geekbench6 --version | awk '{print $2}'\"",
    docker_opts=DOCKER_OPTS
    | dict(
        environment={
            "BENCHMARK_SECRETS_PASSPHRASE": os.environ.get(
                "BENCHMARK_SECRETS_PASSPHRASE"
            )
        },
        mem_limit=int(mem_bytes * 0.85),
        memswap_limit=int(mem_bytes * 0.85),
        mem_swappiness=0,
    ),
    minimum_memory=1.1,
    transform_output=[transform.raw, transform.fetch_geekbench_results],
    command="nice -n -20 geekbench.sh",
    timeout=timedelta(hours=1),
)

compression_text = DockerTask(
    parallel=False,
    priority=5,
    minimum_memory=1,
    timeout=timedelta(hours=1),
    image="ghcr.io/sparecores/benchmark:main",
    command="nice -n -20 python /usr/local/bin/compress.py",
)

bw_mem = DockerTask(
    parallel=False,
    priority=6,
    timeout=timedelta(hours=1),
    image="ghcr.io/sparecores/benchmark:main",
    command="bw_mem.sh",
    # These machines either crash or hang when running this benchmark
    servers_exclude={
        ("ovh", "a10-180"),
        ("ovh", "l4-360"),
    },
)

static_web = DockerTask(
    parallel=False,
    priority=7,
    image="ghcr.io/sparecores/benchmark-web:main",
    version_command="bash -c \"(binserve --version; wrk -v) | egrep -o '(binserve|wrk) [0-9.]+'\"",
    command="nice -n -20 python /usr/local/bin/benchmark.py",
    timeout=timedelta(minutes=30),
)

redis = DockerTask(
    parallel=False,
    priority=8,
    image="ghcr.io/sparecores/benchmark-redis:main",
    version_command="redis-server -v",
    command="nice -n -20 python /usr/local/bin/benchmark.py",
    timeout=timedelta(minutes=10),
)

nvbandwidth = DockerTask(
    parallel=False,
    priority=9,
    image="ghcr.io/sparecores/nvbandwidth:main",
    gpu=True,
    servers_exclude=GPU_EXCLUDE,
    version_command="bash -c \"nvbandwidth --help | head -1 | egrep -o 'v[0-9.]+'\"",
    command="nvbandwidth -j",
    precheck_command="lshw -C display -json | jq -r '.[].vendor'",
    precheck_regex="nvidia",
    timeout=timedelta(minutes=15),
)

passmark = DockerTask(
    parallel=False,
    # might be slow on some machines
    timeout=timedelta(hours=1),
    priority=10,
    image="ghcr.io/sparecores/benchmark-passmark:main",
    command=None,
)

llm = DockerTask(
    parallel=False,
    # might be slow when testing large models
    timeout=timedelta(hours=1.5),
    minimum_memory=1,
    priority=11,
    image="ghcr.io/sparecores/benchmark-llm:main",
    command=None,
    version_command="--version",
)

membench = DockerTask(
    parallel=False,
    priority=12,
    timeout=timedelta(minutes=40),
    image="ghcr.io/sparecores/membench:main",
    # run for 30 minutes max
    command="-Hv -t 1800",
    servers_only=RUN_NEW_TASKS_ON_SERVERS,
    version_command="-V",
    minimum_memory=0.9,
)

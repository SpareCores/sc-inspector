from typing import Callable
import lib
import os
import parse
import transform

STRESSNG_TAG = "b7c7a5877501679a3b0a67d877e6274a801d1e4e"  # V0.17.08

mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class DmiDecode(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/dmidecode:main"
    parse_output: list = [parse.dmidecode]
    version_command: str = "dmidecode --version"
    command: str = "dmidecode"


class Lshw(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/hwinfo:main"
    version_command: str = "lshw -version"
    command: str = "lshw -json"


class Lscpu(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/hwinfo:main"
    version_command: str = "bash -c \"lscpu --version | awk '{print $4}'\""
    # pretty print JSON output
    command: str = "bash -c 'lscpu -JB | jq'"


class Nvidia_Smi(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "nvidia/cuda:12.4.1-base-ubuntu22.04"
    gpu: bool = True
    version_command: str = "bash -c \"nvidia-smi -h | head -1 | egrep -o 'v[0-9.]+'\""
    command: str = "nvidia-smi -q -x"


class StressNg(lib.DockerTask):
    """
    We use this benchmark to determine the "SCore" of a given instance. This should represent the relative
    performance of it, which can be used to compare the "speed" of measured machines.
    After running all stress-ng `--cpu-method`s on a few selected instances (x86_64 and ARM, with and without
    HyperThreading), the `div16` method seemed to be best in terms of scalability.
    Other methods showed either lower or higher scalability than expected. For example on a 2 vCPU machine with
    HyperThreading `div16` showed approx. 1x performance when running with `--cpu 2`, other methods showed up to 1.8x.
    Also, when running on HT machines with many cores, other methods gave either very low (like ~50x single core performance
    on a 64 core machine) or very high (like ~130x on a 64/128 core/thread machine) scalability.
    """
    parallel: bool = False
    priority: int = 1
    image: str = f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}"
    docker_opts: dict = lib.DOCKER_OPTS | dict(entrypoint="sh")
    version_docker_opts: dict = dict(entrypoint="sh")
    version_command: str = "-c \"stress-ng --version | awk '{print $3}'\""
    command: str = "-c \"nice -n -20 stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t 20 -Y /dev/stderr\""


class StressNgSingleCore(lib.DockerTask):
    parallel: bool = False
    priority: int = 2
    image: str = f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}"
    docker_opts: dict = lib.DOCKER_OPTS | dict(entrypoint="sh")
    version_docker_opts: dict = dict(entrypoint="sh")
    version_command: str = "-c \"stress-ng --version | awk '{print $3}'\""
    command: str = "-c \"nice -n -20 stress-ng --metrics --cpu 1 --cpu-method div16 -t 20 -Y /dev/stderr\""


class StressNgLongRun(lib.DockerTask):
    parallel: bool = False
    image: str = f"ghcr.io/colinianking/stress-ng:{STRESSNG_TAG}"
    docker_opts: dict = lib.DOCKER_OPTS | dict(entrypoint="sh")
    version_docker_opts: dict = dict(entrypoint="sh")
    version_command: str = "-c \"stress-ng --version | awk '{print $3}'\""
    command: str = '-c "nice -n -20 for _ in {1..1440}; do stress-ng --metrics --cpu $(nproc) --cpu-method div16 -t 55 -Y /dev/stderr && sleep $((60 - $(date +%-S.%N) )); done"'


class Openssl(lib.DockerTask):
    parallel: bool = False
    priority: int = 3
    image: str = "ghcr.io/sparecores/benchmark:main"
    parse_output: list = [parse.openssl]
    version_command: str = "bash -c \"openssl version | awk '{print $2}'\""
    command: str = "openssl.sh"


class Geekbench(lib.DockerTask):
    parallel: bool = False
    priority: int = 4
    image: str = "ghcr.io/sparecores/benchmark:main"
    version_command: str = "bash -c \"/usr/local/geekbench-$(uname -m)/geekbench6 --version | awk '{print $2}'\""
    docker_opts: dict = lib.DOCKER_OPTS | dict(
        environment={"BENCHMARK_SECRETS_PASSPHRASE": os.environ.get("BENCHMARK_SECRETS_PASSPHRASE")},
        mem_limit=int(mem_bytes * 0.85),
    )
    minimum_memory: float = 1.3
    transform_output: list[Callable] = [transform.raw, transform.fetch_geekbench_results]
    command: str = "nice -n -20 geekbench.sh"


class Compression_Text(lib.DockerTask):
    parallel: bool = False
    priority: int = 5
    image: str = "ghcr.io/sparecores/benchmark:main"
    # try to protect the inspector from OOM situations
    docker_opts: dict = lib.DOCKER_OPTS | dict(mem_limit=int(mem_bytes * 0.85))
    command: str = "nice -n -20 python /usr/local/bin/compress.py"


class Bw_mem(lib.DockerTask):
    parallel: bool = False
    priority: int = 6
    image: str = "ghcr.io/sparecores/benchmark:main"
    command: str = "bw_mem.sh"

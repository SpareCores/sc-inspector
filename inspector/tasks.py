from typing import Callable
import lib
import os
import parse
import transform


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
    command: str = "nvidia-smi -q -x"


class Compression_Text(lib.DockerTask):
    parallel: bool = False
    priority: int = 1
    image: str = "ghcr.io/sparecores/benchmark:main"
    command: str = "python /usr/local/bin/compress.py"


class Openssl(lib.DockerTask):
    parallel: bool = False
    priority: int = 2
    image: str = "ghcr.io/sparecores/benchmark:main"
    parse_output: list = [parse.openssl]
    version_command: str = "bash -c \"openssl version | awk '{print $2}'\""
    command: str = "openssl.sh"


class Bw_mem(lib.DockerTask):
    parallel: bool = False
    priority: int = 3
    image: str = "ghcr.io/sparecores/benchmark:main"
    command: str = "bw_mem.sh"


class Geekbench(lib.DockerTask):
    parallel: bool = False
    priority: int = 4
    image: str = "ghcr.io/sparecores/benchmark:main"
    version_command: str = "bash -c \"/usr/local/geekbench-$(uname -m)/geekbench6 --version | awk '{print $2}'\""
    docker_opts: dict = lib.DOCKER_OPTS | dict(environment={"BENCHMARK_SECRETS_PASSPHRASE": os.environ.get("BENCHMARK_SECRETS_PASSPHRASE")})
    transform_output: list[Callable] = [transform.raw, transform.fetch_geekbench_results]
    command: str = "geekbench.sh"


class StressNg(lib.DockerTask):
    parallel: bool = False
    priority: int = 5
    image: str = "ghcr.io/colinianking/stress-ng:b7c7a5877501679a3b0a67d877e6274a801d1e4e"  # V0.17.08
    docker_opts: dict = lib.DOCKER_OPTS | dict(entrypoint="sh")
    version_docker_opts: dict = dict(entrypoint="sh")
    version_command: str = "-c \"stress-ng version | awk '{print $3}'\""
    command: str = "-c \"stress-ng --metrics --cpu $(nproc) --cpu-method crc16 -t 10 -Y /dev/stderr\""
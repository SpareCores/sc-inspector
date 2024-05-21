import lib
import parse


class DmiDecode(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/dmidecode:main"
    parse_output: list = [parse.dmidecode]
    command: str = "dmidecode"


class Lshw(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/hwinfo:main"
    command: str = "lshw -json"


class Lscpu(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/hwinfo:main"
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
    command: str = "bash -c 'cd /usr/local/turbobench; ./turbobench -v0 -eALL dickens | ansi2txt'"
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

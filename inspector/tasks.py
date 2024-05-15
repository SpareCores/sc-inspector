import lib
import parse


class DmiDecode(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "ghcr.io/sparecores/dmidecode:main"
    command: str = "dmidecode"

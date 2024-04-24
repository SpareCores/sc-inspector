import lib


class DmiDecode(lib.DockerTask):
    parallel: bool = True
    priority: int = 0
    image: str = "shuuji3/dmidecode"
    command: str = "dmidecode"

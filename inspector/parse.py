from dmiparser import DmiParser
import csv
import copy
import dateparser
import fastnumbers
import json
import os
import re


def si_prefixes(binary=False):
    si_base = 1000
    if binary:
        si_base = 1024
    si_prefix = {
        "Q": si_base ** 10,
        "Qi": 1024 ** 10,
        "R": si_base ** 9,
        "Ri": 1024 ** 9,
        "Y": si_base ** 8,
        "Yi": 1024 ** 8,
        "Z": si_base ** 7,
        "Zi": 1024 ** 7,
        "E": si_base ** 6,
        "Ei": 1024 ** 6,
        "P": si_base ** 5,
        "Pi": 1024 ** 5,
        "T": si_base ** 4,
        "Ti": 1024 ** 4,
        "G": si_base ** 3,
        "Gi": 1024 ** 3,
        "M": si_base ** 2,
        "Mi": 1024 ** 2,
        "k": si_base,
        "ki": 1024,
        "K": si_base,  # non-standard
        "Ki": 1024,  # non-standard
        "h": 100,
        "da": 10,
        "": 1,
        "d": 10 ** -1,
        "c": 10 ** -2,
        "m": 10 ** -3,
        "μ": 10 ** -6,
        "n": 10 ** -9,
        "p": 10 ** -12,
        "f": 10 ** -15,
        "a": 10 ** -18,
        "z": 10 ** -21,
        "y": 10 ** -24,
        "r": 10 ** -27,
        "q": 10 ** -30,
    }
    return si_prefix


UNITS = [
    "bits",
    "Hz",
    "B",  # bytes
    "V",  # Volts
    "T/s",  # transactions/sec
    "s",  # seconds
]


def get_multipliers(binary=False):
    res = {}
    for unit in UNITS:
        si_prefix = si_prefixes()
        if unit == "B" and binary:
            # use binary multipliers for bytes if asked to (so kB == kiB)
            si_prefix = si_prefixes(binary)
        for prefix, multiplier in si_prefix.items():
            res[f"{prefix}{unit}"] = (multiplier, unit)
    return res


def computer_readable(input_str: str, binary: bool = False):
    input_num = fastnumbers.fast_real(input_str)
    if type(input_num) in (int, float):
        return input_num, None

    # Regular expression to match the numeric value and the unit
    match = re.match(r"([0-9.]+)\s+([a-zA-Z/]+)$", input_str)
    if not match:
        dt = dateparser.parse(
            input_str,
            settings={
                "STRICT_PARSING": True,
                # exclude relative-time as returns current date for e.g. "0000:00:1c.4"
                "PARSERS": [
                    "timestamp",
                    "custom-formats",
                    "absolute-time",
                ],
            },
        )
        if dt:
            return dt.isoformat(), None
        return input_str, None

    value, unit_with_prefix = match.groups()
    value = float(value)

    multipliers = get_multipliers(binary)
    multiplier, unit = multipliers.get(unit_with_prefix, (None, None))
    if not multiplier:
        return input_str, None

    numeric_value = float(value) * multiplier
    return fastnumbers.fast_real(numeric_value), unit


DMIDECODE_NO_PARSE = {
    "ID",
}


# dmidecode functions
def dmi_propvals(data):
    data_copy = copy.deepcopy(data)
    for k, v in data["props"].items():
        if len(v["values"]) == 1:
            if k in DMIDECODE_NO_PARSE:
                # don't try to convert these fields to numbers/dates, they are either useless, or parsed
                # incorrectly as dates
                data_copy["props"][k] = v["values"][0]
                unit = None
            else:
                # dmidecode uses SI prefixes for binary values (kB == kiB), so we treat them as the same
                data_copy["props"][k], unit = computer_readable(v["values"][0], binary=True)
            if unit is not None:
                data_copy["props"][f"{k}:unit"] = unit
        else:
            data_copy["props"][k] = v["values"]
    return data_copy


def dmidecode(meta, task, task_dir) -> None:
    output = open(os.path.join(task_dir, "stdout"), "r").read()
    dmi = json.loads(str(DmiParser(output)))
    parsed_output = []
    for d in dmi:
        parsed_output.append(dmi_propvals(d))
    with open(os.path.join(task_dir, "parsed.json"), "w") as f:
        json.dump(parsed_output, f, indent=2)


def openssl(meta, task, task_dir) -> None:
    """
    Parses openssl speed -mr output from multiple, separate (one algo per run) runs, with a separator of
    `ALGO: (optional -evp option) algo_name ----`
    """
    data = open(os.path.join(task_dir, "stdout"), "r").read()
    lines = data.splitlines()
    parsed_output = []

    for line in lines:
        if m := re.search(r"^ALGO:( -[^\s]+ | )([^\s]+) --------------", line):
            # algo start
            # ALGO: -evp blake2b512 ----
            # ALGO: X448 ----
            algo = m.group(2)
            # new test, (re)set reused variables
            blocksizes = []

        if m := re.search(r"^\+DT:[^\s:]+:[0-9]+:[0-9]+", line):
            algo = m.group(0).split(":")[1]
        if m := re.search(r"\+H(:[0-9]+)+", line):
            # tells us the block sizes used for this test
            # Single core example (no -multi):
            #   +H:16:64:256:1024
            # Multi core example (-multi, even with -multi 1)
            #   Got: +H:16:64:256:1024:8192:16384 from 15
            blocksizes = list(map(int, m.group(0).split(":")[1:]))
        if m := re.search(r"^\+F:[0-9]+:" + algo + "(:[0-9.]+)+$", line):
            res = m.group(0).split(":")
            algo = res[2]
            speed = res[3:]
            for i in range(len(blocksizes)):
                parsed_output.append(dict(algo=algo, block_size=blocksizes[i], speed=float(speed[i])))
    with open(os.path.join(task_dir, "parsed.json"), "w") as f:
        json.dump(parsed_output, f, indent=2)


class YamlLike(str):
    def extract(self, key):
        return re.search(f"{key}: ([0-9:\\.]*)\\n", self).group(1)

    def slice(self):
        return [YamlLike(s) for s in self.split("...\n---")]


def stressnglongrun(meta, task, task_dir) -> None:
    output = open(os.path.join(task_dir, "stderr"), "r").read()
    chunks = YamlLike(output).slice()
    with open(os.path.join(task_dir, "parsed.csv"), "w", newline="") as f:
        csvfile = csv.writer(f)
        for chunk in chunks:
            date = chunk.extract("date-yyyy-mm-dd").replace(":", "-")
            time = chunk.extract(r"time-hh-mm-ss")
            value = chunk.extract("bogo-ops-per-second-real-time")
            duration = chunk.extract("wall-clock-time")
            csvfile.writerow([date + "T" + time, duration, value])


def parse_wrk(output):
    data = {}

    # Regular expressions to extract the necessary information
    size_re = re.compile(r'@ http://localhost:8080/(\d+k)')
    threads_connections_re = re.compile(r'(\d+) threads and (\d+) connections')
    latency_re = re.compile(r'Latency\s+([\d.]+[a-z]+)\s+([\d.]+[a-z]+)\s+([\d.]+[a-z]+)')
    reqs_re = re.compile(r'Requests/sec:\s+([\d.]+)')
    transfer_re = re.compile(r'Transfer/sec:\s+([\d.]+)([A-Z]+)')

    # Unit multipliers
    unit_multipliers = {
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "k": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    def convert_to_seconds(value):
        """Convert time values to seconds."""
        return float(value[:-2]) * unit_multipliers[value[-2:]]

    def convert_to_bytes(value, unit):
        """Convert human-readable bandwidth to bytes."""
        return float(value) * unit_multipliers[unit]

    for section in output.split("Running"):
        if not section.strip():
            continue

        size_match = size_re.search(section)
        if not size_match:
            continue
        size = size_match.group(1)

        threads_connections_match = threads_connections_re.search(section)
        latency_match = latency_re.search(section)
        reqs_match = reqs_re.search(section)
        transfer_match = transfer_re.search(section)

        if not (threads_connections_match and latency_match and reqs_match and transfer_match):
            continue

        threads = int(threads_connections_match.group(1))
        connections = int(threads_connections_match.group(2))
        lat_avg = convert_to_seconds(latency_match.group(1))
        lat_stdev = convert_to_seconds(latency_match.group(2))
        lat_max = convert_to_seconds(latency_match.group(3))
        rps = float(reqs_match.group(1))
        bandwidth = convert_to_bytes(transfer_match.group(1), transfer_match.group(2))

        # Build the dictionary entry
        entry = {
            "threads": threads,
            "connections": connections,
            "rps": rps,
            "bandwidth": bandwidth,
            "lat_avg": lat_avg,
            "lat_stdev": lat_stdev,
            "lat_max": lat_max,
        }

        if size not in data:
            data[size] = []

        data[size].append(entry)

    return data


def wrk(meta, task, task_dir) -> None:
    output = open(os.path.join(task_dir, "stdout"), "r").read()
    parsed_output = parse_wrk(output)
    with open(os.path.join(task_dir, "parsed.json"), "w") as f:
        json.dump(parsed_output, f, indent=2)

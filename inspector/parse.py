from dmiparser import DmiParser
import csv
import copy
import dateparser
import fastnumbers
import json
import os
import re
import subprocess


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


def lstopo(meta, task, task_dir) -> None:
    """
    Convert lstopo XML output to SVG format.
    Reads XML from stdout, converts using lstopo, and writes SVG to task_dir.
    Only regenerates SVG if it's missing or older than the stdout file.
    """
    stdout_path = os.path.join(task_dir, "stdout")
    svg_output_path = os.path.join(task_dir, "lstopo.svg")
    
    # Check if we need to regenerate the SVG
    need_regenerate = False
    if not os.path.exists(svg_output_path):
        need_regenerate = True
    elif os.path.exists(stdout_path):
        # Regenerate if SVG is older than stdout
        if os.path.getmtime(svg_output_path) < os.path.getmtime(stdout_path):
            need_regenerate = True
    
    if not need_regenerate:
        return
    
    # Convert XML to SVG using lstopo, reading directly from stdout file
    result = subprocess.run(
        ["lstopo", "-i", stdout_path, "--if", "xml", "--of", "svg", "--no-legend"],
        capture_output=True,
        text=True,
        check=True,
    )
    
    # Write SVG output to task_dir
    with open(svg_output_path, "w") as f:
        f.write(result.stdout)
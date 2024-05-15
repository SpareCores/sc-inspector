from dmiparser import DmiParser
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
    match = re.match(r"([0-9.]+)\s*([a-zA-Z/]+)$", input_str)
    if not match:
        dt = dateparser.parse(input_str, settings={'STRICT_PARSING': True})
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


# dmidecode functions
def dmi_propvals(data):
    data_copy = copy.deepcopy(data)
    for k, v in data["props"].items():
        if len(v["values"]) == 1:
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
        json.dump(parsed_output, f)
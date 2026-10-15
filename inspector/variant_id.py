"""Hardware variant identity from lscpu -JB output.

Identical copies live in sc-inspector (inspector/variant_id.py, used on the VM)
and sc-scratch (attila/instance_variants/, used by the backfill); change both.
Matching recomputes the id from stored raw lscpu fields; the stored ``id``
is for humans and grep only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

ID_VERSION = 1

# Fields stored raw in variant.json for later id_version bumps.
RECORDED_LSCPU_FIELDS = (
    "Architecture:",
    "Vendor ID:",
    "CPU family:",
    "Model:",
    "Stepping:",
    "Model name:",
    "Flags:",
    "L1d cache:",
    "L1i cache:",
    "L2 cache:",
    "L3 cache:",
    "Address sizes:",
    "CPU(s):",
    "Socket(s):",
    "NUMA node(s):",
)

_TRADEMARK_RE = re.compile(r"\(R\)|®|™|©", re.IGNORECASE)
_AT_GHZ_RE = re.compile(r"\s*@\s*[\d.]+\s*GHz\s*$", re.IGNORECASE)
_N_CORE_PROC_RE = re.compile(r"\s+\d+-Core\s+Processor\s*$", re.IGNORECASE)
_KVM_PAREN_RE = re.compile(
    r"\((?:(?:no\s+)?TSX|IBRS)(?:\s+(?:(?:no\s+)?TSX|IBRS))*\)",
    re.IGNORECASE,
)
_LEADING_WORDS = (
    "intel",
    "amd",
    "hygon",
    "aws",
    "xeon",
    "epyc",
    "core",
    "platinum",
    "gold",
    "silver",
    "bronze",
    "cpu",
    "processor",
)
_DIGIT_RE = re.compile(r"\d")


def _norm_part(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).casefold()


def parse_lscpu_stdout(raw: Any) -> dict[str, str]:
    """Parse lscpu -JB JSON (dict with ``lscpu`` list or the list itself)."""
    if isinstance(raw, (str, bytes, bytearray)):
        raw = json.loads(raw)
    rows = raw
    if isinstance(raw, Mapping):
        rows = raw.get("lscpu", raw)
    if not isinstance(rows, list):
        raise ValueError("lscpu stdout is not a list of field/data rows")
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        field = row.get("field")
        data = row.get("data")
        if field is None:
            continue
        out[str(field)] = "" if data is None else str(data)
    return out


def load_lscpu_file(path: Path | str) -> dict[str, str]:
    return parse_lscpu_stdout(Path(path).read_text(encoding="utf-8"))


def normalize_x86_sku(model_name: str) -> str:
    """Normalize x86 Model name: into a SKU token (empty if no digit remains)."""
    s = _TRADEMARK_RE.sub("", model_name or "")
    s = " ".join(s.split())
    s = _AT_GHZ_RE.sub("", s)
    s = _N_CORE_PROC_RE.sub("", s)
    # Drop KVM parentheticals whose tokens are only IBRS / TSX / no TSX.
    while True:
        nxt = _KVM_PAREN_RE.sub("", s)
        nxt = " ".join(nxt.split())
        if nxt == s:
            break
        s = nxt
    s = s.casefold()
    tokens = s.split()
    # Strip vendor/line words from both ends so "... 8175m cpu" → "8175m".
    while tokens and tokens[0] in _LEADING_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1] in _LEADING_WORDS:
        tokens.pop()
    sku = " ".join(tokens)
    if not _DIGIT_RE.search(sku):
        return ""
    return sku


def variant_id_from_fields(fields: Mapping[str, str]) -> str:
    """Compute the canonical variant id from a field->data map."""
    arch = _norm_part(fields.get("Architecture:"))
    vendor = _norm_part(fields.get("Vendor ID:"))
    model = _norm_part(fields.get("Model:"))
    stepping = _norm_part(fields.get("Stepping:"))
    if arch == "x86_64":
        family = _norm_part(fields.get("CPU family:"))
        sku = normalize_x86_sku(fields.get("Model name:", "") or "")
        return f"{arch}|{vendor}|{family}|{model}|{stepping}|{sku}"
    # aarch64 and any other arch: no family (ARM lscpu has none today).
    model_name = _norm_part(fields.get("Model name:"))
    return f"{arch}|{vendor}|{model_name}|{model}|{stepping}"


def variant_id_from_lscpu_stdout(raw: Any) -> str:
    return variant_id_from_fields(parse_lscpu_stdout(raw))


def build_variant_json(
    fields: Mapping[str, str],
    *,
    source: str,
    first_seen: Optional[str] = None,
    lscpu_version: Optional[str] = None,
) -> dict[str, Any]:
    """Build a variant.json object from raw lscpu fields."""
    if source not in ("backfill", "observed"):
        raise ValueError(f"invalid source: {source!r}")
    recorded = {k: fields[k] for k in RECORDED_LSCPU_FIELDS if k in fields}
    # Always include id-driving fields even if empty/missing in source.
    for key in (
        "Architecture:",
        "Vendor ID:",
        "CPU family:",
        "Model:",
        "Stepping:",
        "Model name:",
    ):
        recorded.setdefault(key, fields.get(key, ""))
    return {
        "id_version": ID_VERSION,
        "id": variant_id_from_fields(fields),
        "source": source,
        "first_seen": first_seen,
        "lscpu_version": lscpu_version,
        "lscpu": recorded,
    }


def match_variant(
    machine_fields: Mapping[str, str],
    stored_variant: Mapping[str, Any],
) -> bool:
    """True if recomputed id from stored lscpu equals this machine's id."""
    stored_lscpu = stored_variant.get("lscpu")
    if not isinstance(stored_lscpu, Mapping):
        return False
    return variant_id_from_fields(machine_fields) == variant_id_from_fields(
        {str(k): "" if v is None else str(v) for k, v in stored_lscpu.items()}
    )


def load_variant_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_variant_json(path: Path | str, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2) + "\n", encoding="utf-8")

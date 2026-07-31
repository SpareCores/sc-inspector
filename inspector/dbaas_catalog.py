"""Managed database catalog for DBaaS benchmarks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

ManagedDbCatalog: TypeAlias = dict[
    tuple[str, str],
    tuple["ManagedDbTarget", list[str], list[str], dict[str, str | None]],
]

_STATIC_CATALOG_PATH = Path(__file__).with_name("dbaas_catalog_static.json")

# Default deployable-region subset (PoC). Set DBAAS_GCP_REGIONS=all to use every
# region listed for a GCP target in the static JSON; or a comma-separated list.
_GCP_POC_REGIONS = ("us-central1", "us-east1", "europe-west1")


@dataclass(frozen=True)
class ManagedDbTarget:
    """Catalog row for a managed database SKU (analogous to sc-data Server)."""

    vendor_id: str
    engine: str
    native_id: str
    sku_id: str
    engine_version: str
    ha_mode: str
    cpu_count: float
    memory_gib: float
    edition: str | None = None
    sync_commit_session_settable: bool | None = None

    @property
    def instance_key(self) -> str:
        return f"{self.native_id}/postgres/{self.engine_version}/{self.ha_mode}"


def _gcp_region_allowlist() -> set[str] | None:
    raw = os.environ.get("DBAAS_GCP_REGIONS", "").strip()
    if not raw:
        return set(_GCP_POC_REGIONS)
    if raw.lower() == "all":
        return None
    return {r.strip() for r in raw.split(",") if r.strip()}


@lru_cache(maxsize=1)
def _load_static_catalog_doc() -> dict:
    return json.loads(_STATIC_CATALOG_PATH.read_text())


def _zones_for_regions(
    vendor_id: str,
    regions: list[str],
    zones_by_region: dict[str, dict[str, list[str]]],
) -> list[str]:
    vendor_zones = zones_by_region.get(vendor_id) or {}
    out: list[str] = []
    for region in regions:
        out.extend(vendor_zones.get(region) or [])
    return out


def _available_managed_dbs_static(
    vendor: str | None = None,
    region: str | None = None,
    *,
    engine: str = "postgres",
) -> ManagedDbCatalog:
    doc = _load_static_catalog_doc()
    zones_by_region = doc.get("zones_by_region") or {}
    gcp_allow = _gcp_region_allowlist()
    out: ManagedDbCatalog = {}
    for row in doc.get("targets") or []:
        if engine != row.get("engine"):
            continue
        if vendor and row.get("vendor_id") != vendor:
            continue
        target = ManagedDbTarget(
            vendor_id=row["vendor_id"],
            engine=row["engine"],
            native_id=row["native_id"],
            sku_id=row["sku_id"],
            engine_version=str(row["engine_version"]),
            ha_mode=row.get("ha_mode") or "standalone",
            cpu_count=float(row["cpu_count"]),
            memory_gib=float(row["memory_gib"]),
            edition=row.get("edition"),
            sync_commit_session_settable=row.get("sync_commit_session_settable"),
        )
        regions = list(row.get("regions") or [])
        if target.vendor_id == "gcp" and gcp_allow is not None:
            regions = [r for r in regions if r in gcp_allow]
        if region:
            regions = [r for r in regions if r == region]
        zones = _zones_for_regions(target.vendor_id, regions, zones_by_region)
        if region and zones:
            zones = [z for z in zones if z.rsplit("-", 1)[0] == region]
        zone_to_region = {z: z.rsplit("-", 1)[0] for z in zones}
        if not regions and not zones:
            continue
        if not regions and zones:
            regions = sorted({zone_to_region[z] for z in zones})
        out[(target.vendor_id, target.instance_key)] = (
            target,
            regions,
            zones,
            zone_to_region,
        )
    return out


def _available_managed_dbs_sc_data(
    vendor: str | None = None,
    region: str | None = None,
    *,
    engine: str = "postgres",
) -> ManagedDbCatalog:
    raise NotImplementedError("sc-data db_instance catalog not yet available")


def available_managed_dbs(
    vendor: str | None = None,
    region: str | None = None,
    *,
    engine: str = "postgres",
) -> ManagedDbCatalog:
    """Return ACTIVE managed DB targets with deployable regions/zones."""
    if os.environ.get("DBAAS_CATALOG_SOURCE", "static") == "static":
        return _available_managed_dbs_static(vendor, region, engine=engine)
    return _available_managed_dbs_sc_data(vendor, region, engine=engine)

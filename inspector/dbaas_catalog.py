"""Managed database catalog for DBaaS benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeAlias

ManagedDbCatalog: TypeAlias = dict[
    tuple[str, str],
    tuple["ManagedDbTarget", list[str], list[str], dict[str, str | None]],
]


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


STATIC_MANAGED_DB_TARGETS: tuple[ManagedDbTarget, ...] = (
    ManagedDbTarget(
        vendor_id="azure",
        engine="postgres",
        native_id="Standard_E16ds_v5",
        sku_id="Standard_E16ds_v5:MemoryOptimized:18",
        engine_version="18",
        ha_mode="standalone",
        cpu_count=16,
        memory_gib=128,
        edition="MemoryOptimized",
        sync_commit_session_settable=False,
    ),
)

STATIC_TARGET_REGIONS: dict[tuple[str, str], list[str]] = {
    ("azure", "Standard_E16ds_v5/postgres/18/standalone"): [
        "westeurope",
        "northeurope",
        "centralus",
        "westus2",
    ],
}


def _available_managed_dbs_static(
    vendor: str | None = None,
    region: str | None = None,
    *,
    engine: str = "postgres",
) -> ManagedDbCatalog:
    out: ManagedDbCatalog = {}
    for target in STATIC_MANAGED_DB_TARGETS:
        if engine != target.engine:
            continue
        if vendor and target.vendor_id != vendor:
            continue
        key = (target.vendor_id, target.instance_key)
        regions = list(STATIC_TARGET_REGIONS.get(key, []))
        if region:
            regions = [r for r in regions if r == region]
        if not regions:
            continue
        zones: list[str] = []
        zone_to_region: dict[str, str | None] = {}
        out[key] = (target, regions, zones, zone_to_region)
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

"""Managed database catalog for DBaaS benchmarks (sparecores-data ``database*`` tables)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

ManagedDbCatalog: TypeAlias = dict[
    tuple[str, str],
    tuple["ManagedDbTarget", list[str], list[str], dict[str, str | None]],
]

# Inspector / Pulumi ha_mode for the non-HA deploy path (matches existing data dirs).
_STANDALONE_HA_MODE = "standalone"

# Price-row HA values that map to standalone provisioning.
# AWS marks Single-AZ as SINGLE_ZONE (NONE rows are INACTIVE); Azure/GCP use NONE.
_STANDALONE_PRICE_HA: dict[str, str] = {
    "aws": "SINGLE_ZONE",
    "azure": "NONE",
    "gcp": "NONE",
}

_SUPPORTED_VENDORS = frozenset(_STANDALONE_PRICE_HA)

# Optional region allowlists (comma-separated api_reference, or "all"). Empty = all.
_REGION_ALLOWLIST_ENV = {
    "aws": "DBAAS_AWS_REGIONS",
    "azure": "DBAAS_AZURE_REGIONS",
    "gcp": "DBAAS_GCP_REGIONS",
}


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


def standalone_price_ha(vendor_id: str) -> str:
    """Return the ``database_price.ha`` value used for standalone deploys."""
    try:
        return _STANDALONE_PRICE_HA[vendor_id]
    except KeyError as exc:
        raise ValueError(f"unsupported DBaaS vendor: {vendor_id}") from exc


def _region_allowlist(vendor_id: str) -> set[str] | None:
    env_name = _REGION_ALLOWLIST_ENV.get(vendor_id)
    if not env_name:
        return None
    raw = os.environ.get(env_name, "").strip()
    if not raw or raw.lower() == "all":
        return None
    return {r.strip() for r in raw.split(",") if r.strip()}


def _preferred_engine_version(versions: list[str]) -> str | None:
    """Pick engine major version.

    Prefer ``DBAAS_ENGINE_VERSION`` when set and listed; otherwise prefer ``18``
    (matches prior static catalog / data dirs); else the latest numeric version.
    """
    if not versions:
        return None
    preferred = os.environ.get("DBAAS_ENGINE_VERSION", "").strip() or "18"
    if preferred in versions:
        return preferred

    def _sort_key(v: str) -> tuple[int, ...]:
        parts: list[int] = []
        for p in str(v).split("."):
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(-1)
        return tuple(parts) if parts else (-1,)

    return max(versions, key=_sort_key)


def _parse_json_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        return parsed if isinstance(parsed, list) else [parsed]
    return list(raw)


def _sku_id(vendor_id: str, native_id: str, edition: str | None, engine_version: str) -> str:
    if vendor_id == "azure":
        edition_part = edition or "GeneralPurpose"
        return f"{native_id}:{edition_part}:{engine_version}"
    if vendor_id == "gcp":
        return f"{native_id}:POSTGRES_{engine_version}"
    return f"{native_id}:{engine_version}"


def _edition_for_row(vendor_id: str, family: str | None) -> str | None:
    if vendor_id == "azure":
        return family or "GeneralPurpose"
    if vendor_id == "gcp":
        if family and (
            "perf-optimized" in family
            or "memory-optimized" in family
            or family.startswith("c4a")
        ):
            return "PerformanceOptimized"
        return "Enterprise"
    return family


@lru_cache(maxsize=1)
def _db_engine():
    from sqlmodel import create_engine

    import sc_data

    return create_engine(f"sqlite:///{sc_data.db.path}")


def available_managed_dbs(
    vendor: str | None = None,
    region: str | None = None,
    *,
    engine: str = "postgres",
) -> ManagedDbCatalog:
    """Return ACTIVE managed DB targets with deployable regions/zones.

    Regions come from ACTIVE ONDEMAND ``database_price`` rows for the standalone
    HA price key; zones are all ACTIVE zones in those regions (GCP needs zones).
    Ordering of regions/zones is left to callers (cheapest-first via sc-runner data).
    """
    if engine not in ("postgres", "postgresql", "POSTGRESQL"):
        return {}

    from sqlalchemy import text

    vendors = [vendor] if vendor else sorted(_SUPPORTED_VENDORS)
    out: ManagedDbCatalog = {}

    with _db_engine().connect() as conn:
        for vendor_id in vendors:
            if vendor_id not in _SUPPORTED_VENDORS:
                continue
            price_ha = standalone_price_ha(vendor_id)
            allow = _region_allowlist(vendor_id)
            # One row per database SKU with region list ordered by min price.
            stmt = text(
                """
                SELECT
                    d.database_id,
                    d.api_reference,
                    d.family,
                    d.vcpus,
                    d.memory_amount,
                    d.engine_versions,
                    r.api_reference AS region_api,
                    MIN(dp.price) AS min_price
                FROM database AS d
                JOIN database_price AS dp
                  ON dp.vendor_id = d.vendor_id
                 AND dp.database_id = d.database_id
                JOIN region AS r
                  ON r.vendor_id = dp.vendor_id
                 AND r.region_id = dp.region_id
                WHERE d.vendor_id = :vendor_id
                  AND d.status = 'ACTIVE'
                  AND UPPER(CAST(d.engine AS TEXT)) IN ('POSTGRESQL', 'POSTGRES')
                  AND d.vcpus IS NOT NULL
                  AND d.memory_amount IS NOT NULL
                  AND dp.status = 'ACTIVE'
                  AND dp.allocation = 'ONDEMAND'
                  AND dp.ha = :price_ha
                  AND r.status = 'ACTIVE'
                GROUP BY
                    d.database_id,
                    d.api_reference,
                    d.family,
                    d.vcpus,
                    d.memory_amount,
                    d.engine_versions,
                    r.api_reference
                ORDER BY d.database_id, min_price, r.api_reference
                """
            )
            rows = conn.execute(
                stmt, {"vendor_id": vendor_id, "price_ha": price_ha}
            ).mappings().all()
            if not rows:
                logging.warning(
                    "No ACTIVE standalone database_price rows for vendor=%s ha=%s",
                    vendor_id,
                    price_ha,
                )
                continue

            by_db: dict[str, dict] = {}
            for row in rows:
                region_api = row["region_api"]
                if allow is not None and region_api not in allow:
                    continue
                if region and region_api != region:
                    continue
                db_id = row["database_id"]
                entry = by_db.get(db_id)
                if entry is None:
                    versions = [str(v) for v in _parse_json_list(row["engine_versions"])]
                    eng_ver = _preferred_engine_version(versions)
                    if not eng_ver:
                        continue
                    native_id = row["api_reference"] or db_id
                    edition = _edition_for_row(vendor_id, row["family"])
                    target = ManagedDbTarget(
                        vendor_id=vendor_id,
                        engine="postgres",
                        native_id=native_id,
                        sku_id=_sku_id(vendor_id, native_id, edition, eng_ver),
                        engine_version=eng_ver,
                        ha_mode=_STANDALONE_HA_MODE,
                        cpu_count=float(row["vcpus"]),
                        memory_gib=float(row["memory_amount"]) / 1024.0,
                        edition=edition,
                        sync_commit_session_settable=True,
                    )
                    by_db[db_id] = {
                        "target": target,
                        "regions": [],
                        "seen_regions": set(),
                    }
                    entry = by_db[db_id]
                if region_api not in entry["seen_regions"]:
                    entry["seen_regions"].add(region_api)
                    entry["regions"].append(region_api)

            if not by_db:
                continue

            # Zones for the collected regions (api_reference).
            region_list = sorted({r for e in by_db.values() for r in e["regions"]})
            zones_by_region: dict[str, list[str]] = {r: [] for r in region_list}
            if region_list:
                from sqlalchemy import bindparam

                zone_stmt = text(
                    """
                    SELECT r.api_reference AS region_api, z.api_reference AS zone_api
                    FROM zone AS z
                    JOIN region AS r
                      ON r.vendor_id = z.vendor_id
                     AND r.region_id = z.region_id
                    WHERE z.vendor_id = :vendor_id
                      AND z.status = 'ACTIVE'
                      AND r.status = 'ACTIVE'
                      AND r.api_reference IN :regions
                    ORDER BY r.api_reference, z.api_reference
                    """
                ).bindparams(bindparam("regions", expanding=True))
                for zrow in conn.execute(
                    zone_stmt, {"vendor_id": vendor_id, "regions": region_list}
                ).mappings():
                    zones_by_region.setdefault(zrow["region_api"], []).append(
                        zrow["zone_api"]
                    )

            for entry in by_db.values():
                target: ManagedDbTarget = entry["target"]
                regions: list[str] = entry["regions"]
                zones: list[str] = []
                zone_to_region: dict[str, str | None] = {}
                for reg in regions:
                    for z in zones_by_region.get(reg) or []:
                        zones.append(z)
                        zone_to_region[z] = reg
                # GCP needs zones; skip SKUs with priced regions but no zones.
                if vendor_id == "gcp" and not zones:
                    logging.warning(
                        "Skipping %s/%s: no ACTIVE zones for priced regions",
                        vendor_id,
                        target.native_id,
                    )
                    continue
                if not regions:
                    continue
                out[(target.vendor_id, target.instance_key)] = (
                    target,
                    regions,
                    zones,
                    zone_to_region,
                )

    return out

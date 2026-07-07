"""Managed DB provision sizing from cache-tier profiles."""

from __future__ import annotations

import math
from typing import Any

from benchmark_tiers import BUFFER_FRAC
from dbaas_catalog import ManagedDbTarget

# Azure ManagedDiskV2 P-tier coverage for PoC storage sizes.
_AZURE_IOPS_TIERS: list[tuple[int, str]] = [
    (32, "P4"),
    (64, "P6"),
    (128, "P10"),
    (256, "P15"),
    (512, "P20"),
    (1024, "P30"),
    (2048, "P40"),
    (4096, "P50"),
    (8192, "P60"),
    (16384, "P70"),
    (32768, "P80"),
]

_P_TIER_ORDER: tuple[str, ...] = (
    "P1",
    "P2",
    "P3",
    "P4",
    "P6",
    "P10",
    "P15",
    "P20",
    "P30",
    "P40",
    "P50",
    "P60",
    "P70",
    "P80",
)

# Minimum performance tier per cache profile (can exceed size baseline via PremiumV2).
_CACHE_TIER_MIN_IOPS: dict[str, str] = {
    "c100": "P30",
    "c30": "P20",
}


def _max_iops_tier(a: str, b: str) -> str:
    order = {tier: idx for idx, tier in enumerate(_P_TIER_ORDER)}
    return a if order.get(a, -1) >= order.get(b, -1) else b


def _disk_gib_for_cache_tier(mem_gib: float, cache_ratio: float) -> int:
    schema_gib = (BUFFER_FRAC * mem_gib) / max(cache_ratio, 0.05)
    return int(max(mem_gib, math.ceil(schema_gib * 2 / 0.85)))


def _iops_tier_for_gib(storage_gib: int) -> str:
    for limit, tier in _AZURE_IOPS_TIERS:
        if storage_gib <= limit:
            return tier
    return "P80"


def provision_spec(target: ManagedDbTarget, cache_tier: str) -> dict[str, Any]:
    """Return provision parameters for a managed DB at the given cache tier."""
    cache_ratio = 1.0 if cache_tier == "c100" else 0.3
    storage_gib = _disk_gib_for_cache_tier(target.memory_gib, cache_ratio)
    iops_tier = _iops_tier_for_gib(storage_gib)
    min_tier = _CACHE_TIER_MIN_IOPS.get(cache_tier)
    if min_tier:
        iops_tier = _max_iops_tier(iops_tier, min_tier)
    edition = target.edition or "GeneralPurpose"
    sku_name, _, _ = target.sku_id.partition(":")
    return {
        "storage_gib": storage_gib,
        "storage_edition": "ManagedDiskV2",
        "storage_type": "PremiumV2_LRS",
        "iops_tier": iops_tier,
        "cache_tier": cache_tier,
        "cache_ratio": cache_ratio,
        "sku_name": sku_name,
        "sku_tier": edition,
        "schema_gib": (BUFFER_FRAC * target.memory_gib) / cache_ratio,
        "disk_gib_required": storage_gib,
    }

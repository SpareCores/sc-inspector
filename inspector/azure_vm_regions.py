"""Quota-aware region order for Azure VM (inspector server) starts.

Azure VM starts tried regions strictly cheapest-first and let Azure reject the
create when the region had no quota or did not offer the size, typically
walking ~18 regions per SKU (1-2 minutes each). Here regions are split into:

- eligible: the size is offered to the subscription (no Location-type
  restriction) and the region has free capacity for the VM's vCPUs in both its
  family quota and the Total Regional vCPUs quota, plus one free public IP
  (Basic SKU and overall: sc-runner creates a Dynamic, i.e. Basic, IP);
- everything else.

Eligible regions come first in a weighted random order (weight: how many such
VMs still fit, capped), so parallel starts spread over every region with quota
instead of all piling into the cheapest one and saturating its quota and public
IPs. The remaining regions follow in price order as a fallback, in case the
quota data is stale or wrong.

Any lookup failure falls back to the plain price order. Set
``AZURE_SKIP_VM_QUOTA_CHECK=1`` to disable the whole thing.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from azure_dbaas_quota import (
    ARM_BASE,
    ComputeSkuInfo,
    QuotaEntry,
    _arm_get,
    _parse_quota_entries,
    _quota_lookup,
    _region_compute_sku_index,
    _subscription_id,
    _vm_quotas,
    sku_family_quota_name,
)

NETWORK_USAGES_API = "2024-05-01"
# Network usage keys of the public IP caps a VM start consumes one unit of.
PUBLIC_IP_QUOTAS = ("IPv4BasicSkuPublicIpAddresses", "PublicIPAddresses")
REGIONAL_CORES = "cores"

logger = logging.getLogger(__name__)


def _disabled() -> bool:
    return os.environ.get("AZURE_SKIP_VM_QUOTA_CHECK", "").lower() in ("1", "true", "yes")


@lru_cache(maxsize=64)
def _network_quotas(subscription_id: str, region: str) -> dict[str, QuotaEntry]:
    url = (
        f"{ARM_BASE}/subscriptions/{subscription_id}/providers/Microsoft.Network/"
        f"locations/{region}/usages"
    )
    return _parse_quota_entries(_arm_get(url, params={"api-version": NETWORK_USAGES_API}))


def location_restricted(sku_info: ComputeSkuInfo, region: str) -> bool:
    """True when the size is unavailable in the whole region for this subscription.

    Zone-type restrictions only block some availability zones; sc-runner
    deploys without a zone, so those sizes are still usable in the region.
    """
    for restriction in sku_info.restrictions:
        if restriction.get("type") != "Location":
            continue
        info = restriction.get("restrictionInfo") or {}
        locations = [str(loc).lower() for loc in info.get("locations") or []]
        if not locations or region.lower() in locations:
            return True
    return False


@dataclass(frozen=True)
class RegionCheck:
    region: str
    ok: bool
    reason: str = ""
    fits: int = 0  # how many more such VMs the family quota could take


def _free(quotas: dict[str, QuotaEntry], region: str, key: str) -> int | None:
    """Free units for ``key``; None if Azure returned no such entry."""
    entry = _quota_lookup(quotas, key)
    if entry is None:
        return None
    return entry.available


def check_region(
    region: str,
    *,
    sku: str,
    vcpus: int,
    family: str | None,
    sku_index: dict[str, ComputeSkuInfo],
    vm_quotas: dict[str, QuotaEntry],
    network_quotas: dict[str, QuotaEntry],
) -> RegionCheck:
    """Evaluate one region against cached Azure usage and SKU data."""
    sku_info = sku_index.get(sku)
    if sku_info is None:
        return RegionCheck(region, False, "size not offered")
    if location_restricted(sku_info, region):
        return RegionCheck(region, False, "size not available for subscription")
    family = family or sku_info.family
    cores = _free(vm_quotas, region, REGIONAL_CORES)
    if cores is None or cores < vcpus:
        return RegionCheck(region, False, f"regional vCPUs: need {vcpus}, free {cores}")
    fits = cores // vcpus
    if family:
        fam_free = _free(vm_quotas, region, family)
        if fam_free is not None:
            if fam_free < vcpus:
                return RegionCheck(region, False, f"{family}: need {vcpus}, free {fam_free}")
            fits = min(fits, fam_free // vcpus)
    for key in PUBLIC_IP_QUOTAS:
        ip_free = _free(network_quotas, region, key)
        if ip_free is not None and ip_free < 1:
            return RegionCheck(region, False, f"{key}: none free")
    return RegionCheck(region, True, fits=max(fits, 1))


# Cap on a region's weight: a huge regional cap (e.g. 22k vCPUs) should not make
# one region win almost every draw when a family quota entry is missing.
MAX_WEIGHT = 20


def weighted_shuffle(checks: list[RegionCheck], rng: Any = random) -> list[str]:
    """Order eligible regions randomly; the more such VMs still fit, the likelier first.

    Efraimidis-Spirakis weighted sampling without replacement (key = u ** (1/w)).
    """
    keyed = [
        (rng.random() ** (1.0 / min(max(check.fits, 1), MAX_WEIGHT)), check.region)
        for check in checks
    ]
    return [region for _key, region in sorted(keyed, reverse=True)]


def order_regions(
    sku: str,
    vcpus: int,
    price_ordered: list[str],
    *,
    rng: Any = None,
    subscription_id: str | None = None,
) -> tuple[list[str], str | None]:
    """Return (region order, quota family) for an Azure VM start.

    ``price_ordered`` is the cheapest-first candidate list (the fallback order).
    """
    if _disabled() or not price_ordered:
        return list(price_ordered), None
    sub = subscription_id or _subscription_id()
    if not sub:
        logger.warning("Azure VM region quota check skipped: subscription id not configured")
        return list(price_ordered), None
    try:
        family = sku_family_quota_name(sku)
        data = {
            region: (
                _region_compute_sku_index(sub, region),
                _vm_quotas(sub, region),
                _network_quotas(sub, region),
            )
            for region in price_ordered
        }
    except Exception as exc:
        logger.warning("Azure VM region quota check failed for %s, using price order: %s", sku, exc)
        return list(price_ordered), None

    checks = [
        check_region(
            region,
            sku=sku,
            vcpus=vcpus,
            family=family,
            sku_index=data[region][0],
            vm_quotas=data[region][1],
            network_quotas=data[region][2],
        )
        for region in price_ordered
    ]
    eligible = [c for c in checks if c.ok]
    rest = [c.region for c in checks if not c.ok]
    ordered = weighted_shuffle(eligible, rng or random) + rest
    logger.info(
        "Azure quota-aware region order for %s (%d vCPU, %s): eligible %s; fallback %d region(s)",
        sku,
        vcpus,
        family,
        [f"{c.region}(fits {c.fits})" for c in eligible],
        len(rest),
    )
    for check in checks:
        if not check.ok:
            logger.debug("Azure region %s skipped for %s: %s", check.region, sku, check.reason)
    return ordered, family

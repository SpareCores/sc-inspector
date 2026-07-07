"""Pre-flight Azure quota and SKU availability checks for DBaaS stacks."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import requests

ARM_BASE = "https://management.azure.com"
PG_USAGES_API = "2025-06-01-preview"
PG_CAPABILITIES_API = "2025-06-01-preview"
COMPUTE_USAGES_API = "2024-11-01"
COMPUTE_SKUS_API = "2021-07-01"


@dataclass(frozen=True)
class QuotaEntry:
    name: str
    current: int
    limit: int

    @property
    def available(self) -> int:
        return max(0, self.limit - self.current)


@dataclass(frozen=True)
class ComputeSkuInfo:
    family: str | None
    restrictions: list[dict[str, Any]]


def _skip_quota_check() -> bool:
    return os.environ.get("DBAAS_SKIP_QUOTA_CHECK", "").lower() in ("1", "true", "yes")


def _subscription_id() -> str | None:
    return os.environ.get("ARM_SUBSCRIPTION_ID") or os.environ.get("AZURE_SUBSCRIPTION_ID")


@cache
def _catalog_engine():
    from sqlmodel import create_engine

    import sc_data

    return create_engine(f"sqlite:///{sc_data.db.path}")


@lru_cache(maxsize=512)
def _catalog_server_family(vendor: str, api_reference: str) -> str | None:
    """Server.family from sc-data (derived from Azure Resource SKUs at crawl time)."""
    from sc_crawler.tables import Server
    from sqlmodel import Session, select

    with Session(_catalog_engine()) as session:
        return session.exec(
            select(Server.family)
            .where(Server.vendor_id == vendor)
            .where(Server.api_reference == api_reference)
        ).first()


def _quota_family_from_catalog_family(catalog_family: str) -> str:
    """Reverse sc-crawler family normalization back to Azure quota usage keys."""
    if catalog_family.startswith("Standard"):
        return f"{catalog_family}Family"
    return f"standard{catalog_family}Family"


@lru_cache(maxsize=1)
def _credential():
    from azure.identity import ClientSecretCredential, DefaultAzureCredential

    client_id = os.getenv("ARM_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
    client_secret = os.getenv("ARM_CLIENT_SECRET") or os.getenv("AZURE_CLIENT_SECRET")
    tenant_id = os.getenv("ARM_TENANT_ID") or os.getenv("AZURE_TENANT_ID")
    if client_id and client_secret and tenant_id:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
        )
    return DefaultAzureCredential()


def _arm_get(url: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
    token = _credential().get_token("https://management.azure.com/.default")
    response = requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {token.token}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _parse_quota_entries(payload: dict[str, Any]) -> dict[str, QuotaEntry]:
    out: dict[str, QuotaEntry] = {}
    for item in payload.get("value", []):
        name = item.get("name") or {}
        key = name.get("value") or name.get("localizedValue")
        if not key:
            continue
        out[key] = QuotaEntry(
            name=key,
            current=int(item.get("currentValue", 0)),
            limit=int(item.get("limit", 0)),
        )
    return out


def _quota_lookup(quotas: dict[str, QuotaEntry], family: str) -> QuotaEntry | None:
    needle = family.lower()
    for key, entry in quotas.items():
        if key.lower() == needle:
            return entry
    return None


def sku_family_quota_name(
    sku: str,
    *,
    subscription_id: str | None = None,
    region: str | None = None,
    vendor: str = "azure",
) -> str | None:
    """Map an Azure VM or Flexible Server SKU name to a quota family key."""
    catalog_family = _catalog_server_family(vendor, sku)
    if catalog_family:
        return _quota_family_from_catalog_family(catalog_family)
    if subscription_id and region:
        return _compute_sku_info(subscription_id, region, sku).family
    return None


def _quota_headroom(
    quotas: dict[str, QuotaEntry],
    *,
    label: str,
    family: str | None,
    vcpus: int,
) -> tuple[bool, str]:
    cores = _quota_lookup(quotas, "cores")
    if cores is None:
        return False, f"{label}: regional cores quota missing"
    if cores.available < vcpus:
        return False, (
            f"{label}: need {vcpus} regional cores, "
            f"only {cores.available}/{cores.limit} available"
        )
    if family is None:
        logging.warning("%s: no quota family mapping for SKU", label)
        return True, ""
    family_entry = _quota_lookup(quotas, family)
    if family_entry is None:
        logging.warning("%s: quota family %s not returned by Azure", label, family)
        return True, ""
    if family_entry.available < vcpus:
        return False, (
            f"{label}: need {vcpus} {family} vCPUs, "
            f"only {family_entry.available}/{family_entry.limit} available"
        )
    return True, ""


@lru_cache(maxsize=64)
def _postgres_quotas(subscription_id: str, region: str) -> dict[str, QuotaEntry]:
    url = (
        f"{ARM_BASE}/subscriptions/{subscription_id}/providers/Microsoft.DBforPostgreSQL/"
        f"locations/{region}/resourceType/flexibleServers/usages"
    )
    return _parse_quota_entries(_arm_get(url, params={"api-version": PG_USAGES_API}))


@lru_cache(maxsize=64)
def _postgres_capabilities(subscription_id: str, region: str) -> list[dict[str, Any]]:
    url = (
        f"{ARM_BASE}/subscriptions/{subscription_id}/providers/Microsoft.DBforPostgreSQL/"
        f"locations/{region}/capabilities"
    )
    payload = _arm_get(url, params={"api-version": PG_CAPABILITIES_API})
    return payload.get("value", [])


@lru_cache(maxsize=64)
def _vm_quotas(subscription_id: str, region: str) -> dict[str, QuotaEntry]:
    url = (
        f"{ARM_BASE}/subscriptions/{subscription_id}/providers/Microsoft.Compute/"
        f"locations/{region}/usages"
    )
    return _parse_quota_entries(_arm_get(url, params={"api-version": COMPUTE_USAGES_API}))


@lru_cache(maxsize=256)
def _compute_sku_info(
    subscription_id: str,
    region: str,
    sku: str,
) -> ComputeSkuInfo:
    url = f"{ARM_BASE}/subscriptions/{subscription_id}/providers/Microsoft.Compute/skus"
    filt = f"location eq '{region}' and name eq '{sku}'"
    payload = _arm_get(
        url,
        params={"api-version": COMPUTE_SKUS_API, "$filter": filt},
    )
    family: str | None = None
    restrictions: list[dict[str, Any]] = []
    for item in payload.get("value", []):
        family = item.get("family") or family
        restrictions.extend(item.get("restrictions") or [])
    return ComputeSkuInfo(family=family, restrictions=restrictions)


def _postgres_offer_restricted(capabilities: list[dict[str, Any]]) -> bool:
    for cap in capabilities:
        for feature in cap.get("supportedFeatures") or []:
            if feature.get("name") == "OfferRestricted" and feature.get("status") == "Enabled":
                return True
        reason = cap.get("reason")
        if reason and "restricted" in reason.lower():
            return True
    return False


def _postgres_sku_listed(capabilities: list[dict[str, Any]], sku: str) -> bool:
    for cap in capabilities:
        for edition in cap.get("supportedServerEditions") or []:
            for listed in edition.get("supportedServerSkus") or []:
                if listed.get("name") == sku:
                    return True
    return False


def check_azure_postgres_region(
    region: str,
    *,
    pg_sku: str,
    pg_vcpus: int,
    subscription_id: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason) for managed Postgres in region."""
    sub = subscription_id or _subscription_id()
    if not sub:
        logging.warning("DBaaS postgres quota check skipped: subscription id not configured")
        return True, ""

    try:
        pg_caps = _postgres_capabilities(sub, region)
        if _postgres_offer_restricted(pg_caps):
            return False, f"postgres offer restricted in {region}"
        if not _postgres_sku_listed(pg_caps, pg_sku):
            return False, f"postgres SKU {pg_sku} not offered in {region}"

        pg_family = sku_family_quota_name(pg_sku, subscription_id=sub, region=region)
        return _quota_headroom(
            _postgres_quotas(sub, region),
            label="postgres",
            family=pg_family,
            vcpus=pg_vcpus,
        )
    except requests.HTTPError as exc:
        logging.warning("DBaaS postgres quota check failed for %s: %s", region, exc)
        return True, ""
    except Exception as exc:
        logging.warning("DBaaS postgres quota check failed for %s: %s", region, exc)
        return True, ""


def check_azure_client_vm_region(
    region: str,
    *,
    vm_sku: str,
    vm_vcpus: int,
    subscription_id: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason) for a benchmark client VM in region."""
    sub = subscription_id or _subscription_id()
    if not sub:
        logging.warning("DBaaS VM quota check skipped: subscription id not configured")
        return True, ""

    try:
        sku_info = _compute_sku_info(sub, region, vm_sku)
        vm_family = sku_family_quota_name(
            vm_sku,
            subscription_id=sub,
            region=region,
        ) or sku_info.family
        ok, reason = _quota_headroom(
            _vm_quotas(sub, region),
            label="client vm",
            family=vm_family,
            vcpus=vm_vcpus,
        )
        if not ok:
            return False, reason

        if sku_info.restrictions:
            codes = sorted(
                {
                    str(r.get("reasonCode") or r.get("type") or "restricted")
                    for r in sku_info.restrictions
                }
            )
            return False, f"client vm SKU {vm_sku} restricted in {region}: {', '.join(codes)}"
    except requests.HTTPError as exc:
        logging.warning("DBaaS VM quota check failed for %s/%s: %s", region, vm_sku, exc)
        return True, ""
    except Exception as exc:
        logging.warning("DBaaS VM quota check failed for %s/%s: %s", region, vm_sku, exc)
        return True, ""

    return True, ""


def check_azure_dbaas_region(
    region: str,
    *,
    pg_sku: str,
    pg_vcpus: int,
    vm_sku: str,
    vm_vcpus: int,
    subscription_id: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason) for provisioning a DBaaS stack in region."""
    sub = subscription_id or _subscription_id()
    ok, reason = check_azure_postgres_region(
        region,
        pg_sku=pg_sku,
        pg_vcpus=pg_vcpus,
        subscription_id=sub,
    )
    if not ok:
        return False, reason
    return check_azure_client_vm_region(
        region,
        vm_sku=vm_sku,
        vm_vcpus=vm_vcpus,
        subscription_id=sub,
    )


def check_dbaas_postgres_quota(
    vendor: str,
    region: str,
    pg_sku: str,
    pg_vcpus: int,
) -> tuple[bool, str]:
    if _skip_quota_check() or vendor != "azure":
        return True, ""
    return check_azure_postgres_region(
        region,
        pg_sku=pg_sku,
        pg_vcpus=pg_vcpus,
    )


def check_dbaas_vm_quota(
    vendor: str,
    region: str,
    vm_sku: str,
    vm_vcpus: int,
) -> tuple[bool, str]:
    if _skip_quota_check() or vendor != "azure":
        return True, ""
    return check_azure_client_vm_region(
        region,
        vm_sku=vm_sku,
        vm_vcpus=int(vm_vcpus),
    )


def check_dbaas_region_quota(
    vendor: str,
    region: str,
    pg_sku: str,
    pg_vcpus: int,
    vm_sku: str,
    vm_vcpus: int,
) -> tuple[bool, str]:
    """Vendor-aware wrapper: full DBaaS pair preflight."""
    if _skip_quota_check():
        return True, ""
    if vendor != "azure":
        return True, ""
    return check_azure_dbaas_region(
        region,
        pg_sku=pg_sku,
        pg_vcpus=pg_vcpus,
        vm_sku=vm_sku,
        vm_vcpus=int(vm_vcpus),
    )

"""Managed DB provision sizing from the shared ``db_storage`` plan."""

from __future__ import annotations

from typing import Any

from benchmark_tiers import target_schema_gib
from dbaas_catalog import ManagedDbTarget
from db_storage import db_storage_plan, dbaas_storage_fields


def _provision_spec_azure(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    edition = target.edition or "GeneralPurpose"
    sku_name, _, _ = target.sku_id.partition(":")
    return {
        **storage,
        "sku_name": sku_name,
        "sku_tier": edition,
        "schema_gib": schema_gib,
        "admin_login": "scadmin",
        "database_name": "bench",
    }


def _provision_spec_gcp(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    return {
        **storage,
        "sku_name": target.native_id,
        "sku_tier": target.edition or "Enterprise",
        "schema_gib": schema_gib,
        "admin_login": "scadmin",
        "database_name": "bench",
    }


def _provision_spec_aws(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    return {
        **storage,
        "sku_name": target.native_id,
        "sku_tier": target.edition or "",
        "schema_gib": schema_gib,
        "admin_login": "scadmin",
        "database_name": "bench",
    }


def _provision_spec_ovh(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    # native_id: postgresql-<plan>-<flavor>; edition is plan (essential/…).
    flavor = target.native_id
    plan = target.edition or "essential"
    if flavor.startswith("postgresql-"):
        parts = flavor.split("-")
        if len(parts) >= 3:
            plan = parts[1]
            flavor = "-".join(parts[2:])
    # OVH's "flex" DBaaS disk sizing requires the requested size (GiB) to be a
    # multiple of 10, or the create call fails with FlexDiskSizeNotMultiple.
    storage_gib = storage.get("storage_gib")
    if storage_gib:
        storage = {**storage, "storage_gib": -(-int(storage_gib) // 10) * 10}
    return {
        **storage,
        "sku_name": flavor,
        "sku_tier": plan,
        "schema_gib": schema_gib,
        "admin_login": "avnadmin",
        "database_name": "bench",
    }


def _provision_spec_upcloud(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    return {
        **storage,
        "sku_name": target.native_id,
        "sku_tier": target.edition or "",
        "schema_gib": schema_gib,
        "admin_login": "scadmin",
        "database_name": "bench",
    }


def _provision_spec_vultr(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    return {
        **storage,
        "sku_name": target.native_id,
        "sku_tier": target.edition or "",
        "schema_gib": schema_gib,
        "admin_login": "vultradmin",
        "database_name": "bench",
    }


def provision_spec(target: ManagedDbTarget) -> dict[str, Any]:
    """Return provision parameters sized from the managed instance's memory."""
    mem_gib = float(target.memory_gib or 0) or 16.0
    schema_gib = target_schema_gib(mem_gib)
    plan = db_storage_plan(
        target.vendor_id,
        mem_gib,
        vcpus=int(target.cpu_count),
        machine_type=target.native_id,
    )
    storage = dbaas_storage_fields(plan, tier=target.native_id)
    if target.vendor_id == "gcp":
        return _provision_spec_gcp(target, storage, schema_gib)
    if target.vendor_id == "aws":
        return _provision_spec_aws(target, storage, schema_gib)
    if target.vendor_id == "ovh":
        return _provision_spec_ovh(target, storage, schema_gib)
    if target.vendor_id == "upcloud":
        return _provision_spec_upcloud(target, storage, schema_gib)
    if target.vendor_id == "vultr":
        return _provision_spec_vultr(target, storage, schema_gib)
    return _provision_spec_azure(target, storage, schema_gib)

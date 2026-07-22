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
    }


def _provision_spec_gcp(target: ManagedDbTarget, storage: dict[str, Any], schema_gib: float) -> dict[str, Any]:
    return {
        **storage,
        "sku_name": target.native_id,
        "sku_tier": target.edition or "Enterprise",
        "schema_gib": schema_gib,
    }


def provision_spec(target: ManagedDbTarget) -> dict[str, Any]:
    """Return provision parameters sized from the managed instance's memory."""
    mem_gib = float(target.memory_gib or 0) or 16.0
    schema_gib = target_schema_gib(mem_gib)
    plan = db_storage_plan(target.vendor_id, mem_gib)
    storage = dbaas_storage_fields(plan)
    if target.vendor_id == "gcp":
        return _provision_spec_gcp(target, storage, schema_gib)
    return _provision_spec_azure(target, storage, schema_gib)

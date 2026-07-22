"""Path helpers for DBaaS benchmark data layout."""

from __future__ import annotations

import os
from types import SimpleNamespace

from dbaas_catalog import ManagedDbTarget


def dbaas_instance_key(target: ManagedDbTarget) -> str:
    return target.instance_key


def dbaas_data_dir(repo_path: str | os.PathLike, vendor: str, instance_key: str) -> str:
    """Return repo-relative path: dbaas/<vendor>/<instance_key>/"""
    return os.path.join(repo_path, "dbaas", vendor, instance_key)


def dbaas_sparse_path(vendor: str, instance_key: str) -> str:
    """Sparse-checkout prefix for one DBaaS instance root."""
    return f"dbaas/{vendor}/{instance_key}"


def stack_slug(target: ManagedDbTarget) -> str:
    """Short Pulumi stack slug (operational, not in git path)."""
    native = target.native_id.lower().replace("_", "")
    if native.startswith("db-perf-optimized-"):
        native = "perfopt" + native.removeprefix("db-perf-optimized-").replace("-", "")
    elif native.startswith("db-custom-"):
        native = "dbc" + native.removeprefix("db-custom-").replace("-", "")
    else:
        native = native.replace("standard", "")
    return f"{native}-pg{target.engine_version}"


def target_sizing_stub(target: ManagedDbTarget) -> SimpleNamespace:
    """Sizing object compatible with benchmark_tiers client/disk helpers."""
    return SimpleNamespace(
        vendor_id=target.vendor_id,
        vcpus=target.cpu_count,
        memory_amount=target.memory_gib * 1024,
        api_reference=target.native_id,
        gpu_count=0,
        cpu_architecture="x86_64",
    )

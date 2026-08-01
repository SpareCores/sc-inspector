"""Shared DB storage sizing for multi-VM and DBaaS (apples-to-apples I/O).

Both topologies use ``disk_profiles.provision_for_vcpus()`` to derive a
size and performance target that scales with the VM.  This ensures:
  * Small instances don't overprovision storage.
  * Large instances have enough I/O headroom that benchmarks measure
    compute, not disk bottlenecks.
  * Cross-vendor parity: all vendors get a similar MB/s-per-vCPU budget.

For size-derived types (GCP pd-ssd) the result is a larger disk; for
independently-provisioned types (Azure PremiumV2, AWS gp3) it's explicit
IOPS/throughput values.  See ``disk_profiles.py`` for the full model.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from benchmark_tiers import DISK_SCHEMA_RATIO, target_schema_gib
from disk_profiles import DiskProvision, provision_for_vcpus

# Usable fraction of the provisioned volume (OS/reserved headroom).
DISK_USABLE_FRAC = 0.85
# Floor large enough for Ubuntu + Docker images on multi-VM, and applied to
# DBaaS as well so GCP size-derived IOPS match.
MIN_STORAGE_GIB = 64

AZURE_IOPS_TIER = "P30"  # Flexible Server maps tier → IOPS/throughput

# Default disk types per vendor (used when no override is set).
DEFAULT_DISK_TYPES: dict[str, str] = {
    "gcp": "pd-ssd",
    "azure": "PremiumV2_LRS",
    "aws": "gp3",
}


@dataclass(frozen=True)
class DbStoragePlan:
    """Vendor-native storage knobs shared by multi-VM and DBaaS."""

    vendor: str
    storage_gib: int
    storage_type: str
    storage_edition: str
    iops: int | None
    throughput_mb_s: int | None
    iops_tier: str  # Azure Flexible Server tier label; empty elsewhere
    # Effective performance (informational, from disk_profiles)
    effective_write_iops: int | None
    effective_write_mbps: int | None

    def multi_vm_disk_opts(self) -> dict[str, Any]:
        """Args for ``MultiVmStackSpec`` / ``db_disk_options`` consumers."""
        opts: dict[str, Any] = {"disk_type": self.storage_type}
        if self.iops is not None:
            opts["disk_iops"] = self.iops
        if self.throughput_mb_s is not None:
            opts["disk_throughput"] = self.throughput_mb_s
        return opts


def storage_gib_for_mem(mem_gib: float) -> int:
    """Provisioned disk GiB from RAM-scaled schema, floored for OS parity."""
    need = target_schema_gib(mem_gib) * DISK_SCHEMA_RATIO
    return max(MIN_STORAGE_GIB, int(math.ceil(need / DISK_USABLE_FRAC)))


def db_storage_plan(
    vendor: str,
    mem_gib: float,
    vcpus: int = 0,
    machine_type: str | None = None,
) -> DbStoragePlan:
    """Return the shared storage plan for ``vendor`` at ``mem_gib`` / ``vcpus``.

    When ``vcpus`` > 0, uses ``disk_profiles.provision_for_vcpus()`` to ensure
    sufficient I/O headroom.  The final size is the max of the schema-derived
    minimum and the I/O-scaled provision.

    ``machine_type`` (GCE name or Cloud SQL tier) selects Hyperdisk when the
    series rejects Persistent Disk (C4/N4/…).
    """
    schema_min_gib = storage_gib_for_mem(mem_gib)
    vendor_lower = (vendor or "").lower()
    disk_type = DEFAULT_DISK_TYPES.get(vendor_lower, "")
    if vendor_lower == "gcp" and machine_type:
        try:
            from sc_runner.gcp_disks import (
                GCE_HYPERDISK_BALANCED,
                cloud_sql_requires_hyperdisk,
                gcp_requires_hyperdisk,
            )
        except ImportError:
            pass
        else:
            if gcp_requires_hyperdisk(machine_type) or cloud_sql_requires_hyperdisk(
                machine_type
            ):
                disk_type = GCE_HYPERDISK_BALANCED

    if vcpus > 0 and disk_type:
        provision = provision_for_vcpus(
            vendor_lower, disk_type, vcpus, min_size_gib=schema_min_gib
        )
        plan = _plan_from_provision(provision, vendor_lower)
    elif vendor_lower == "gcp":
        plan = _gcp_plan(schema_min_gib, disk_type=disk_type or "pd-ssd")
    elif vendor_lower == "azure":
        plan = _azure_plan(schema_min_gib, vcpus)
    elif vendor_lower == "aws":
        plan = _aws_plan(schema_min_gib, vcpus)
    else:
        plan = DbStoragePlan(
            vendor=vendor_lower,
            storage_gib=schema_min_gib,
            storage_type="",
            storage_edition="",
            iops=None,
            throughput_mb_s=None,
            iops_tier="",
            effective_write_iops=None,
            effective_write_mbps=None,
        )
    return _apply_env_overrides(plan)


def _plan_from_provision(p: DiskProvision, vendor: str) -> DbStoragePlan:
    """Convert a DiskProvision to a DbStoragePlan."""
    edition_map = {
        ("gcp", "pd-ssd"): "PD_SSD",
        ("gcp", "hyperdisk-balanced"): "HYPERDISK_BALANCED",
        ("azure", "PremiumV2_LRS"): "ManagedDiskV2",
        ("aws", "gp3"): "gp3",
    }
    edition = edition_map.get((vendor, p.disk_type), p.disk_type)
    iops_tier = AZURE_IOPS_TIER if vendor == "azure" else ""
    return DbStoragePlan(
        vendor=vendor,
        storage_gib=p.size_gib,
        storage_type=p.disk_type,
        storage_edition=edition,
        iops=p.iops,
        throughput_mb_s=p.throughput_mb_s,
        iops_tier=iops_tier,
        effective_write_iops=p.effective_write_iops,
        effective_write_mbps=p.effective_write_mbps,
    )


def _gcp_plan(storage_gib: int, disk_type: str = "pd-ssd") -> DbStoragePlan:
    edition = (
        "HYPERDISK_BALANCED" if disk_type == "hyperdisk-balanced" else "PD_SSD"
    )
    return DbStoragePlan(
        vendor="gcp",
        storage_gib=storage_gib,
        storage_type=disk_type,
        storage_edition=edition,
        iops=None,
        throughput_mb_s=None,
        iops_tier="",
        effective_write_iops=None,
        effective_write_mbps=None,
    )


def _azure_plan(storage_gib: int, vcpus: int = 0) -> DbStoragePlan:
    from disk_profiles import MIN_WRITE_IOPS, MIN_WRITE_MBPS, WRITE_IOPS_PER_VCPU, WRITE_MBPS_PER_VCPU
    iops = max(MIN_WRITE_IOPS, int(vcpus * WRITE_IOPS_PER_VCPU)) if vcpus > 0 else 5000
    mbps = max(MIN_WRITE_MBPS, int(vcpus * WRITE_MBPS_PER_VCPU)) if vcpus > 0 else 200
    return DbStoragePlan(
        vendor="azure",
        storage_gib=storage_gib,
        storage_type="PremiumV2_LRS",
        storage_edition="ManagedDiskV2",
        iops=iops,
        throughput_mb_s=mbps,
        iops_tier=AZURE_IOPS_TIER,
        effective_write_iops=iops,
        effective_write_mbps=mbps,
    )


def _aws_plan(storage_gib: int, vcpus: int = 0) -> DbStoragePlan:
    from disk_profiles import MIN_WRITE_IOPS, MIN_WRITE_MBPS, WRITE_IOPS_PER_VCPU, WRITE_MBPS_PER_VCPU
    iops = max(MIN_WRITE_IOPS, int(vcpus * WRITE_IOPS_PER_VCPU)) if vcpus > 0 else 5000
    mbps = max(MIN_WRITE_MBPS, int(vcpus * WRITE_MBPS_PER_VCPU)) if vcpus > 0 else 200
    return DbStoragePlan(
        vendor="aws",
        storage_gib=storage_gib,
        storage_type="gp3",
        storage_edition="gp3",
        iops=iops,
        throughput_mb_s=mbps,
        iops_tier="",
        effective_write_iops=iops,
        effective_write_mbps=mbps,
    )


def _apply_env_overrides(plan: DbStoragePlan) -> DbStoragePlan:
    """Optional MULTI_VM_DB_DISK_* overrides (ops escape hatch)."""
    disk_type = os.environ.get("MULTI_VM_DB_DISK_TYPE")
    disk_iops = os.environ.get("MULTI_VM_DB_DISK_IOPS")
    disk_throughput = os.environ.get("MULTI_VM_DB_DISK_THROUGHPUT")
    if disk_type is None and not disk_iops and not disk_throughput:
        return plan
    storage_type = plan.storage_type
    if disk_type is not None:
        storage_type = disk_type or plan.storage_type
    iops = int(disk_iops) if disk_iops else plan.iops
    throughput = int(disk_throughput) if disk_throughput else plan.throughput_mb_s
    return DbStoragePlan(
        vendor=plan.vendor,
        storage_gib=plan.storage_gib,
        storage_type=storage_type,
        storage_edition=plan.storage_edition,
        iops=iops,
        throughput_mb_s=throughput,
        iops_tier=plan.iops_tier,
        effective_write_iops=iops if iops else plan.effective_write_iops,
        effective_write_mbps=throughput if throughput else plan.effective_write_mbps,
    )


def dbaas_storage_fields(plan: DbStoragePlan, tier: str | None = None) -> dict[str, Any]:
    """Fields merged into ``dbaas_tiers.provision_spec``."""
    storage_type = plan.storage_type
    if plan.vendor == "gcp":
        storage_type = plan.storage_edition or "PD_SSD"
        if tier:
            try:
                from sc_runner.gcp_disks import cloud_sql_disk_type
            except ImportError:
                pass
            else:
                storage_type = cloud_sql_disk_type(tier, storage_type)
    return {
        "storage_gib": plan.storage_gib,
        "storage_type": storage_type,
        "storage_edition": plan.storage_edition or storage_type,
        "iops_tier": plan.iops_tier,
        "disk_iops": plan.iops,
        "disk_throughput_mb_s": plan.throughput_mb_s,
        "disk_gib_required": plan.storage_gib,
    }

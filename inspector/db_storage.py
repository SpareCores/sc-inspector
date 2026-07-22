"""Shared DB storage sizing for multi-VM and DBaaS (apples-to-apples I/O).

Both topologies provision the same ``storage_gib`` and the same performance
target for a given vendor. Azure uses Premium SSD v2 with explicit IOPS /
throughput; GCP uses PD-SSD where performance is size-derived, so matching
size is the contract. AWS (gp3) is stubbed for a later rollout.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from benchmark_tiers import DISK_SCHEMA_RATIO, target_schema_gib

# Usable fraction of the provisioned volume (OS/reserved headroom).
DISK_USABLE_FRAC = 0.85
# Floor large enough for Ubuntu + Docker images on multi-VM, and applied to
# DBaaS as well so GCP size-derived IOPS match.
MIN_STORAGE_GIB = 64

# Shared performance target (Azure PremiumV2 / future AWS gp3 / Hyperdisk).
# P30-equivalent: enough that durable wikipedia is not trivially IOPS-starved,
# while staying within PremiumV2 limits on the 64 GiB floor.
TARGET_IOPS = 5000
TARGET_THROUGHPUT_MB_S = 200
AZURE_IOPS_TIER = "P30"  # Flexible Server maps tier → IOPS/throughput


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


def _azure_plan(storage_gib: int) -> DbStoragePlan:
    # PremiumV2: IOPS/throughput are independently provisioned (size only caps max).
    return DbStoragePlan(
        vendor="azure",
        storage_gib=storage_gib,
        storage_type="PremiumV2_LRS",
        storage_edition="ManagedDiskV2",
        iops=TARGET_IOPS,
        throughput_mb_s=TARGET_THROUGHPUT_MB_S,
        iops_tier=AZURE_IOPS_TIER,
    )


def _gcp_plan(storage_gib: int) -> DbStoragePlan:
    # PD-SSD IOPS/throughput scale with size on both GCE and Cloud SQL — same
    # storage_gib is what makes multi-VM and DBaaS comparable.
    return DbStoragePlan(
        vendor="gcp",
        storage_gib=storage_gib,
        storage_type="pd-ssd",
        storage_edition="PD_SSD",
        iops=None,
        throughput_mb_s=None,
        iops_tier="",
    )


def _aws_plan(storage_gib: int) -> DbStoragePlan:
    # Stub for later: gp3 with the same IOPS/throughput target as Azure.
    return DbStoragePlan(
        vendor="aws",
        storage_gib=storage_gib,
        storage_type="gp3",
        storage_edition="gp3",
        iops=TARGET_IOPS,
        throughput_mb_s=TARGET_THROUGHPUT_MB_S,
        iops_tier="",
    )


def db_storage_plan(vendor: str, mem_gib: float) -> DbStoragePlan:
    """Return the shared storage plan for ``vendor`` at ``mem_gib`` RAM."""
    storage_gib = storage_gib_for_mem(mem_gib)
    vendor = (vendor or "").lower()
    if vendor == "azure":
        plan = _azure_plan(storage_gib)
    elif vendor == "gcp":
        plan = _gcp_plan(storage_gib)
    elif vendor == "aws":
        plan = _aws_plan(storage_gib)
    else:
        # Unknown vendor: size only; leave type to provider default.
        plan = DbStoragePlan(
            vendor=vendor,
            storage_gib=storage_gib,
            storage_type="",
            storage_edition="",
            iops=None,
            throughput_mb_s=None,
            iops_tier="",
        )
    return _apply_env_overrides(plan)


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
    )


def dbaas_storage_fields(plan: DbStoragePlan) -> dict[str, Any]:
    """Fields merged into ``dbaas_tiers.provision_spec``."""
    # GCP Cloud SQL expects PD_SSD; multi-VM GCE expects pd-ssd.
    storage_type = plan.storage_type
    if plan.vendor == "gcp":
        storage_type = "PD_SSD"
    return {
        "storage_gib": plan.storage_gib,
        "storage_type": storage_type,
        "storage_edition": plan.storage_edition or storage_type,
        "iops_tier": plan.iops_tier,
        "disk_iops": plan.iops,
        "disk_throughput_mb_s": plan.throughput_mb_s,
        "disk_gib_required": plan.storage_gib,
    }

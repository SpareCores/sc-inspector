"""Vendor disk performance profiles and vCPU-scaled provisioning.

Central registry of how each cloud vendor's block storage scales with size,
and how the per-VM I/O budget scales with vCPUs.  Consumers call
``provision_for_vcpus(vendor, disk_type, vcpus)`` to get a ``DiskProvision``
describing the size, IOPS, and throughput that should be provisioned so
storage never bottlenecks a CPU-proportional workload.

Design goals:
  * Small instances get modest storage (no overprovision).
  * Large instances get enough throughput that I/O is not the ceiling.
  * Cross-vendor parity: all vendors converge on a similar MB/s-per-vCPU
    budget so benchmarks measure compute, not accidental I/O starvation.
  * Reusable: DB benchmarks, storage benchmarks, or any task that needs
    "enough disk for this VM size" can call into this module.

Data sources (2025-2026):
  * GCP pd-ssd: https://cloud.google.com/compute/docs/disks/performance
  * GCP Hyperdisk Balanced: https://cloud.google.com/compute/docs/disks/hyperdisks
  * Azure PremiumSSD v2: https://learn.microsoft.com/azure/virtual-machines/disks-types#premium-ssd-v2
  * AWS gp3: https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html
  * AWS io2: https://docs.aws.amazon.com/ebs/latest/userguide/provisioned-iops.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Disk type performance curves
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiskProfile:
    """Performance model for a single disk type on a given vendor.

    All rates are *per GiB of provisioned capacity* up to the stated cap.
    For types with independently provisioned IOPS/throughput (Azure PremiumV2,
    AWS gp3/io2), ``iops_per_gib`` and ``write_mbps_per_gib`` are zero and
    the provisioned values come from the request, not the size.
    """

    vendor: str
    disk_type: str
    # Size-derived performance (set to 0 for independently-provisioned types)
    read_iops_per_gib: float
    write_iops_per_gib: float
    read_mbps_per_gib: float
    write_mbps_per_gib: float
    # Per-disk caps (absolute maximum regardless of size)
    max_read_iops: int
    max_write_iops: int
    max_read_mbps: int
    max_write_mbps: int
    # Per-VM caps scale with vCPUs (0 = no per-VM cap / use disk cap only)
    read_iops_per_vcpu: float
    write_iops_per_vcpu: float
    read_mbps_per_vcpu: float
    write_mbps_per_vcpu: float
    # Minimum provisioned size (vendor minimum)
    min_size_gib: int
    # Whether IOPS/throughput are provisioned independently of size
    independently_provisioned: bool

    def size_derived_write_mbps(self, size_gib: int) -> float:
        """Write throughput achievable at ``size_gib`` (disk cap only)."""
        if self.independently_provisioned:
            return float(self.max_write_mbps)
        return min(size_gib * self.write_mbps_per_gib, self.max_write_mbps)

    def size_derived_read_mbps(self, size_gib: int) -> float:
        if self.independently_provisioned:
            return float(self.max_read_mbps)
        return min(size_gib * self.read_mbps_per_gib, self.max_read_mbps)

    def size_derived_write_iops(self, size_gib: int) -> float:
        if self.independently_provisioned:
            return float(self.max_write_iops)
        return min(size_gib * self.write_iops_per_gib, self.max_write_iops)

    def size_derived_read_iops(self, size_gib: int) -> float:
        if self.independently_provisioned:
            return float(self.max_read_iops)
        return min(size_gib * self.read_iops_per_gib, self.max_read_iops)

    def vm_write_mbps_cap(self, vcpus: int) -> float:
        """Per-VM write throughput cap for ``vcpus`` (inf if no cap)."""
        if self.write_mbps_per_vcpu <= 0:
            return float("inf")
        return vcpus * self.write_mbps_per_vcpu

    def vm_read_mbps_cap(self, vcpus: int) -> float:
        if self.read_mbps_per_vcpu <= 0:
            return float("inf")
        return vcpus * self.read_mbps_per_vcpu

    def min_gib_for_write_mbps(self, target_mbps: float) -> int:
        """Minimum disk size to achieve ``target_mbps`` write (size-derived types)."""
        if self.independently_provisioned or self.write_mbps_per_gib <= 0:
            return self.min_size_gib
        if target_mbps >= self.max_write_mbps:
            return int(math.ceil(self.max_write_mbps / self.write_mbps_per_gib))
        return max(self.min_size_gib, int(math.ceil(target_mbps / self.write_mbps_per_gib)))

    def min_gib_for_write_iops(self, target_iops: float) -> int:
        """Minimum disk size to achieve ``target_iops`` write (size-derived types)."""
        if self.independently_provisioned or self.write_iops_per_gib <= 0:
            return self.min_size_gib
        if target_iops >= self.max_write_iops:
            return int(math.ceil(self.max_write_iops / self.write_iops_per_gib))
        return max(self.min_size_gib, int(math.ceil(target_iops / self.write_iops_per_gib)))


# ---------------------------------------------------------------------------
# Vendor disk profiles registry
# ---------------------------------------------------------------------------

# GCP pd-ssd (Persistent Disk SSD)
# Source: https://cloud.google.com/compute/docs/disks/performance#performance_limits
# Read: 30 IOPS/GiB (cap 100K), 0.48 MB/s/GiB (cap 1200 MB/s)
# Write: 30 IOPS/GiB (cap 30K), 0.48 MB/s/GiB (cap 400 MB/s)
# Per-VM (n2/c3): read 15K IOPS + 800 IOPS/vCPU, write 15K + 800/vCPU
#                 read 240 + 6 MB/s/vCPU, write 240 + 6 MB/s/vCPU (capped at disk)
GCP_PD_SSD = DiskProfile(
    vendor="gcp",
    disk_type="pd-ssd",
    read_iops_per_gib=30,
    write_iops_per_gib=30,
    read_mbps_per_gib=0.48,
    write_mbps_per_gib=0.48,
    max_read_iops=100_000,
    max_write_iops=30_000,
    max_read_mbps=1200,
    max_write_mbps=400,
    read_iops_per_vcpu=800,
    write_iops_per_vcpu=800,
    read_mbps_per_vcpu=6,
    write_mbps_per_vcpu=6,
    min_size_gib=10,
    independently_provisioned=False,
)

# GCP Hyperdisk Balanced
# Source: https://cloud.google.com/compute/docs/disks/hyperdisks#hyperdisk-balanced
# Independently provisioned IOPS (up to 160K) and throughput (up to 2400 MB/s).
# Per-VM caps same structure as pd-ssd but higher ceilings.
GCP_HYPERDISK_BALANCED = DiskProfile(
    vendor="gcp",
    disk_type="hyperdisk-balanced",
    read_iops_per_gib=0,
    write_iops_per_gib=0,
    read_mbps_per_gib=0,
    write_mbps_per_gib=0,
    max_read_iops=160_000,
    max_write_iops=160_000,
    max_read_mbps=2400,
    max_write_mbps=2400,
    read_iops_per_vcpu=0,
    write_iops_per_vcpu=0,
    read_mbps_per_vcpu=0,
    write_mbps_per_vcpu=0,
    min_size_gib=10,
    independently_provisioned=True,
)

# Azure Premium SSD v2
# Source: https://learn.microsoft.com/azure/virtual-machines/disks-types#premium-ssd-v2
# Independently provisioned: base 3000 IOPS + 500 IOPS/GiB, base 125 MB/s + 0.25 MB/s/IOPS
# Max 80K IOPS, 1200 MB/s per disk.
AZURE_PREMIUMV2 = DiskProfile(
    vendor="azure",
    disk_type="PremiumV2_LRS",
    read_iops_per_gib=0,
    write_iops_per_gib=0,
    read_mbps_per_gib=0,
    write_mbps_per_gib=0,
    max_read_iops=80_000,
    max_write_iops=80_000,
    max_read_mbps=1200,
    max_write_mbps=1200,
    read_iops_per_vcpu=0,
    write_iops_per_vcpu=0,
    read_mbps_per_vcpu=0,
    write_mbps_per_vcpu=0,
    min_size_gib=1,
    independently_provisioned=True,
)

# AWS gp3
# Source: https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html
# Base: 3000 IOPS, 125 MB/s (free). Provisioned up to 16K IOPS, 1000 MB/s.
# No per-GiB scaling — size only determines capacity.
AWS_GP3 = DiskProfile(
    vendor="aws",
    disk_type="gp3",
    read_iops_per_gib=0,
    write_iops_per_gib=0,
    read_mbps_per_gib=0,
    write_mbps_per_gib=0,
    max_read_iops=16_000,
    max_write_iops=16_000,
    max_read_mbps=1000,
    max_write_mbps=1000,
    read_iops_per_vcpu=0,
    write_iops_per_vcpu=0,
    read_mbps_per_vcpu=0,
    write_mbps_per_vcpu=0,
    min_size_gib=1,
    independently_provisioned=True,
)

# AWS io2 Block Express
# Source: https://docs.aws.amazon.com/ebs/latest/userguide/provisioned-iops.html
# Up to 256K IOPS (1000 IOPS/GiB), 4000 MB/s.
AWS_IO2 = DiskProfile(
    vendor="aws",
    disk_type="io2",
    read_iops_per_gib=0,
    write_iops_per_gib=0,
    read_mbps_per_gib=0,
    write_mbps_per_gib=0,
    max_read_iops=256_000,
    max_write_iops=256_000,
    max_read_mbps=4000,
    max_write_mbps=4000,
    read_iops_per_vcpu=0,
    write_iops_per_vcpu=0,
    read_mbps_per_vcpu=0,
    write_mbps_per_vcpu=0,
    min_size_gib=4,
    independently_provisioned=True,
)

# Registry: (vendor, disk_type) -> DiskProfile
PROFILES: dict[tuple[str, str], DiskProfile] = {
    ("gcp", "pd-ssd"): GCP_PD_SSD,
    ("gcp", "hyperdisk-balanced"): GCP_HYPERDISK_BALANCED,
    ("azure", "PremiumV2_LRS"): AZURE_PREMIUMV2,
    ("aws", "gp3"): AWS_GP3,
    ("aws", "io2"): AWS_IO2,
}


def get_profile(vendor: str, disk_type: str) -> DiskProfile:
    """Look up a disk profile by vendor and type. Raises KeyError if unknown."""
    key = (vendor.lower(), disk_type)
    if key not in PROFILES:
        raise KeyError(f"No disk profile for ({vendor!r}, {disk_type!r}). "
                       f"Available: {sorted(PROFILES.keys())}")
    return PROFILES[key]


# ---------------------------------------------------------------------------
# vCPU-scaled provisioning
# ---------------------------------------------------------------------------

# Target I/O budget per vCPU. Enough that async TPC-B is not trivially
# I/O-starved on large SKUs, without overprovisioning: on n2-standard-128,
# 800 GiB (3 MB/s/vCPU) delivered ~600 MB/s writes but the same ~105–110K TPS
# peak as 151 GiB — the ceiling was WALInsert/ProcArray/transactionid, not
# disk. Halved budget (~192 MB/s at 128 vCPUs) still beats the old schema-only
# floor and aligns with the prior Azure/AWS ~200 MB/s target.
WRITE_MBPS_PER_VCPU = 1.5
WRITE_IOPS_PER_VCPU = 25.0
# Minimum throughput floor (never provision below this even for 1 vCPU).
MIN_WRITE_MBPS = 50
MIN_WRITE_IOPS = 1000


@dataclass(frozen=True)
class DiskProvision:
    """Concrete provisioning decision for a VM."""

    vendor: str
    disk_type: str
    size_gib: int
    iops: int | None        # None = size-derived (vendor handles it)
    throughput_mb_s: int | None  # None = size-derived
    # Effective performance at this provision (for informational / logging)
    effective_write_iops: int
    effective_write_mbps: int
    effective_read_iops: int
    effective_read_mbps: int


def provision_for_vcpus(
    vendor: str,
    disk_type: str,
    vcpus: int,
    *,
    min_size_gib: int = 0,
    extra_write_mbps: float = 0,
    extra_write_iops: float = 0,
) -> DiskProvision:
    """Compute disk provisioning so I/O scales with ``vcpus``.

    For size-derived types (GCP pd-ssd), returns a larger ``size_gib`` to
    achieve the throughput target. For independently-provisioned types
    (Azure PremiumV2, AWS gp3), returns explicit IOPS/throughput values.

    Parameters
    ----------
    vendor, disk_type : str
        Must match a registered ``DiskProfile``.
    vcpus : int
        Number of vCPUs on the target VM.
    min_size_gib : int
        Minimum disk size (e.g. for OS + data). The result is at least this.
    extra_write_mbps, extra_write_iops : float
        Additional throughput/IOPS beyond the per-vCPU formula (e.g. for
        schema that needs more capacity than the formula alone).
    """
    profile = get_profile(vendor, disk_type)
    vcpus = max(1, int(vcpus))

    target_mbps = max(MIN_WRITE_MBPS, vcpus * WRITE_MBPS_PER_VCPU + extra_write_mbps)
    target_iops = max(MIN_WRITE_IOPS, vcpus * WRITE_IOPS_PER_VCPU + extra_write_iops)

    # Clamp to per-VM cap if applicable
    vm_mbps_cap = profile.vm_write_mbps_cap(vcpus)
    vm_iops_cap = vcpus * profile.write_iops_per_vcpu if profile.write_iops_per_vcpu > 0 else float("inf")
    target_mbps = min(target_mbps, vm_mbps_cap, profile.max_write_mbps)
    target_iops = min(target_iops, vm_iops_cap, profile.max_write_iops)

    if profile.independently_provisioned:
        size_gib = max(profile.min_size_gib, min_size_gib)
        prov_iops = int(math.ceil(target_iops))
        prov_mbps = int(math.ceil(target_mbps))
        eff_w_iops = prov_iops
        eff_w_mbps = prov_mbps
        eff_r_iops = prov_iops
        eff_r_mbps = prov_mbps
    else:
        # Size-derived: pick the smallest size that meets both targets
        size_for_mbps = profile.min_gib_for_write_mbps(target_mbps)
        size_for_iops = profile.min_gib_for_write_iops(target_iops)
        size_gib = max(profile.min_size_gib, min_size_gib, size_for_mbps, size_for_iops)
        eff_w_iops = int(min(profile.size_derived_write_iops(size_gib), vm_iops_cap))
        eff_w_mbps = int(min(profile.size_derived_write_mbps(size_gib), vm_mbps_cap))
        eff_r_iops = int(min(profile.size_derived_read_iops(size_gib), vcpus * profile.read_iops_per_vcpu if profile.read_iops_per_vcpu > 0 else float("inf")))
        eff_r_mbps = int(min(profile.size_derived_read_mbps(size_gib), profile.vm_read_mbps_cap(vcpus)))
        prov_iops = None
        prov_mbps = None

    return DiskProvision(
        vendor=vendor,
        disk_type=disk_type,
        size_gib=size_gib,
        iops=prov_iops,
        throughput_mb_s=prov_mbps,
        effective_write_iops=eff_w_iops,
        effective_write_mbps=eff_w_mbps,
        effective_read_iops=eff_r_iops,
        effective_read_mbps=eff_r_mbps,
    )


# ---------------------------------------------------------------------------
# Convenience: show provisioning table for a range of vCPUs
# ---------------------------------------------------------------------------

def print_provision_table(vendor: str, disk_type: str, min_size_gib: int = 64) -> None:
    """Print a human-readable table of provisions across vCPU counts."""
    print(f"{'vCPUs':>5} {'size_GiB':>8} {'w_IOPS':>7} {'w_MB/s':>7} {'r_IOPS':>7} {'r_MB/s':>7} {'prov_IOPS':>9} {'prov_MB/s':>9}")
    for v in [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 128, 192, 256]:
        p = provision_for_vcpus(vendor, disk_type, v, min_size_gib=min_size_gib)
        pi = str(p.iops) if p.iops is not None else "size"
        pm = str(p.throughput_mb_s) if p.throughput_mb_s is not None else "size"
        print(f"{v:>5} {p.size_gib:>8} {p.effective_write_iops:>7} {p.effective_write_mbps:>7} "
              f"{p.effective_read_iops:>7} {p.effective_read_mbps:>7} {pi:>9} {pm:>9}")


if __name__ == "__main__":
    for vendor, dtype in [("gcp", "pd-ssd"), ("azure", "PremiumV2_LRS"), ("aws", "gp3")]:
        print(f"\n{'='*70}")
        print(f"  {vendor} / {dtype}")
        print(f"{'='*70}")
        print_provision_table(vendor, dtype)

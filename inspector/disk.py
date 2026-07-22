"""Dynamic disk provisioning helpers for inspector tasks."""

from __future__ import annotations

import math

VOLUME_SIZE = 128  # keep in sync with lib.VOLUME_SIZE
# Fraction of provisioned root volume treated as usable for schema.
DISK_USABLE_FRAC = 0.85

DISK_CONFIGURABLE_VENDORS = frozenset(
    {"aws", "azure", "gcp", "alicloud", "upcloud", "vultr"}
)


def effective_disk_gib(vendor: str, srv, required_gib: float) -> float:
    if vendor in DISK_CONFIGURABLE_VENDORS:
        if required_gib <= 0:
            return float(VOLUME_SIZE)
        # Reserve headroom so disk_feasible(required) passes at runtime.
        return max(VOLUME_SIZE, math.ceil(required_gib / DISK_USABLE_FRAC))
    return float(srv.storage_size or 0)


def disk_feasible(need_gib: float, avail_gib: float) -> bool:
    if need_gib <= 0:
        return True
    return need_gib <= avail_gib * DISK_USABLE_FRAC

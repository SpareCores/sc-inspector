"""Dynamic disk provisioning helpers for inspector tasks."""

from __future__ import annotations

import math

VOLUME_SIZE = 128  # keep in sync with lib.VOLUME_SIZE

DISK_CONFIGURABLE_VENDORS = frozenset(
    {"aws", "azure", "gcp", "alicloud", "upcloud", "vultr"}
)


def effective_disk_gib(vendor: str, srv, required_gib: float) -> float:
    if vendor in DISK_CONFIGURABLE_VENDORS:
        return max(VOLUME_SIZE, math.ceil(required_gib))
    return float(srv.storage_size or 0)


def disk_feasible(need_gib: float, avail_gib: float) -> bool:
    if need_gib <= 0:
        return True
    return need_gib <= avail_gib * 0.85

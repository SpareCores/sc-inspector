#!/usr/bin/env python3
"""Live Azure quota check against the current subscription (requires ARM auth)."""

from __future__ import annotations

import sys

from azure_dbaas_quota import check_azure_dbaas_region


def main() -> int:
    regions = sys.argv[1:] or ["northeurope", "westeurope", "centralus"]
    for region in regions:
        ok, reason = check_azure_dbaas_region(
            region,
            pg_sku="Standard_E16ds_v5",
            pg_vcpus=16,
            vm_sku="Standard_D4s_v3",
            vm_vcpus=4,
        )
        status = "OK" if ok else "SKIP"
        print(f"{region}: {status} {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

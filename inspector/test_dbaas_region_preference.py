"""Azure DBaaS region ordering prefers offer/quota headroom over raw price."""

from dbaas_start import _prefer_azure_regions_with_quota


def test_prefer_quota_regions_before_cheaper_restricted(monkeypatch):
    prices = {
        "eastus": 0.01,
        "eastus2": 0.02,
        "uaenorth": 0.05,
        "canadacentral": 0.03,
    }

    def fake_pg_quota(_vendor, region, _sku, _vcpus):
        if region == "eastus":
            return False, "postgres offer restricted in eastus"
        return True, ""

    def fake_cores(region):
        return {"eastus2": 10, "uaenorth": 100, "canadacentral": 40, "eastus": 0}[region]

    monkeypatch.setattr(
        "dbaas_start.check_dbaas_postgres_quota", fake_pg_quota
    )
    monkeypatch.setattr(
        "azure_dbaas_quota.vm_regional_cores_available", fake_cores
    )

    ordered = _prefer_azure_regions_with_quota(
        ["eastus", "eastus2", "uaenorth", "canadacentral"],
        pg_sku="Standard_D16ads_v6",
        pg_vcpus=16,
        prices=prices,
    )
    # Quota OK first, higher VM core headroom before cheaper eastus2, restricted last.
    assert ordered == ["uaenorth", "canadacentral", "eastus2", "eastus"]

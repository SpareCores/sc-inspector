"""Azure quota family keys come from the ARM Resource SKUs API, not the sc-data catalog."""

import azure_dbaas_quota as q


def _fake_index(monkeypatch, index):
    q._compute_sku_info.cache_clear()
    monkeypatch.setattr(q, "_region_compute_sku_index", lambda _sub, _region: index)


def test_quota_family_uses_arm_family(monkeypatch):
    _fake_index(
        monkeypatch,
        {"Standard_D2ads_v6": q.ComputeSkuInfo(family="standardDADSv6Family", restrictions=[])},
    )
    assert (
        q.sku_family_quota_name("Standard_D2ads_v6", subscription_id="sub", region="centralus")
        == "standardDADSv6Family"
    )


def test_quota_family_needs_subscription_and_region():
    assert q.sku_family_quota_name("Standard_D2ads_v6") is None


def test_client_vm_quota_checks_arm_family(monkeypatch):
    quotas = {
        "cores": q.QuotaEntry(name="cores", current=0, limit=100),
        "standardDALSv7Family": q.QuotaEntry(name="standardDALSv7Family", current=9, limit=10),
    }
    ok, reason = q._evaluate_client_vm_quota(
        "centralus",
        vm_sku="Standard_D2als_v7",
        vm_vcpus=2,
        subscription_id="sub",
        vm_quotas=quotas,
        sku_index={
            "Standard_D2als_v7": q.ComputeSkuInfo(family="standardDALSv7Family", restrictions=[])
        },
    )
    assert not ok
    assert "standardDALSv7Family" in reason

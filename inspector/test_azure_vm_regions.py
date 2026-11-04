"""Quota-aware region order for Azure VM starts (no Azure calls: everything is faked)."""

import pytest

import azure_vm_regions as avr
from azure_dbaas_quota import ComputeSkuInfo, QuotaEntry

SKU = "Standard_D16ds_v6"
FAMILY = "StandardDdsv6Family"


def _quotas(**limits_used):
    """_quotas(cores=(limit, used), StandardDdsv6Family=(limit, used), ...)"""
    return {k: QuotaEntry(name=k, limit=l, current=c) for k, (l, c) in limits_used.items()}


def _ips(basic=(20, 0), total=(50, 0)):
    return _quotas(IPv4BasicSkuPublicIpAddresses=basic, PublicIPAddresses=total)


ZONE_ONLY = [{"type": "Zone", "reasonCode": "NotAvailableForSubscription",
              "restrictionInfo": {"locations": ["swedencentral"], "zones": ["1"]}}]
WHOLE_REGION = [{"type": "Location", "reasonCode": "NotAvailableForSubscription",
                 "restrictionInfo": {"locations": ["westeurope"]}}]

# region -> (sku restrictions or None if not offered, vm quotas, network quotas)
FAKE = {
    "eastus": ([], _quotas(cores=(378, 0), **{FAMILY: (10, 0)}), _ips()),           # family too small
    "centralus": ([], _quotas(cores=(22239, 0), **{FAMILY: (192, 0)}), _ips()),      # fits 12
    "westeurope": (WHOLE_REGION, _quotas(cores=(1673, 0), **{FAMILY: (64, 0)}), _ips()),
    "northeurope": ([], _quotas(cores=(400, 0), **{FAMILY: (64, 0)}), _ips(basic=(20, 20))),  # no Basic IP
    "uksouth": (None, _quotas(cores=(400, 0), **{FAMILY: (64, 0)}), _ips()),         # size not offered
    "swedencentral": (ZONE_ONLY, _quotas(cores=(100, 0), **{FAMILY: (64, 0)}), _ips()),  # zone-only: OK
    "canadacentral": ([], _quotas(cores=(10, 0), **{FAMILY: (64, 0)}), _ips()),      # regional cores too small
}
PRICE_ORDER = list(FAKE)


@pytest.fixture
def fake_azure(monkeypatch):
    monkeypatch.delenv("AZURE_SKIP_VM_QUOTA_CHECK", raising=False)
    monkeypatch.setattr(avr, "_subscription_id", lambda: "sub")
    monkeypatch.setattr(avr, "sku_family_quota_name", lambda sku: FAMILY)

    def sku_index(sub, region):
        restrictions = FAKE[region][0]
        return {} if restrictions is None else {SKU: ComputeSkuInfo(family=FAMILY, restrictions=restrictions)}

    monkeypatch.setattr(avr, "_region_compute_sku_index", sku_index)
    monkeypatch.setattr(avr, "_vm_quotas", lambda sub, region: FAKE[region][1])
    monkeypatch.setattr(avr, "_network_quotas", lambda sub, region: FAKE[region][2])


def test_eligible_regions_first_then_price_ordered_fallback(fake_azure):
    order, family = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert family == FAMILY
    assert set(order[:2]) == {"centralus", "swedencentral"}
    assert order[2:] == ["eastus", "westeurope", "northeurope", "uksouth", "canadacentral"]


def test_reasons(fake_azure):
    def check(region):
        restrictions, vm, net = FAKE[region]
        index = {} if restrictions is None else {SKU: ComputeSkuInfo(FAMILY, restrictions)}
        return avr.check_region(region, sku=SKU, vcpus=16, family=FAMILY,
                                sku_index=index, vm_quotas=vm, network_quotas=net)

    assert check("centralus").ok and check("centralus").fits == 12
    assert "StandardDdsv6Family" in check("eastus").reason
    assert check("westeurope").reason == "size not available for subscription"
    assert "IPv4BasicSkuPublicIpAddresses" in check("northeurope").reason
    assert check("uksouth").reason == "size not offered"
    assert check("swedencentral").ok
    assert "regional vCPUs" in check("canadacentral").reason


def test_disabled_keeps_price_order(fake_azure, monkeypatch):
    monkeypatch.setenv("AZURE_SKIP_VM_QUOTA_CHECK", "1")
    assert avr.order_regions(SKU, 16, PRICE_ORDER) == (PRICE_ORDER, None)


def test_lookup_failure_keeps_price_order(fake_azure, monkeypatch):
    def boom(sub, region):
        raise RuntimeError("ARM down")

    monkeypatch.setattr(avr, "_vm_quotas", boom)
    assert avr.order_regions(SKU, 16, PRICE_ORDER) == (PRICE_ORDER, None)


def test_no_subscription_keeps_price_order(fake_azure, monkeypatch):
    monkeypatch.setattr(avr, "_subscription_id", lambda: None)
    assert avr.order_regions(SKU, 16, PRICE_ORDER) == (PRICE_ORDER, None)


def test_location_restriction_without_location_list_blocks_everywhere():
    info = ComputeSkuInfo(FAMILY, [{"type": "Location", "restrictionInfo": {}}])
    assert avr.location_restricted(info, "centralus")


def test_weighted_shuffle_spreads_first_choice_by_capacity():
    import random
    from collections import Counter

    rng = random.Random(42)
    checks = [avr.RegionCheck("big", True, fits=12), avr.RegionCheck("small", True, fits=4)]
    firsts = Counter(avr.weighted_shuffle(checks, rng)[0] for _ in range(4000))
    # P(big first) = 12 / (12 + 4) = 0.75
    assert 0.70 < firsts["big"] / 4000 < 0.80
    assert firsts["small"] > 0


def test_weighted_shuffle_caps_weight():
    import random
    from collections import Counter

    rng = random.Random(7)
    checks = [avr.RegionCheck("huge", True, fits=10_000), avr.RegionCheck("cap", True, fits=avr.MAX_WEIGHT)]
    firsts = Counter(avr.weighted_shuffle(checks, rng)[0] for _ in range(4000))
    # both capped to the same weight: roughly a coin flip
    assert 0.45 < firsts["huge"] / 4000 < 0.55


def test_order_is_randomized_but_fallback_stays_last(fake_azure):
    import random

    firsts = {avr.order_regions(SKU, 16, PRICE_ORDER, rng=random.Random(seed))[0][0] for seed in range(200)}
    assert firsts == {"centralus", "swedencentral"}
    for seed in range(20):
        order, _ = avr.order_regions(SKU, 16, PRICE_ORDER, rng=random.Random(seed))
        assert order[2:] == ["eastus", "westeurope", "northeurope", "uksouth", "canadacentral"]

"""Quota-aware region order for Azure VM starts (no Azure calls: everything is faked)."""

import time

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
    avr.reset_ledger()
    avr.reset_region_cache()
    yield
    avr.reset_ledger()
    avr.reset_region_cache()


def test_eligible_regions_first_then_price_ordered_fallback(fake_azure):
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    order = plan.regions
    assert plan.family == FAMILY
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
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert plan.regions == PRICE_ORDER and plan.held is None


def test_lookup_failure_keeps_price_order(fake_azure, monkeypatch):
    def boom(sub, region):
        raise RuntimeError("ARM down")

    monkeypatch.setattr(avr, "_vm_quotas", boom)
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert plan.regions == PRICE_ORDER and plan.held is None


def test_no_subscription_keeps_price_order(fake_azure, monkeypatch):
    monkeypatch.setattr(avr, "_subscription_id", lambda: None)
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert plan.regions == PRICE_ORDER and plan.held is None


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

    firsts = set()
    for seed in range(200):
        plan = avr.order_regions(SKU, 16, PRICE_ORDER, rng=random.Random(seed))
        firsts.add(plan.regions[0])
        avr.release(plan.held)
    assert firsts == {"centralus", "swedencentral"}
    for seed in range(20):
        plan = avr.order_regions(SKU, 16, PRICE_ORDER, rng=random.Random(seed))
        avr.release(plan.held)
        assert plan.regions[2:] == ["eastus", "westeurope", "northeurope", "uksouth", "canadacentral"]


def test_planning_reserves_first_choice(fake_azure):
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    first = plan.regions[0]
    assert plan.held is not None and plan.held.region == first
    assert avr.reserved(first, FAMILY) == 16
    assert avr.reserved(first, "cores") == 16
    assert avr.reserved(first, "IPv4BasicSkuPublicIpAddresses") == 1
    avr.release(plan.held)
    avr.release(plan.held)  # idempotent
    assert avr.reserved(first, FAMILY) == 0


def test_regions_fill_up_then_fall_back_to_price_order(fake_azure):
    from collections import Counter

    # centralus fits 12 x 16 vCPU (192), swedencentral 4 (64): 16 starts in total
    placed = Counter()
    for _ in range(16):
        plan = avr.order_regions(SKU, 16, PRICE_ORDER)
        assert plan.held is not None
        placed[plan.held.region] += 1
    assert placed == {"centralus": 12, "swedencentral": 4}
    # quota exhausted everywhere: nothing held, plain price order of all regions
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert plan.held is None
    assert plan.regions == PRICE_ORDER


def test_release_after_failure_frees_capacity(fake_azure):
    plans = [avr.order_regions(SKU, 16, PRICE_ORDER) for _ in range(16)]
    assert avr.order_regions(SKU, 16, PRICE_ORDER).held is None
    avr.release(plans[0].held)
    again = avr.order_regions(SKU, 16, PRICE_ORDER)
    assert again.held is not None and again.held.region == plans[0].held.region


def test_plan_reserve_reuses_or_moves_the_held_reservation(fake_azure):
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    first = plan.regions[0]
    assert plan.reserve(first) is not None and avr.reserved(first, FAMILY) == 16  # reused, not doubled
    other = plan.regions[1]
    res = plan.reserve(other)  # held one already handed out: plain new hold
    assert avr.reserved(other, FAMILY) == 16

    plan2 = avr.order_regions(SKU, 16, PRICE_ORDER)
    held_region = plan2.held.region
    before = avr.reserved(held_region, FAMILY)
    elsewhere = next(r for r in PRICE_ORDER if r != held_region)
    plan2.reserve(elsewhere)  # skipped the planned region: its hold is released
    assert avr.reserved(held_region, FAMILY) == before - 16
    avr.release(res)


def test_parallel_planning_never_oversubscribes(fake_azure):
    import threading
    from collections import Counter

    placed = Counter()
    lock = threading.Lock()

    def start():
        plan = avr.order_regions(SKU, 16, PRICE_ORDER)
        with lock:
            placed[plan.held.region if plan.held else None] += 1

    threads = [threading.Thread(target=start) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert placed["centralus"] == 12
    assert placed["swedencentral"] == 4
    assert placed[None] == 14


def test_unknown_vcpus_keeps_price_order(fake_azure):
    plan = avr.order_regions(SKU, 0, PRICE_ORDER)
    assert plan.regions == PRICE_ORDER and plan.held is None


def test_public_ip_lookup_failure_only_skips_the_ip_check(fake_azure, monkeypatch):
    def boom(sub, region):
        raise RuntimeError("403 Microsoft.Network/locations/usages/read")

    monkeypatch.setattr(avr, "_network_quotas", boom)
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    # northeurope (no Basic IP left) becomes eligible as IPs are unknown; vCPU checks still apply
    assert set(plan.regions[:3]) == {"centralus", "swedencentral", "northeurope"}
    assert plan.regions[3:] == ["eastus", "westeurope", "uksouth", "canadacentral"]


def test_one_failing_region_only_moves_that_region_to_fallback(fake_azure, monkeypatch):
    real = avr._vm_quotas

    def flaky(sub, region):
        if region == "centralus":
            raise RuntimeError("429 Too Many Requests")
        return real(sub, region)

    monkeypatch.setattr(avr, "_vm_quotas", flaky)
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    # swedencentral still planned quota-aware; centralus tried last-ish in the fallback
    assert plan.regions[0] == "swedencentral"
    assert plan.held is not None and plan.held.region == "swedencentral"
    assert plan.regions[1:] == ["eastus", "centralus", "westeurope", "northeurope", "uksouth", "canadacentral"]


def test_parallel_planning_fetches_each_region_once(fake_azure, monkeypatch):
    import threading
    from collections import Counter

    calls = Counter()
    lock = threading.Lock()
    real = avr._vm_quotas

    def counting(sub, region):
        with lock:
            calls[region] += 1
        time.sleep(0.01)  # make the threads overlap
        return real(sub, region)

    monkeypatch.setattr(avr, "_vm_quotas", counting)
    threads = [threading.Thread(target=avr.order_regions, args=(SKU, 16, PRICE_ORDER)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == {region: 1 for region in PRICE_ORDER}


def test_failed_region_is_not_refetched_within_ttl(fake_azure, monkeypatch):
    from collections import Counter

    calls = Counter()

    def failing(sub, region):
        calls[region] += 1
        raise RuntimeError("timeout")

    monkeypatch.setattr(avr, "_vm_quotas", failing)
    for _ in range(5):
        avr.order_regions(SKU, 16, PRICE_ORDER)
    assert calls == {region: 1 for region in PRICE_ORDER}

    monkeypatch.setattr(avr, "REGION_DATA_ERROR_TTL_SEC", 0)
    avr.order_regions(SKU, 16, PRICE_ORDER)
    assert calls == {region: 2 for region in PRICE_ORDER}


def test_family_from_sku_index_is_reserved_when_catalog_has_none(fake_azure, monkeypatch):
    # SKU missing from the sc-data catalog: the family comes from each region's SKU index
    monkeypatch.setattr(avr, "sku_family_quota_name", lambda sku: None)
    plan = avr.order_regions(SKU, 16, PRICE_ORDER)
    first = plan.regions[0]
    assert plan.family is None and plan.family_for(first) == FAMILY
    assert avr.reserved(first, FAMILY) == 16
    other = next(r for r in plan.regions[1:] if r in ("centralus", "swedencentral"))
    plan.reserve(first)
    plan.reserve(other)
    assert avr.reserved(other, FAMILY) == 16


def test_family_quota_not_oversubscribed_without_catalog_family(fake_azure, monkeypatch):
    from collections import Counter

    monkeypatch.setattr(avr, "sku_family_quota_name", lambda sku: None)
    placed = Counter()
    for _ in range(20):
        plan = avr.order_regions(SKU, 16, PRICE_ORDER)
        placed[plan.held.region if plan.held else None] += 1
    # family quota limits: centralus 192/16 = 12, swedencentral 64/16 = 4
    assert placed == {"centralus": 12, "swedencentral": 4, None: 4}

"""Azure DBaaS region ordering: ready regions (PG quota + client headroom) first, randomized."""

import random
from collections import Counter

import pytest

from benchmark_tiers import DBAAS_CLIENT_MAX_VCPUS
from dbaas_start import _prefer_azure_regions_with_quota

PRICES = {
    "eastus": 0.01,
    "eastus2": 0.02,
    "canadacentral": 0.03,
    "uaenorth": 0.05,
    "westus": 0.06,
    "northeurope": 0.07,
    "swedencentral": 0.08,
}
REGIONS = list(PRICES)


@pytest.fixture
def fake_quota(monkeypatch):
    monkeypatch.delenv("DBAAS_SKIP_QUOTA_CHECK", raising=False)
    state = {
        # PG offer/quota per region
        "pg_ok": {"eastus": False, "eastus2": True, "canadacentral": True, "uaenorth": True,
                  "westus": True, "northeurope": True, "swedencentral": True},
        # free regional VM vCPUs net of reservations (None = lookup failed / unknown)
        "cores": {"eastus": 0, "eastus2": 10, "canadacentral": 40, "uaenorth": 100,
                  "westus": 3, "northeurope": 400, "swedencentral": None},
        # free public IPs (None = unknown)
        "ips": {"eastus": 5, "eastus2": None, "canadacentral": 20, "uaenorth": 20,
                "westus": 20, "northeurope": 0, "swedencentral": None},
    }

    def fake_pg_quota(_vendor, region, _sku, _vcpus):
        return (True, "") if state["pg_ok"][region] else (False, f"postgres offer restricted in {region}")

    monkeypatch.setattr("dbaas_start.check_dbaas_postgres_quota", fake_pg_quota)
    monkeypatch.setattr("azure_vm_regions.regional_cores_free", lambda r: state["cores"][r])
    monkeypatch.setattr("azure_vm_regions.public_ips_free", lambda r: state["ips"][r])
    monkeypatch.setattr("azure_vm_regions.prefetch", lambda regions: None)
    return state


def _order(rng=None):
    return _prefer_azure_regions_with_quota(
        REGIONS, pg_sku="Standard_D16ds_v5", pg_vcpus=16, prices=PRICES, rng=rng
    )


def test_ready_then_unknown_then_no_client_headroom_then_no_pg_quota(fake_quota):
    ordered = _order(random.Random(1))
    # ready: PG quota, room for the largest client and a free (or unknown) public IP
    assert set(ordered[:3]) == {"eastus2", "canadacentral", "uaenorth"}
    # PG quota, client capacity unknown (lookup failed): not pushed below no-headroom regions
    assert ordered[3] == "swedencentral"
    # PG quota but no client headroom (westus: 3 vCPUs < 4, northeurope: no IP), by price
    assert ordered[4:6] == ["westus", "northeurope"]
    # no PG quota/offer last
    assert ordered[6:] == ["eastus"]


def test_client_headroom_uses_the_largest_client_size(fake_quota):
    assert DBAAS_CLIENT_MAX_VCPUS == 4
    fake_quota["cores"]["westus"] = DBAAS_CLIENT_MAX_VCPUS
    assert "westus" in _order(random.Random(2))[:4]


def test_ready_regions_are_spread_not_most_cores_first(fake_quota):
    firsts = Counter(_order(random.Random(seed))[0] for seed in range(3000))
    # all ready regions get picked first sometimes, in proportion to how many
    # 4-vCPU clients fit: eastus2 10 cores -> 2, canadacentral 40 -> 10,
    # uaenorth 100 cores but 20 free IPs -> 20 (so it does not take every draw)
    assert set(firsts) == {"eastus2", "canadacentral", "uaenorth"}
    assert 1.6 < firsts["uaenorth"] / firsts["canadacentral"] < 2.4
    assert 3 < firsts["canadacentral"] / firsts["eastus2"] < 8


def test_quota_check_disabled_keeps_price_order(fake_quota, monkeypatch):
    monkeypatch.setenv("DBAAS_SKIP_QUOTA_CHECK", "1")
    ordered = _order(random.Random(3))
    assert ordered == ["eastus2", "canadacentral", "uaenorth", "westus", "northeurope", "swedencentral", "eastus"]

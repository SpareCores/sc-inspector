"""Azure DBaaS region ordering: ready regions (PG quota + client headroom) first, randomized."""

import random
from collections import Counter

import pytest

from dbaas_start import _prefer_azure_regions_with_quota

PRICES = {
    "eastus": 0.01,
    "eastus2": 0.02,
    "canadacentral": 0.03,
    "uaenorth": 0.05,
    "westus": 0.06,
    "northeurope": 0.07,
}
REGIONS = list(PRICES)


@pytest.fixture
def fake_quota(monkeypatch):
    state = {
        # PG offer/quota per region
        "pg_ok": {"eastus": False, "eastus2": True, "canadacentral": True, "uaenorth": True,
                  "westus": True, "northeurope": True},
        # free regional VM vCPUs (client needs 2)
        "cores": {"eastus": 0, "eastus2": 10, "canadacentral": 40, "uaenorth": 100,
                  "westus": 1, "northeurope": 400},
        # free public IPs (None = unknown)
        "ips": {"eastus": 5, "eastus2": None, "canadacentral": 20, "uaenorth": 20,
                "westus": 20, "northeurope": 0},
    }

    def fake_pg_quota(_vendor, region, _sku, _vcpus):
        return (True, "") if state["pg_ok"][region] else (False, f"postgres offer restricted in {region}")

    monkeypatch.setattr("dbaas_start.check_dbaas_postgres_quota", fake_pg_quota)
    monkeypatch.setattr("azure_dbaas_quota.vm_regional_cores_available", lambda r: state["cores"][r])
    monkeypatch.setattr("azure_vm_regions.public_ips_free", lambda r: state["ips"][r])
    return state


def _order(rng=None):
    return _prefer_azure_regions_with_quota(
        REGIONS, pg_sku="Standard_D16ds_v5", pg_vcpus=16, prices=PRICES, rng=rng
    )


def test_ready_regions_first_then_no_client_headroom_then_no_pg_quota(fake_quota):
    ordered = _order(random.Random(1))
    # ready: PG quota, >= 2 free vCPUs and a free (or unknown) public IP
    assert set(ordered[:3]) == {"eastus2", "canadacentral", "uaenorth"}
    # PG quota but no client headroom (westus: 1 vCPU, northeurope: no IP), by price
    assert ordered[3:5] == ["westus", "northeurope"]
    # no PG quota/offer last
    assert ordered[5:] == ["eastus"]


def test_ready_regions_are_spread_not_most_cores_first(fake_quota):
    firsts = Counter(_order(random.Random(seed))[0] for seed in range(3000))
    # all ready regions get picked first sometimes; big regions are capped,
    # so uaenorth (100 cores) does not take every draw as it did before
    assert set(firsts) == {"eastus2", "canadacentral", "uaenorth"}
    assert firsts["eastus2"] > 0
    assert abs(firsts["canadacentral"] - firsts["uaenorth"]) < 300  # both capped at the same weight


def test_unknown_vm_cores_keeps_price_order(fake_quota):
    # vm_regional_cores_available() returns 0 when the quota check is skipped/unavailable
    for region in REGIONS:
        fake_quota["cores"][region] = 0
        fake_quota["ips"][region] = None
    ordered = _order(random.Random(3))
    assert ordered == ["eastus2", "canadacentral", "uaenorth", "westus", "northeurope", "eastus"]

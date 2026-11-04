"""DBaaS client attempts reserve vCPUs/public IP in the job-wide ledger (Azure)."""

from types import SimpleNamespace

import pytest

import azure_vm_regions as avr
import dbaas_start
import lib
from azure_dbaas_quota import QuotaEntry

CLIENT = SimpleNamespace(api_reference="Standard_D2als_v7", vcpus=2)
FAMILY = "StandardDalsv7Family"


class FakeDbaasTask:
    pass


def _region(ips_limit):
    return avr.RegionData(
        sku_index={},
        vm_quotas={"cores": QuotaEntry("cores", current=0, limit=100)},
        network_quotas={
            "IPv4BasicSkuPublicIpAddresses": QuotaEntry("IPv4BasicSkuPublicIpAddresses", current=0, limit=ips_limit),
            "PublicIPAddresses": QuotaEntry("PublicIPAddresses", current=0, limit=50),
        },
    )


@pytest.fixture
def dbaas(monkeypatch):
    regions = {"centralus": _region(2), "westeurope": _region(20)}
    attempts = []
    outcome = {"result": (False, False, False)}

    monkeypatch.setattr(lib, "DbaasDbTask", FakeDbaasTask)
    monkeypatch.setattr(dbaas_start, "provision_spec", lambda target: {"sku_name": "Standard_D16ds_v5"})
    monkeypatch.setattr(dbaas_start, "dbaas_client_req", lambda: None)
    monkeypatch.setattr(dbaas_start, "stack_slug", lambda target: "slug")
    monkeypatch.setattr(
        dbaas_start, "_dbaas_location_candidates",
        lambda *a, **kw: [("region", "centralus"), ("region", "westeurope")],
    )
    monkeypatch.setattr(dbaas_start, "check_dbaas_postgres_quota", lambda *a: (True, ""))
    monkeypatch.setattr(dbaas_start, "rank_dbaas_client_instances", lambda *a: [CLIENT])
    monkeypatch.setattr(dbaas_start, "filter_clients_by_vm_quota", lambda vendor, region, clients: clients)

    def try_provision(vendor, target, client, region, *rest):
        attempts.append(region)
        # what this job holds in the region while the attempt runs
        outcome.setdefault("held_during", []).append(avr.reserved(region, "IPv4BasicSkuPublicIpAddresses"))
        result = outcome["result"]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(dbaas_start, "_try_provision_dbaas_stack", try_provision)
    monkeypatch.setattr(avr, "_cached_region_data", lambda region, sub=None: regions[region])
    monkeypatch.setattr(avr, "sku_family_quota_name", lambda sku: FAMILY)
    avr.reset_ledger()
    yield SimpleNamespace(attempts=attempts, outcome=outcome)
    avr.reset_ledger()


def _start():
    return dbaas_start.try_start_dbaas_inspect(
        None, None, "/tmp", "azure", SimpleNamespace(cpu_count=16, instance_key="x"),
        [FakeDbaasTask()], ["centralus", "westeurope"], [], {}, 60, "", "", None, None, [],
    )


def test_successful_stack_keeps_its_reservation(dbaas):
    dbaas.outcome["result"] = (True, False, False)
    assert _start() == (True, True)
    assert dbaas.outcome["held_during"] == [1]
    assert avr.reserved("centralus", "IPv4BasicSkuPublicIpAddresses") == 1
    assert avr.reserved("centralus", FAMILY) == 2
    assert avr.reserved("centralus", "cores") == 2


def test_failed_attempt_releases_its_reservation(dbaas):
    dbaas.outcome["result"] = (False, True, False)  # region-fatal: next region
    assert _start() == (False, True)
    assert dbaas.attempts == ["centralus", "westeurope"]
    assert dbaas.outcome["held_during"] == [1, 1]
    assert avr.reserved("centralus", "IPv4BasicSkuPublicIpAddresses") == 0
    assert avr.reserved("westeurope", "IPv4BasicSkuPublicIpAddresses") == 0


def test_exception_releases_reservation(dbaas):
    dbaas.outcome["result"] = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        _start()
    assert avr.reserved("centralus", "IPv4BasicSkuPublicIpAddresses") == 0


def test_region_with_ips_used_up_by_other_stacks_is_skipped(dbaas):
    # centralus has 2 Basic IPs; two stacks of this job already hold them
    avr.reserve("centralus", FAMILY, 2)
    avr.reserve("centralus", FAMILY, 2)
    dbaas.outcome["result"] = (True, False, False)
    assert _start() == (True, True)
    assert dbaas.attempts == ["westeurope"]


def test_other_vendors_do_not_touch_the_ledger(dbaas, monkeypatch):
    dbaas.outcome["result"] = (True, False, False)
    started = dbaas_start.try_start_dbaas_inspect(
        None, None, "/tmp", "gcp", SimpleNamespace(cpu_count=16, instance_key="x"),
        [FakeDbaasTask()], ["centralus"], [], {}, 60, "", "", None, None, [],
    )
    assert started == (True, True)
    assert avr.reserved("centralus", "IPv4BasicSkuPublicIpAddresses") == 0

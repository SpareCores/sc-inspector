"""DBaaS client ranking: Azure Confidential VM sizes are never offered as clients."""

from types import SimpleNamespace

import companion_picker


def _stub(api_reference):
    return SimpleNamespace(api_reference=api_reference, cpu_allocation="DEDICATED")


def _rank(monkeypatch, vendor, refs):
    rows = [(_stub(ref), 0.1 + i / 100) for i, ref in enumerate(refs)]
    monkeypatch.setattr(
        companion_picker,
        "_eligible_dbaas_clients_with_prices",
        lambda *args, **kwargs: list(rows),
    )
    return [s.api_reference for s in companion_picker.rank_dbaas_client_instances(vendor, "westeurope", None)]


def test_azure_confidential_clients_are_dropped(monkeypatch):
    ranked = _rank(
        monkeypatch,
        "azure",
        ["Standard_D2als_v7", "Standard_DC2as_v5", "Standard_EC2ads_v5", "Standard_DC2es_v6", "Standard_D2als_v6"],
    )
    assert ranked == ["Standard_D2als_v7", "Standard_D2als_v6"]


def test_azure_sgx_sizes_are_not_confidential_vms(monkeypatch):
    # Intel SGX sizes (DCsv2/DCsv3) don't use the ConfidentialVM security type
    ranked = _rank(monkeypatch, "azure", ["Standard_DC1s_v2", "Standard_D2als_v7"])
    assert ranked == ["Standard_DC1s_v2", "Standard_D2als_v7"]


def test_other_vendors_unaffected(monkeypatch):
    # api_reference that would look confidential on Azure is irrelevant elsewhere
    ranked = _rank(monkeypatch, "gcp", ["Standard_DC2as_v5", "e2-standard-2"])
    assert ranked == ["Standard_DC2as_v5", "e2-standard-2"]

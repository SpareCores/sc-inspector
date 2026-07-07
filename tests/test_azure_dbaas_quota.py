import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from azure_dbaas_quota import (  # noqa: E402
    ComputeSkuInfo,
    QuotaEntry,
    _postgres_offer_restricted,
    _postgres_sku_listed,
    _quota_family_from_catalog_family,
    _quota_headroom,
    check_azure_client_vm_region,
    check_azure_dbaas_region,
    check_azure_postgres_region,
    sku_family_quota_name,
)

_CATALOG_FAMILIES = {
    "Standard_E16ds_v5": "EDSv5",
    "Standard_D4s_v3": "DSv3",
    "Standard_F16ams_v6": "StandardFamsv6",
    "Standard_B8als_v2": "Basv2",
}


def _mock_catalog(_vendor: str, sku: str) -> str | None:
    return _CATALOG_FAMILIES.get(sku)


def test_quota_family_from_catalog_family():
    assert _quota_family_from_catalog_family("DSv3") == "standardDSv3Family"
    assert _quota_family_from_catalog_family("StandardFamsv6") == "StandardFamsv6Family"
    assert _quota_family_from_catalog_family("Basv2") == "standardBasv2Family"


@patch("azure_dbaas_quota._catalog_server_family", side_effect=_mock_catalog)
def test_sku_family_mapping(_mock_catalog):
    assert sku_family_quota_name("Standard_E16ds_v5") == "standardEDSv5Family"
    assert sku_family_quota_name("Standard_D4s_v3") == "standardDSv3Family"
    assert sku_family_quota_name("Standard_F16ams_v6") == "StandardFamsv6Family"
    assert sku_family_quota_name("Standard_B8als_v2") == "standardBasv2Family"


@patch(
    "azure_dbaas_quota._compute_sku_info",
    return_value=ComputeSkuInfo(family="standardUnknownFamily", restrictions=[]),
)
@patch("azure_dbaas_quota._catalog_server_family", return_value=None)
def test_sku_family_mapping_arm_fallback(_mock_catalog, _mock_sku_info):
    assert (
        sku_family_quota_name(
            "Standard_Unknown_v9",
            subscription_id="00000000-0000-0000-0000-000000000001",
            region="northeurope",
        )
        == "standardUnknownFamily"
    )


def test_quota_headroom_passes():
    quotas = {
        "cores": QuotaEntry("cores", 0, 64),
        "standardEDSv5Family": QuotaEntry("standardEDSv5Family", 0, 32),
    }
    ok, reason = _quota_headroom(
        quotas,
        label="postgres",
        family="standardEDSv5Family",
        vcpus=16,
    )
    assert ok
    assert reason == ""


def test_quota_headroom_fails_family():
    quotas = {
        "cores": QuotaEntry("cores", 0, 64),
        "standardEDSv5Family": QuotaEntry("standardEDSv5Family", 17, 32),
    }
    ok, reason = _quota_headroom(
        quotas,
        label="postgres",
        family="standardEDSv5Family",
        vcpus=16,
    )
    assert not ok
    assert "standardEDSv5Family" in reason


def test_postgres_offer_restricted():
    caps = [
        {
            "supportedFeatures": [{"name": "OfferRestricted", "status": "Enabled"}],
            "supportedServerEditions": [],
        }
    ]
    assert _postgres_offer_restricted(caps)


def test_postgres_sku_listed():
    caps = [
        {
            "supportedServerEditions": [
                {
                    "supportedServerSkus": [{"name": "Standard_E16ds_v5", "vCores": 16}],
                }
            ]
        }
    ]
    assert _postgres_sku_listed(caps, "Standard_E16ds_v5")
    assert not _postgres_sku_listed(caps, "Standard_D4s_v3")


@patch("azure_dbaas_quota._catalog_server_family", side_effect=_mock_catalog)
@patch(
    "azure_dbaas_quota._compute_sku_info",
    return_value=ComputeSkuInfo(family="standardDSv3Family", restrictions=[]),
)
@patch("azure_dbaas_quota._vm_quotas")
@patch("azure_dbaas_quota._postgres_quotas")
@patch("azure_dbaas_quota._postgres_capabilities")
def test_check_azure_dbaas_region_ok(mock_caps, mock_pg_q, mock_vm_q, _mock_sku_info, _mock_catalog):
    mock_caps.return_value = [
        {
            "supportedServerEditions": [
                {"supportedServerSkus": [{"name": "Standard_E16ds_v5"}]},
            ],
        }
    ]
    mock_pg_q.return_value = {
        "cores": QuotaEntry("cores", 0, 256),
        "standardEDSv5Family": QuotaEntry("standardEDSv5Family", 0, 32),
    }
    mock_vm_q.return_value = {
        "cores": QuotaEntry("cores", 0, 64),
        "standardDSv3Family": QuotaEntry("standardDSv3Family", 0, 64),
    }
    ok, reason = check_azure_dbaas_region(
        "northeurope",
        pg_sku="Standard_E16ds_v5",
        pg_vcpus=16,
        vm_sku="Standard_D4s_v3",
        vm_vcpus=4,
        subscription_id="00000000-0000-0000-0000-000000000001",
    )
    assert ok
    assert reason == ""


@patch("azure_dbaas_quota._postgres_capabilities")
def test_check_azure_dbaas_region_offer_restricted(mock_caps):
    mock_caps.return_value = [
        {
            "supportedFeatures": [{"name": "OfferRestricted", "status": "Enabled"}],
            "supportedServerEditions": [],
        }
    ]
    ok, reason = check_azure_dbaas_region(
        "westeurope",
        pg_sku="Standard_E16ds_v5",
        pg_vcpus=16,
        vm_sku="Standard_D4s_v3",
        vm_vcpus=4,
        subscription_id="00000000-0000-0000-0000-000000000001",
    )
    assert not ok
    assert "offer restricted" in reason


@patch("azure_dbaas_quota._postgres_capabilities")
def test_check_azure_postgres_region_offer_restricted(mock_caps):
    mock_caps.return_value = [
        {
            "supportedFeatures": [{"name": "OfferRestricted", "status": "Enabled"}],
            "supportedServerEditions": [],
        }
    ]
    ok, reason = check_azure_postgres_region(
        "westeurope",
        pg_sku="Standard_E16ds_v5",
        pg_vcpus=16,
        subscription_id="00000000-0000-0000-0000-000000000001",
    )
    assert not ok
    assert "offer restricted" in reason


@patch("azure_dbaas_quota._catalog_server_family", side_effect=_mock_catalog)
@patch("azure_dbaas_quota._compute_sku_info")
@patch("azure_dbaas_quota._vm_quotas")
def test_check_azure_client_vm_region_sku_restricted(mock_vm_q, mock_sku_info, _mock_catalog):
    mock_sku_info.return_value = ComputeSkuInfo(
        family="standardBasv2Family",
        restrictions=[{"reasonCode": "NotAvailableForSubscription"}],
    )
    mock_vm_q.return_value = {
        "cores": QuotaEntry("cores", 0, 64),
        "standardBasv2Family": QuotaEntry("standardBasv2Family", 0, 64),
    }
    ok, reason = check_azure_client_vm_region(
        "northeurope",
        vm_sku="Standard_B8als_v2",
        vm_vcpus=8,
        subscription_id="00000000-0000-0000-0000-000000000001",
    )
    assert not ok
    assert "Standard_B8als_v2" in reason


def test_filter_clients_by_vm_quota_uses_region_index():
    from types import SimpleNamespace

    clients = [
        SimpleNamespace(api_reference="Standard_B8als_v2", vcpus=8),
        SimpleNamespace(api_reference="Standard_D4s_v3", vcpus=4),
    ]
    sku_index = {
        "Standard_B8als_v2": ComputeSkuInfo(
            family="standardBasv2Family",
            restrictions=[{"reasonCode": "NotAvailableForSubscription"}],
        ),
        "Standard_D4s_v3": ComputeSkuInfo(family="standardDSv3Family", restrictions=[]),
    }
    vm_quotas = {
        "cores": QuotaEntry("cores", 0, 64),
        "standardBasv2Family": QuotaEntry("standardBasv2Family", 0, 64),
        "standardDSv3Family": QuotaEntry("standardDSv3Family", 0, 64),
    }

    with patch("azure_dbaas_quota._subscription_id", return_value="sub"), patch(
        "azure_dbaas_quota._vm_quotas",
        return_value=vm_quotas,
    ), patch(
        "azure_dbaas_quota._region_compute_sku_index",
        return_value=sku_index,
    ), patch(
        "azure_dbaas_quota._catalog_server_family",
        side_effect=_mock_catalog,
    ):
        from azure_dbaas_quota import filter_clients_by_vm_quota

        eligible = filter_clients_by_vm_quota("azure", "northeurope", clients)

    assert [c.api_reference for c in eligible] == ["Standard_D4s_v3"]


if __name__ == "__main__":
    tests = [
        test_quota_family_from_catalog_family,
        test_sku_family_mapping,
        test_sku_family_mapping_arm_fallback,
        test_quota_headroom_passes,
        test_quota_headroom_fails_family,
        test_postgres_offer_restricted,
        test_postgres_sku_listed,
        test_check_azure_dbaas_region_ok,
        test_check_azure_dbaas_region_offer_restricted,
        test_check_azure_postgres_region_offer_restricted,
        test_check_azure_client_vm_region_sku_restricted,
        test_filter_clients_by_vm_quota_uses_region_index,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL PASSED")

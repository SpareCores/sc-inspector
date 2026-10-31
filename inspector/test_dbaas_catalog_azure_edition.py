"""Azure Flexible Server edition must be Sku.Tier, not database.family (series)."""

from dbaas_catalog import (
    _azure_edition_from_api_object,
    _azure_edition_from_native_id,
    _edition_for_row,
)


def test_azure_edition_from_gp_sku_name():
    assert (
        _azure_edition_from_api_object({"sku_name": "GP_Standard_D4ads_v5"})
        == "GeneralPurpose"
    )


def test_azure_edition_from_mo_and_burstable():
    assert (
        _azure_edition_from_api_object({"sku_name": "MO_Standard_E2ads_v5"})
        == "MemoryOptimized"
    )
    assert (
        _azure_edition_from_api_object({"sku_name": "B_Standard_B1ms"}) == "Burstable"
    )


def test_azure_edition_for_row_ignores_series_family():
    # family=Dadsv5 is the crawler series — must not become Sku.Tier.
    assert (
        _edition_for_row(
            "azure",
            "Dadsv5",
            native_id="Standard_D4ads_v5",
            api_reference_object={"sku_name": "GP_Standard_D4ads_v5"},
        )
        == "GeneralPurpose"
    )


def test_azure_edition_fallback_from_native_id():
    assert _azure_edition_from_native_id("Standard_D4ads_v5") == "GeneralPurpose"
    assert _azure_edition_from_native_id("Standard_B1ms") == "Burstable"
    assert _azure_edition_from_native_id("Standard_E2ads_v5") == "MemoryOptimized"
    assert _azure_edition_from_native_id("Standard_EC2ads_v5") == "MemoryOptimized"

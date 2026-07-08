from dbaas_catalog import available_managed_dbs
from dbaas_tiers import provision_spec


def test_available_managed_dbs_poc_azure():
    catalog = available_managed_dbs(vendor="azure")
    assert ("azure", "Standard_E16ds_v5/postgres/18/standalone") in catalog
    target, regions, zones, _ztr = catalog[("azure", "Standard_E16ds_v5/postgres/18/standalone")]
    assert target.cpu_count == 16
    assert "westeurope" in regions
    assert zones == []


def test_provision_spec_c100_c30():
    from dbaas_catalog import ManagedDbTarget

    target = ManagedDbTarget(
        vendor_id="azure",
        engine="postgres",
        native_id="Standard_E16ds_v5",
        sku_id="Standard_E16ds_v5:MemoryOptimized:18",
        engine_version="18",
        ha_mode="standalone",
        cpu_count=16,
        memory_gib=128,
        sync_commit_session_settable=False,
    )
    c100 = provision_spec(target, "c100")
    c30 = provision_spec(target, "c30")
    assert c100["storage_gib"] == 128
    assert c30["storage_gib"] == 251
    assert c100["iops_tier"] == "P30"
    assert c30["iops_tier"] == "P20"


def test_async_tasks_supported_on_azure_target():
    from dbaas_catalog import ManagedDbTarget
    from dbaas_tasks import (
        hammerdb_postgres_dbaas_oltp_mixed_c100,
        hammerdb_postgres_dbaas_oltp_mixed_durable_c100,
    )

    azure = ManagedDbTarget(
        vendor_id="azure",
        engine="postgres",
        native_id="Standard_E16ds_v5",
        sku_id="x",
        engine_version="18",
        ha_mode="standalone",
        cpu_count=16,
        memory_gib=128,
        sync_commit_session_settable=True,
    )
    assert hammerdb_postgres_dbaas_oltp_mixed_c100.supported_on_target(azure)
    assert hammerdb_postgres_dbaas_oltp_mixed_durable_c100.supported_on_target(azure)

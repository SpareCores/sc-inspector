import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from dbaas_catalog import available_managed_dbs  # noqa: E402
from dbaas_selector import stack_slug  # noqa: E402
from dbaas_tiers import provision_spec  # noqa: E402

GCP_KEY = ("gcp", "db-perf-optimized-N-16/postgres/18/standalone")


def test_gcp_catalog_has_zones():
    catalog = available_managed_dbs(vendor="gcp")
    assert GCP_KEY in catalog
    target, regions, zones, zone_to_region = catalog[GCP_KEY]
    assert target.native_id == "db-perf-optimized-N-16"
    assert target.sku_id == "db-perf-optimized-N-16:POSTGRES_18"
    assert target.edition == "PerformanceOptimized"
    assert target.sync_commit_session_settable is True
    assert regions
    assert zones
    assert zone_to_region["us-central1-a"] == "us-central1"


def test_gcp_provision_spec_uses_pd_ssd():
    catalog = available_managed_dbs(vendor="gcp")
    target = catalog[GCP_KEY][0]
    spec = provision_spec(target, "c100")
    assert spec["storage_type"] == "PD_SSD"
    assert spec["sku_name"] == "db-perf-optimized-N-16"


def test_gcp_stack_slug():
    catalog = available_managed_dbs(vendor="gcp")
    target = catalog[GCP_KEY][0]
    assert stack_slug(target, "c100") == "perfoptn16-pg18-c100"


if __name__ == "__main__":
    test_gcp_catalog_has_zones()
    test_gcp_provision_spec_uses_pd_ssd()
    test_gcp_stack_slug()
    print("ok")

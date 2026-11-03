"""SHUTDOWN_MINS floor for inspect user_data."""
from __future__ import annotations

from datetime import timedelta
from unittest import mock

import lib


def test_shutdown_mins_uses_boot_when_larger_than_catalog():
    with mock.patch.object(lib, "max_server_runtime", return_value=timedelta(hours=1)):
        # boot set 5h + 30m setup > catalog 1h + 2h slack
        assert lib.shutdown_mins_for_inspect("vultr", "vc2-8c-32gb", timeout_mins=300) == 330


def test_shutdown_mins_floors_at_catalog_plus_boot_slack():
    with mock.patch.object(lib, "max_server_runtime", return_value=timedelta(hours=4)):
        # boot set 61m + 30 = 91; catalog 4h + 2h slack = 360
        assert lib.shutdown_mins_for_inspect("vultr", "voc-c-16c-32gb-300s-amd", timeout_mins=61) == 360


def test_shutdown_mins_matches_cleanup_boot_slack_constant():
    assert lib.CLEANUP_BOOT_SLACK == timedelta(hours=2)


def test_dbaas_client_shutdown_mins_uses_boot_when_larger_than_dbaas_runtime():
    with mock.patch.object(lib, "max_dbaas_runtime", return_value=timedelta(minutes=30)):
        # boot set 5h + 30m setup > dbaas runtime 30m + 2h slack
        assert (
            lib.shutdown_mins_for_dbaas_client(
                "azure", "Standard_E4ads_v6/postgres/18/standalone", timeout_mins=300
            )
            == 330
        )


def test_dbaas_client_shutdown_mins_floors_at_dbaas_runtime_plus_boot_slack():
    with mock.patch.object(lib, "max_dbaas_runtime", return_value=timedelta(hours=2)):
        # boot set 10m + 30 = 40; dbaas runtime 2h + 2h slack = 240
        assert (
            lib.shutdown_mins_for_dbaas_client(
                "azure", "Standard_E4ads_v6/postgres/18/standalone", timeout_mins=10
            )
            == 240
        )


def test_dbaas_client_shutdown_mins_does_not_use_full_server_catalog():
    """Regression: DBaaS client SHUTDOWN_MINS must not float on max_server_runtime.

    A real azure/Standard_D2als_v7 client once got ``shutdown +2660`` (~44h) from
    the full multi-VM catalog's max_server_runtime instead of the ~2h-2.5h floor
    max_dbaas_runtime gives for the pgbench-only DBaaS benchmark - leaving a
    boot-failed DBaaS stack with no working absolute timeout at all.
    """
    with (
        mock.patch.object(lib, "max_dbaas_runtime", return_value=timedelta(hours=2)),
        mock.patch.object(lib, "max_server_runtime", return_value=timedelta(hours=40)),
    ):
        mins = lib.shutdown_mins_for_dbaas_client(
            "azure", "Standard_E4ads_v6/postgres/18/standalone", timeout_mins=10
        )
        assert mins == 240
        assert mins < 300  # far below what max_server_runtime(40h) would floor to

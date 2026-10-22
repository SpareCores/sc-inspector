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

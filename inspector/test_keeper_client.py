"""Tests for keeper inspector eligibility client and wiring hooks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import keeper_client
import pytest
from keeper_client import KeeperUnavailable


def test_server_task_block_reasons_parses_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "membench": [],
        "nvidia_smi": ["No GPUs available."],
    }
    with patch("keeper_client.requests.get", return_value=mock_resp) as get:
        out = keeper_client.server_task_block_reasons(
            "hcloud", "cx22", ["membench", "nvidia_smi"]
        )
    assert out["membench"] == []
    assert out["nvidia_smi"] == ["No GPUs available."]
    assert get.call_args.kwargs["params"] == [
        ("tasks", "membench"),
        ("tasks", "nvidia_smi"),
    ]
    assert "/inspector/server/hcloud/cx22/task_block_reason_codes" in get.call_args.args[0]


def test_database_task_block_reasons_parses_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "pgbench_postgres_dbaas_ro_durable": [
            "Database doesn't support single-node deployment."
        ]
    }
    with patch("keeper_client.requests.get", return_value=mock_resp):
        out = keeper_client.database_task_block_reasons(
            "ovh", "postgresql-advanced-b3-8", ["pgbench_postgres_dbaas_ro_durable"]
        )
    assert out["pgbench_postgres_dbaas_ro_durable"] == [
        "Database doesn't support single-node deployment."
    ]


def test_servers_bulk_nests_by_vendor_and_ref():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "hcloud": {"cx22": {"membench": []}},
        "aws": {"t3.medium": {"membench": ["Cloud credit/budget exhausted."]}},
    }
    with patch("keeper_client.requests.get", return_value=mock_resp):
        out = keeper_client.servers_task_block_reasons(["membench"])
    assert out["hcloud"]["cx22"]["membench"] == []
    assert out["aws"]["t3.medium"]["membench"] == ["Cloud credit/budget exhausted."]


def test_http_error_is_fail_closed():
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "boom"
    with (
        patch("keeper_client.requests.get", return_value=mock_resp),
        pytest.raises(KeeperUnavailable),
    ):
        keeper_client.server_task_block_reasons("hcloud", "cx22", ["membench"])


def test_task_allowed():
    assert keeper_client.task_allowed([]) is True
    assert keeper_client.task_allowed(["x"]) is False


def test_should_start_skips_when_keeper_blocks(tmp_path):
    import lib

    srv = MagicMock()
    srv.vendor_id = "hcloud"
    srv.api_reference = "cx22"
    srv.gpu_count = 0
    srv.vcpus = 2
    srv.memory_amount = 4 * 1024

    task = lib.DockerTask(
        parallel=False,
        image="x",
        command="true",
        start_with_instance=False,
        always_run=False,
    )
    task.name = "membench"

    with (
        patch("lib.load_task_meta", return_value=lib.Meta()),
        patch("lib.task_hash", return_value="abc"),
        patch("lib.is_unfinished_spot_meta", return_value=False),
    ):
        assert (
            lib.should_start(
                task,
                tmp_path,
                srv,
                block_reasons={"membench": ["No GPUs available."]},
            )
            is False
        )
        assert (
            lib.should_start(task, tmp_path, srv, block_reasons={"membench": []})
            is True
        )


def test_should_run_fail_closed_on_keeper_error(tmp_path):
    import lib

    task = lib.DockerTask(parallel=False, image="x", command="true")
    task.name = "membench"
    with patch(
        "lib._keeper_task_reasons",
        side_effect=KeeperUnavailable("down"),
    ):
        assert (
            lib.should_run(task, tmp_path, "hcloud", "cx22", 0.0, resource_kind="server")
            is False
        )


def test_should_run_skips_nonempty_reasons(tmp_path):
    import lib

    task = lib.DockerTask(parallel=False, image="x", command="true")
    task.name = "membench"
    with patch(
        "lib._keeper_task_reasons",
        return_value=["Benchmark limited to a few SKUs."],
    ):
        assert (
            lib.should_run(task, tmp_path, "hcloud", "cx22", 0.0, resource_kind="server")
            is False
        )

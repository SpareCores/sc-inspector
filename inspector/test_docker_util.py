"""Timeout helpers must match docker-py semantics (pull ignores client timeout)."""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest import mock

import pytest

import docker_util


def test_pull_image_raises_after_deadline():
    client = SimpleNamespace(images=SimpleNamespace(pull=lambda *_a, **_k: time.sleep(2)))
    with pytest.raises(TimeoutError, match="docker pull"):
        docker_util.pull_image(client, "busybox:latest", timeout_sec=0.2)


def test_pull_image_returns_on_success():
    image = object()
    client = SimpleNamespace(images=SimpleNamespace(pull=lambda repo, **_k: image))
    assert docker_util.pull_image(client, "busybox:latest", timeout_sec=5) is image


def test_stop_container_lowers_api_timeout_around_stop():
    seen = {}

    def stop(*, timeout):
        seen["grace"] = timeout
        seen["api_during"] = api.timeout

    api = SimpleNamespace(timeout=1800)
    container = SimpleNamespace(
        client=SimpleNamespace(api=api),
        stop=stop,
    )
    docker_util.stop_container(
        container, grace_sec=10, http_timeout_sec=60
    )
    assert seen["grace"] == 10
    assert seen["api_during"] == 50.0  # http_timeout - grace
    assert api.timeout == 1800  # restored


def test_run_attached_stops_on_wait_timeout():
    container = mock.Mock()
    container.wait.side_effect = Exception("ReadTimeout")
    container.client.api = SimpleNamespace(timeout=1800)
    container.stop = mock.Mock()
    container.remove = mock.Mock()
    container.logs.return_value = b""

    client = mock.Mock()
    client.containers.run.return_value = container

    with pytest.raises(TimeoutError, match="docker run"):
        docker_util.run_attached(client, "img", "true", timeout_sec=1)

    container.stop.assert_called()
    container.remove.assert_called_with(force=True)

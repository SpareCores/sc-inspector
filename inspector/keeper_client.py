"""HTTP client for Spare Cores keeper inspector task block-reason APIs."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from urllib.parse import quote

import requests

DEFAULT_KEEPER_API_BASE = "https://keeper.sparecores.net"
DEFAULT_TIMEOUT_SEC = 60

logger = logging.getLogger(__name__)


class KeeperUnavailable(Exception):
    """Raised when keeper cannot be reached or returns an unexpected response (fail-closed)."""


def keeper_api_base() -> str:
    return os.environ.get("KEEPER_API_BASE", DEFAULT_KEEPER_API_BASE).rstrip("/")


def _timeout() -> float:
    raw = os.environ.get("KEEPER_API_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SEC))
    try:
        return float(raw)
    except ValueError:
        return float(DEFAULT_TIMEOUT_SEC)


def _get_json(path: str, tasks: Iterable[str]) -> dict:
    task_list = list(tasks)
    if not task_list:
        return {}
    url = f"{keeper_api_base()}{path}"
    try:
        resp = requests.get(
            url,
            params=[("tasks", t) for t in task_list],
            timeout=_timeout(),
        )
    except requests.RequestException as e:
        raise KeeperUnavailable(f"keeper request failed: {url}: {e}") from e
    if resp.status_code != 200:
        raise KeeperUnavailable(
            f"keeper HTTP {resp.status_code} for {url}: {resp.text[:200]}"
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise KeeperUnavailable(f"keeper returned non-JSON for {url}") from e
    if not isinstance(data, dict):
        raise KeeperUnavailable(f"keeper returned unexpected JSON type for {url}")
    return data


def server_task_block_reasons(
    vendor: str, server: str, tasks: Iterable[str]
) -> dict[str, list[str]]:
    """Return task → reason list for one server."""
    path = (
        f"/inspector/server/{quote(vendor, safe='')}/"
        f"{quote(server, safe='')}/task_block_reason_codes"
    )
    data = _get_json(path, tasks)
    return {str(k): list(v) for k, v in data.items()}


def database_task_block_reasons(
    vendor: str, database: str, tasks: Iterable[str]
) -> dict[str, list[str]]:
    """Return task → reason list for one database (api_reference or id)."""
    path = (
        f"/inspector/database/{quote(vendor, safe='')}/"
        f"{quote(database, safe='')}/task_block_reason_codes"
    )
    data = _get_json(path, tasks)
    return {str(k): list(v) for k, v in data.items()}


def servers_task_block_reasons(
    tasks: Iterable[str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Bulk ACTIVE servers: vendor → api_reference → task → reasons."""
    data = _get_json("/inspector/servers/task_block_reason_codes", tasks)
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for vendor, servers in data.items():
        if not isinstance(servers, dict):
            continue
        out[str(vendor)] = {
            str(ref): {str(t): list(rs) for t, rs in task_map.items()}
            for ref, task_map in servers.items()
            if isinstance(task_map, dict)
        }
    return out


def databases_task_block_reasons(
    tasks: Iterable[str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Bulk ACTIVE databases: vendor → api_reference → task → reasons."""
    data = _get_json("/inspector/databases/task_block_reason_codes", tasks)
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for vendor, databases in data.items():
        if not isinstance(databases, dict):
            continue
        out[str(vendor)] = {
            str(ref): {str(t): list(rs) for t, rs in task_map.items()}
            for ref, task_map in databases.items()
            if isinstance(task_map, dict)
        }
    return out


def task_allowed(reasons: list[str] | None) -> bool:
    return not reasons


def log_block_reasons(task_name: str, vendor: str, resource: str, reasons: list[str]) -> None:
    logger.info(
        "Skipping task %s for %s/%s: keeper block reasons: %s",
        task_name,
        vendor,
        resource,
        "; ".join(reasons),
    )

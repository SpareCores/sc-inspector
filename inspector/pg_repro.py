"""Collect Postgres GUCs and related knobs for benchmark reproducibility."""

from __future__ import annotations

import json
import logging
from typing import Any


def pg_guc_settings_dict(guc_docker_args: list[str]) -> dict[str, str]:
    """Parse ``postgres -c name=value`` argv fragments into a name→value map."""
    out: dict[str, str] = {}
    i = 0
    while i < len(guc_docker_args):
        if guc_docker_args[i] == "-c" and i + 1 < len(guc_docker_args):
            raw = guc_docker_args[i + 1]
            name, sep, value = raw.partition("=")
            if sep:
                out[name] = value
            i += 2
            continue
        i += 1
    return out


def collect_postgres_repro(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    connect_kwargs: dict[str, Any] | None = None,
    requested_gucs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Query a live Postgres for settings useful when reproducing a run."""
    import psycopg2

    kwargs = dict(connect_kwargs or {})
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        connect_timeout=10,
        **kwargs,
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = str(cur.fetchone()[0])
            cur.execute("SHOW server_version")
            server_version = str(cur.fetchone()[0])
            cur.execute("SHOW server_version_num")
            server_version_num = int(cur.fetchone()[0])
            cur.execute("SELECT pg_is_in_recovery()")
            in_recovery = bool(cur.fetchone()[0])

            cur.execute(
                """
                SELECT name, setting, unit, current_setting(name) AS pretty,
                       category, short_desc, context, vartype,
                       source, pending_restart
                FROM pg_settings
                ORDER BY name
                """
            )
            settings: dict[str, str] = {}
            nondefault: dict[str, dict[str, Any]] = {}
            for (
                name,
                setting,
                unit,
                pretty,
                category,
                short_desc,
                context,
                vartype,
                source,
                pending_restart,
            ) in cur.fetchall():
                # Prefer SHOW-style values (e.g. 4GB) over raw unit counts
                # (e.g. 524288 with unit 8kB) so stdout is directly reusable
                # as postgresql.conf / -c arguments.
                settings[name] = pretty
                if source and source != "default":
                    entry: dict[str, Any] = {
                        "setting": pretty,
                        "source": source,
                        "context": context,
                        "vartype": vartype,
                        "category": category,
                    }
                    if unit:
                        entry["unit"] = unit
                        entry["setting_raw"] = setting
                    if short_desc:
                        entry["short_desc"] = short_desc
                    if pending_restart:
                        entry["pending_restart"] = True
                    nondefault[name] = entry

            cur.execute(
                """
                SELECT e.extname, e.extversion
                FROM pg_extension e
                ORDER BY e.extname
                """
            )
            extensions = [
                {"name": name, "version": ver} for name, ver in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT
                    COALESCE(r.rolname, 'All'),
                    COALESCE(d.datname, 'All'),
                    s.setconfig
                FROM pg_db_role_setting s
                LEFT JOIN pg_roles r ON r.oid = s.setrole
                LEFT JOIN pg_database d ON d.oid = s.setdatabase
                ORDER BY 1, 2
                """
            )
            role_settings = [
                {
                    "role": role,
                    "database": database,
                    "config": list(config or []),
                }
                for role, database, config in cur.fetchall()
            ]
    finally:
        conn.close()

    out: dict[str, Any] = {
        "version": version,
        "server_version": server_version,
        "server_version_num": server_version_num,
        "in_recovery": in_recovery,
        "settings": settings,
        "nondefault_settings": nondefault,
        "extensions": extensions,
        "role_settings": role_settings,
    }
    if requested_gucs:
        out["requested_gucs"] = requested_gucs
    return out


def merge_postgres_into_stdout(
    stdout: bytes | str,
    postgres: dict[str, Any] | None,
    *,
    extra: dict[str, Any] | None = None,
) -> bytes:
    """Embed ``postgres`` (and optional top-level ``extra``) into JSON stdout.

    Non-JSON or empty stdout is left unchanged when there is nothing to merge.
    On parse failure with payload to merge, wraps prior text under ``raw_stdout``.
    """
    if not postgres and not extra:
        if isinstance(stdout, bytes):
            return stdout
        return stdout.encode("utf-8")

    text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
    stripped = text.strip()
    summary: dict[str, Any]
    if stripped:
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                summary = {"raw_stdout": parsed}
            else:
                summary = parsed
        except json.JSONDecodeError:
            summary = {"raw_stdout": text}
    else:
        summary = {}

    if postgres:
        summary["postgres"] = postgres
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if key not in summary or summary[key] in (None, "", [], {}):
                summary[key] = value

    return (json.dumps(summary, indent=2) + "\n").encode("utf-8")


def safe_collect_postgres_repro(**kwargs: Any) -> dict[str, Any] | None:
    """Like :func:`collect_postgres_repro` but logs and returns None on failure."""
    try:
        return collect_postgres_repro(**kwargs)
    except Exception:
        logging.exception("Failed to collect Postgres repro settings")
        return None

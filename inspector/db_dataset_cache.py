"""CDN-backed benchmark database dataset cache.

Datasets are keyed by every parameter that affects the resulting schema and data.
When a matching dump exists on the CDN it is restored by streaming HTTPS → zstd → psql
(plain SQL dumps use PostgreSQL COPY FROM stdin blocks).  Otherwise the caller's
build function runs and the fresh dump is uploaded to S3 (served via CDN).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

CDN_PREFIX = "sc-inspector"
HAMMERDB_PARTITION_WAREHOUSE_THRESHOLD = 200


@dataclass(frozen=True)
class DatasetSpec:
    """Identifies a cached benchmark database dump."""

    tool: str
    workload: str
    filename: str

    @property
    def s3_key(self) -> str:
        return f"{CDN_PREFIX}/{self.filename}"


def hammerdb_tpcc_partition(warehouses: int) -> bool:
    """Match HammerDB sample buildschema scripts (partition when warehouses >= 200)."""
    return warehouses >= HAMMERDB_PARTITION_WAREHOUSE_THRESHOLD


def hammerdb_tpcc_filename(*, warehouses: int, storedprocs: bool = True) -> str:
    partition = hammerdb_tpcc_partition(warehouses)
    return (
        f"hammerdb-tpcc-wh{warehouses}-storedprocs-{str(storedprocs).lower()}"
        f"-partition-{str(partition).lower()}.sql.zst"
    )


def benchbase_filename(*, workload: str, scalefactor: int) -> str:
    return f"benchbase-{workload}-sf{scalefactor}.sql.zst"


def cdn_base_url() -> str:
    base = os.environ.get("SC_CDN_BASE_URL", "https://cdn.sparecores.net/sc-inspector").rstrip("/")
    return base


def cdn_url(spec: DatasetSpec) -> str:
    return f"{cdn_base_url()}/{spec.filename}"


def _parse_cdn_dataset_post() -> dict[str, Any] | None:
    raw = os.environ.get("SC_CDN_DATASET_POST_B64", "").strip()
    if not raw:
        return None
    try:
        post = json.loads(base64.b64decode(raw))
    except (json.JSONDecodeError, ValueError):
        logging.exception("Invalid SC_CDN_DATASET_POST_B64")
        return None
    prefix = post.get("prefix", "")
    if not prefix or "url" not in post or "fields" not in post:
        logging.warning("SC_CDN_DATASET_POST_B64 is missing url/fields/prefix")
        return None
    return post


def cdn_exists(url: str, *, timeout: int = 30) -> bool:
    """Return True when the CDN object exists (HTTP HEAD returns 200)."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        # CloudFront/S3 often return 403 (not 404) for missing private objects.
        if exc.code in (403, 404):
            return False
        raise


def _run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: int = 14400) -> None:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{detail}")


def _pg_env(password: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    return env


def _psql_base(host: str, port: int, user: str, password: str, dbname: str) -> list[str]:
    return [
        "psql",
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        user,
        "-d",
        dbname,
        "-v",
        "ON_ERROR_STOP=1",
    ]


def restore_from_cdn(
    url: str,
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    admin_db: str,
) -> None:
    """Stream a zstd-compressed plain SQL dump from HTTPS into psql.

    PostgreSQL does not support COPY FROM a remote URL without extensions.
    The fastest portable approach is to stream the dump over HTTPS on the client
    and feed it to psql, which executes COPY FROM stdin blocks in the dump.
    """
    _ensure_empty_database(host, port, user, password, admin_db, dbname)
    psql = _psql_base(host, port, user, password, dbname)
    env = _pg_env(password)
    sslmode = os.environ.get("SC_DB_SSLMODE", "prefer").strip() or "prefer"
    env["PGSSLMODE"] = sslmode
    shell = f"curl -fsSL {sh_quote(url)} | zstd -d | {' '.join(sh_quote(a) for a in psql)}"
    _run(["bash", "-c", shell], env=env)


def sh_quote(value: str) -> str:
    return subprocess.list2cmdline([value])


def _pg_dump_cmd(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
) -> tuple[list[str], dict[str, str]]:
    sslmode = os.environ.get("SC_DB_SSLMODE", "prefer").strip() or "prefer"
    env = _pg_env(password)
    env["PGSSLMODE"] = sslmode
    return (
        [
            "pg_dump",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            user,
            "-d",
            dbname,
            "--no-owner",
            "--no-privileges",
            "--format=plain",
        ],
        env,
    )


def _pipeline_stderr(proc: subprocess.Popen[bytes]) -> str:
    if proc.stderr is None:
        return ""
    return proc.stderr.read().decode("utf-8", errors="replace").strip()[-4000:]


def stream_dump_to_cdn(
    spec: DatasetSpec,
    post: dict[str, Any],
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
) -> bool:
    """Stream pg_dump → zstd → S3 presigned POST without buffering on local disk."""
    pg_dump_args, env = _pg_dump_cmd(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
    )
    pg_dump = subprocess.Popen(
        pg_dump_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if pg_dump.stdout is None:
        raise RuntimeError("pg_dump stdout unavailable")

    zstd = subprocess.Popen(
        ["zstd", "-T0", "-c"],
        stdin=pg_dump.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pg_dump.stdout.close()
    if zstd.stdout is None:
        zstd.kill()
        pg_dump.kill()
        raise RuntimeError("zstd stdout unavailable")

    key = spec.s3_key
    fields = dict(post["fields"])
    fields["key"] = key
    curl_args = ["curl", "-fsS", "-X", "POST", post["url"]]
    for field_name, field_value in fields.items():
        curl_args.extend(["-F", f"{field_name}={field_value}"])
    curl_args.extend(["-F", f"file=@-;filename={spec.filename};type=application/zstd"])

    curl = subprocess.Popen(
        curl_args,
        stdin=zstd.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    zstd.stdout.close()

    curl_rc = curl.wait()
    zstd_rc = zstd.wait()
    pg_dump_rc = pg_dump.wait()
    if curl_rc != 0 or zstd_rc != 0 or pg_dump_rc != 0:
        detail = "\n".join(
            part
            for part in (
                f"curl exit {curl_rc}: {_pipeline_stderr(curl)}",
                f"pg_dump exit {pg_dump_rc}: {_pipeline_stderr(pg_dump)}",
                f"zstd exit {zstd_rc}: {_pipeline_stderr(zstd)}",
            )
            if part
        )
        raise RuntimeError(f"dataset stream upload failed for {key}\n{detail}")

    logging.info("Stream-uploaded dataset %s via presigned POST", key)
    return True


def _ensure_empty_database(
    host: str,
    port: int,
    user: str,
    password: str,
    admin_db: str,
    dbname: str,
) -> None:
    import psycopg

    kwargs: dict[str, Any] = {"connect_timeout": 10, "autocommit": True}
    sslmode = os.environ.get("SC_DB_SSLMODE", "prefer").strip() or "prefer"
    if sslmode != "disable":
        kwargs["sslmode"] = sslmode

    with psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=admin_db,
        **kwargs,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (dbname,),
            )
            cur.execute("DROP DATABASE IF EXISTS %s" % _quote_ident(dbname))
            cur.execute("CREATE DATABASE %s" % _quote_ident(dbname))


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def cdn_upload_enabled() -> bool:
    if os.environ.get("SC_CDN_UPLOAD", "1").strip().lower() in {"0", "false", "no"}:
        return False
    return _parse_cdn_dataset_post() is not None


def prepare_database(
    spec: DatasetSpec,
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    admin_db: str,
    build: Callable[[], None],
) -> dict[str, Any]:
    """Restore from CDN when available, otherwise build and optionally upload."""
    url = cdn_url(spec)
    meta: dict[str, Any] = {"dataset": spec.filename, "cdn_url": url}

    if cdn_exists(url):
        logging.info("Restoring dataset from CDN: %s", url)
        restore_from_cdn(
            url,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            admin_db=admin_db,
        )
        meta["source"] = "cdn"
        return meta

    logging.info("CDN miss for %s; running native DB preparation", spec.filename)
    build()
    meta["source"] = "built"

    if not cdn_upload_enabled():
        logging.info("CDN upload skipped (presigned POST or SC_CDN_UPLOAD disabled)")
        meta["uploaded"] = False
        return meta

    post = _parse_cdn_dataset_post()
    assert post is not None
    try:
        meta["uploaded"] = stream_dump_to_cdn(
            spec,
            post,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
        )
    except Exception as exc:
        logging.exception("Dataset dump/upload failed for %s", spec.filename)
        meta["uploaded"] = False
        meta["upload_error"] = str(exc)
    return meta


def cdn_env_for_benchmark() -> dict[str, str]:
    """Env vars forwarded into benchmark containers for CDN cache I/O."""
    env: dict[str, str] = {
        "SC_CDN_BASE_URL": cdn_base_url(),
    }
    for key in (
        "SC_CDN_DATASET_POST_B64",
        "SC_CDN_UPLOAD",
        "SC_DB_SSLMODE",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value
    return env


def dataset_spec_for_hammerdb_tpcc(*, warehouses: int) -> DatasetSpec:
    return DatasetSpec(
        tool="hammerdb",
        workload="tpcc",
        filename=hammerdb_tpcc_filename(warehouses=warehouses),
    )


def dataset_spec_for_benchbase(*, workload: str, scalefactor: int) -> DatasetSpec:
    return DatasetSpec(
        tool="benchbase",
        workload=workload,
        filename=benchbase_filename(workload=workload, scalefactor=scalefactor),
    )

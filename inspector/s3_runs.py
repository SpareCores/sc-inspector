from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
import logging
import os
import re
from typing import Any

import boto3
from botocore.config import Config
import requests

# Presigned PUT URLs are signed with the GHA OIDC role (configure-aws-credentials).
# STS credentials cap URL lifetime at session duration (we request 12h in start.yml).
PRESIGN_EXPIRES_SECONDS = 12 * 60 * 60


INSPECTOR_RUNS_CATEGORY = "inspect"


def workflow_slug() -> str:
    workflow = os.environ.get("GITHUB_WORKFLOW", "inspector")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", workflow).strip("-").lower()
    return slug or "inspector"


def _sanitize_segment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _date_path(when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"{when.year}/{when.month:02d}/{when.day:02d}"


def log_key(vendor: str, instance: str, run_id: str, *, when: datetime | None = None) -> str:
    return (
        f"logs/{INSPECTOR_RUNS_CATEGORY}/{_sanitize_segment(vendor)}/{_sanitize_segment(instance)}/"
        f"{_date_path(when)}/{_sanitize_segment(run_id)}.log"
    )


def task_logs_prefix(vendor: str, instance: str, run_id: str, *, when: datetime | None = None) -> str:
    return (
        f"logs/{INSPECTOR_RUNS_CATEGORY}/{_sanitize_segment(vendor)}/{_sanitize_segment(instance)}/"
        f"{_date_path(when)}/{_sanitize_segment(run_id)}/tasks/"
    )


def run_key(vendor: str, instance: str, run_id: str, *, when: datetime | None = None) -> str:
    return (
        f"runs/{INSPECTOR_RUNS_CATEGORY}/{_sanitize_segment(vendor)}/{_sanitize_segment(instance)}/"
        f"{_date_path(when)}/{_sanitize_segment(run_id)}.json"
    )


def _s3_client():
    region = os.environ.get("INSPECTOR_RUNS_REGION", "us-east-1")
    return boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))


def bucket_name() -> str:
    name = os.environ.get("INSPECTOR_RUNS_BUCKET")
    if not name:
        raise RuntimeError("INSPECTOR_RUNS_BUCKET is not configured")
    return name


def presigned_put_url(key: str, *, content_type: str) -> str:
    return _s3_client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket_name(),
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=PRESIGN_EXPIRES_SECONDS,
    )


def presigned_post_for_prefix(prefix: str, *, max_bytes: int = 100 * 1024 * 1024) -> dict[str, Any]:
    # Key must end with ${filename} so boto3 emits a starts-with condition on prefix.
    # A bare prefix would add an exact key match and reject keys like prefix/task/stdout.
    return _s3_client().generate_presigned_post(
        Bucket=bucket_name(),
        Key=f"{prefix}${{filename}}",
        Conditions=[
            ["content-length-range", 1, max_bytes],
        ],
        ExpiresIn=PRESIGN_EXPIRES_SECONDS,
    )


def presigned_task_logs_post(
    vendor: str,
    instance: str,
    run_id: str,
    *,
    when: datetime | None = None,
) -> dict[str, Any]:
    prefix = task_logs_prefix(vendor, instance, run_id, when=when)
    post = presigned_post_for_prefix(prefix)
    return {"url": post["url"], "fields": post["fields"], "prefix": prefix}


def presigned_task_logs_post_b64(vendor: str, instance: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    when = datetime.now(timezone.utc)
    payload = presigned_task_logs_post(vendor, instance, run_id, when=when)
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _upload_presigned_post(path: str, key: str, post: dict[str, Any]) -> None:
    with open(path, "rb") as f:
        fields = dict(post["fields"])
        fields["key"] = key
        response = requests.post(post["url"], data=fields, files={"file": ("", f)}, timeout=600)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = (response.text or "").strip()
            if body:
                raise requests.HTTPError(f"{exc}; S3 response: {body}", response=response) from exc
            raise


def upload_task_logs_to_s3(data_dir: str) -> None:
    """Upload per-task stdout/stderr from the inspect data directory to S3."""
    raw = os.environ.get("TASK_LOGS_S3_POST_B64")
    if not raw:
        return
    try:
        post = json.loads(base64.b64decode(raw))
    except (json.JSONDecodeError, ValueError):
        logging.exception("Invalid TASK_LOGS_S3_POST_B64")
        return
    prefix = post.get("prefix", "")
    if not prefix or "url" not in post or "fields" not in post:
        logging.warning("TASK_LOGS_S3_POST_B64 is missing url/fields/prefix")
        return
    uploaded = 0
    for name in sorted(os.listdir(data_dir)):
        task_dir = os.path.join(data_dir, name)
        if not os.path.isdir(task_dir):
            continue
        safe_name = _sanitize_segment(name)
        for stream in ("stdout", "stderr"):
            path = os.path.join(task_dir, stream)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            key = f"{prefix}{safe_name}/{stream}"
            try:
                _upload_presigned_post(path, key, post)
                uploaded += 1
                logging.info("Uploaded task log %s", key)
            except Exception:
                logging.exception("Failed to upload task log %s", key)
    logging.info("Uploaded %d task log file(s) to S3 under %s", uploaded, prefix)


def presigned_urls_for_instance(vendor: str, instance: str) -> tuple[str, str]:
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    when = datetime.now(timezone.utc)
    log_url = presigned_put_url(log_key(vendor, instance, run_id, when=when), content_type="text/plain")
    run_url = presigned_put_url(
        run_key(vendor, instance, run_id, when=when),
        content_type="application/json",
    )
    return log_url, run_url


@dataclass(frozen=True)
class RunRecord:
    key: str
    vendor: str
    instance: str
    region: str
    zone: str
    workflow: str
    run_id: str
    terminated_at: datetime | None
    success: bool | None
    exit_code: int | None
    topology: str = ""
    instance_key: str = ""
    dbaas_slug: str = ""


def _parse_run_record(key: str, body: bytes) -> RunRecord | None:
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.exception("Invalid run record at %s", key)
        return None
    terminated_at = None
    if raw := data.get("terminated_at"):
        try:
            terminated_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            logging.warning("Invalid terminated_at in %s: %s", key, raw)
    return RunRecord(
        key=key,
        vendor=data.get("vendor", ""),
        instance=data.get("instance", ""),
        region=data.get("region", ""),
        zone=data.get("zone", ""),
        workflow=data.get("workflow", ""),
        run_id=str(data.get("run_id", "")),
        terminated_at=terminated_at,
        success=data.get("success"),
        exit_code=data.get("exit_code"),
        topology=data.get("topology", ""),
        instance_key=data.get("instance_key", ""),
        dbaas_slug=data.get("dbaas_slug", ""),
    )


def list_completed_runs(*, vendor: str | None = None) -> list[RunRecord]:
    client = _s3_client()
    bucket = bucket_name()
    prefix = "runs/"
    records: list[RunRecord] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            response = client.get_object(Bucket=bucket, Key=key)
            record = _parse_run_record(key, response["Body"].read())
            if not record or not record.terminated_at:
                continue
            if vendor and record.vendor != vendor:
                continue
            records.append(record)
    return records


def delete_run_record(key: str) -> None:
    _s3_client().delete_object(Bucket=bucket_name(), Key=key)
    logging.info("Deleted run record s3://%s/%s", bucket_name(), key)

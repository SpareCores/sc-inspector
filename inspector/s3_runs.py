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
CDN_DATASET_MAX_BYTES = 256 * 1024 * 1024 * 1024  # 256 GiB compressed dumps


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


def _cdn_s3_client():
    region = os.environ.get("SC_CDN_REGION", os.environ.get("CDN_REGION", "eu-central-1"))
    return boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))


def cdn_bucket_name() -> str:
    return os.environ.get("SC_CDN_BUCKET", os.environ.get("CDN_BUCKET", "sc-cdn-cae3awai"))


def cdn_dataset_prefix() -> str:
    prefix = os.environ.get("CDN_PREFIX", "sc-inspector").strip("/")
    return f"{prefix}/"


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


def presigned_cdn_dataset_post(*, max_bytes: int = CDN_DATASET_MAX_BYTES) -> dict[str, Any]:
    """Presigned POST for benchmark DB dumps under the CDN bucket prefix."""
    prefix = cdn_dataset_prefix()
    post = _cdn_s3_client().generate_presigned_post(
        Bucket=cdn_bucket_name(),
        Key=f"{prefix}${{filename}}",
        Conditions=[
            ["content-length-range", 1, max_bytes],
        ],
        ExpiresIn=PRESIGN_EXPIRES_SECONDS,
    )
    return {"url": post["url"], "fields": post["fields"], "prefix": prefix}


def presigned_cdn_dataset_post_b64() -> str:
    payload = presigned_cdn_dataset_post()
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


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


def _parse_task_logs_post() -> dict[str, Any] | None:
    raw = os.environ.get("TASK_LOGS_S3_POST_B64")
    if not raw:
        return None
    try:
        post = json.loads(base64.b64decode(raw))
    except (json.JSONDecodeError, ValueError):
        logging.exception("Invalid TASK_LOGS_S3_POST_B64")
        return None
    prefix = post.get("prefix", "")
    if not prefix or "url" not in post or "fields" not in post:
        logging.warning("TASK_LOGS_S3_POST_B64 is missing url/fields/prefix")
        return None
    return post


def upload_task_artifact(
    task_name: str,
    task_dir: str | os.PathLike,
    filename: str,
    *,
    post: dict[str, Any] | None = None,
    delete_after: bool = False,
) -> bool:
    """Upload one task output file via the inspect presigned POST (TASK_LOGS_S3_POST_B64)."""
    post = post if post is not None else _parse_task_logs_post()
    if not post:
        return False
    path = os.path.join(task_dir, filename)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    prefix = post["prefix"]
    key = f"{prefix}{_sanitize_segment(task_name)}/{filename}"
    try:
        _upload_presigned_post(path, key, post)
        logging.info("Uploaded task artifact %s", key)
        if delete_after:
            os.remove(path)
        return True
    except Exception:
        logging.exception("Failed to upload task artifact %s", key)
        return False


def upload_task_logs_to_s3(data_dir: str) -> None:
    """Upload per-task stdout/stderr and resource-tracker metrics from the inspect data directory to S3."""
    from resource_tracker import RESOURCE_TRACKER_OUTPUT_FILENAME

    post = _parse_task_logs_post()
    if not post:
        return
    prefix = post["prefix"]
    uploaded = 0
    for name in sorted(os.listdir(data_dir)):
        task_dir = os.path.join(data_dir, name)
        if not os.path.isdir(task_dir):
            continue
        for stream in ("stdout", "stderr", RESOURCE_TRACKER_OUTPUT_FILENAME):
            if upload_task_artifact(
                name,
                task_dir,
                stream,
                post=post,
                delete_after=(stream == RESOURCE_TRACKER_OUTPUT_FILENAME),
            ):
                uploaded += 1
    logging.info("Uploaded %d task artifact(s) to S3 under %s", uploaded, prefix)


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

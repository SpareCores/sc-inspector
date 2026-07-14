"""resource-tracker-rs integration for inspector benchmark containers."""

from __future__ import annotations

import copy
import os

RESOURCE_TRACKER_OUTPUT_FILENAME = "resource_tracker.jsonl"
RESOURCE_TRACKER_CONTAINER_MOUNT = "/inspector-output"
RESOURCE_TRACKER_CONTAINER_PATH = (
    f"{RESOURCE_TRACKER_CONTAINER_MOUNT}/{RESOURCE_TRACKER_OUTPUT_FILENAME}"
)


def uses_resource_tracker(env: dict | None) -> bool:
    """True when docker env carries resource-tracker metadata from tracker_docker_opts."""
    if not env:
        return False
    return bool(env.get("TRACKER_JOB_NAME") or env.get("TRACKER_PROJECT_NAME"))


def upload_resource_tracker_metrics(task_name: str, task_dir: str | os.PathLike) -> None:
    """Upload resource-tracker JSONL to S3 and remove the local copy (not committed to git)."""
    from s3_runs import upload_task_artifact

    upload_task_artifact(
        task_name,
        task_dir,
        RESOURCE_TRACKER_OUTPUT_FILENAME,
        delete_after=True,
    )


def configure_resource_tracker_docker_opts(
    docker_opts: dict,
    task_dir: str | os.PathLike,
) -> dict:
    """Mount task_dir and persist resource-tracker samples beside stdout/stderr."""
    opts = copy.deepcopy(docker_opts)
    env = dict(opts.get("environment") or {})
    if not uses_resource_tracker(env):
        return opts
    # Image ENV sets TRACKER_QUIET=true, which suppresses --output as well.
    env["TRACKER_QUIET"] = "false"
    env["TRACKER_OUTPUT"] = RESOURCE_TRACKER_CONTAINER_PATH
    opts["environment"] = env
    host_task_dir = os.path.abspath(task_dir)
    volumes = dict(opts.get("volumes") or {})
    volumes[host_task_dir] = {"bind": RESOURCE_TRACKER_CONTAINER_MOUNT, "mode": "rw"}
    opts["volumes"] = volumes
    return opts

import base64
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import cache
from itertools import chain
from queue import Queue
from typing import Any, ClassVar, Literal
from zlib import crc32

import docker
import repo
import transform
import user_data as user_data_pack
from pydantic import BaseModel
from resource_tracker import (
    configure_resource_tracker_docker_opts,
    upload_resource_tracker_metrics,
)

logger = logging.getLogger(__name__)

META_NAME = "meta.json"
TIMING_TASK_NAME = "timing"
SCHEDULE_DIRNAME = "schedule"
VARIANT_JSON_NAME = "variant.json"
# Task names must not collide with layout directories.
RESERVED_INSTANCE_DIRNAMES = frozenset({SCHEDULE_DIRNAME})
# UTC timestamps for instance startup intelligence (separate from meta.json to avoid git conflicts)
# Instance cloud-resource creation window (from Pulumi engine events / output), not full stack.up runtime
TIMING_API_START = "api_start"
TIMING_API_END = "api_end"
TIMING_MACHINE_START = "machine_start"
TIMING_INSPECTOR_START = "inspector_start"
TIMING_INSPECTOR_END = "inspector_end"
TIMING_USER_DATA_START = "user_data_start"
TIMING_USER_DATA_END = "user_data_end"
# Host path for user_data timestamps; mounted read-only at HOST_TIMING_MOUNT in inspect container
HOST_TIMING_BASE = "/var/lib/sparecores-inspector/timing"
HOST_TIMING_MOUNT = "/host-timing"
# add options to the task hash, whose function is to signal any
# changes in the tasks' runtime parameters, which might alter the output
TASK_HASH_KEYS = {"command", "transform_output", "image", "workload_proxy", "durability"}
# don't start task if it has already been started less than 2 hours ago
WAIT_SINCE_LAST_START = timedelta(hours=2)
# fail if a job has already started, but didn't produce output
FAIL_IF_NO_OUTPUT = timedelta(days=3)
FAIL_ON_ERROR = timedelta(days=3)
# stale no-output tasks may trigger a new instance this many times before giving up
MAX_START_RETRIES = 3
# task stopped triggering instance starts after MAX_START_RETRIES stale no-output generations
EXIT_CODE_STALE_RETRIES_EXHAUSTED = -5
# destroy the instance 15 mins after the last task has timed out
DESTROY_AFTER = timedelta(minutes=15)
# extra slack for slow cloud boot / user-data before inspect starts
CLEANUP_BOOT_SLACK = timedelta(hours=2)
# keep retrying Pulumi destroy after tasks finish
CLEANUP_DESTROY_RETRY = timedelta(hours=24)
DOCKER_OPTS = dict(detach=True, privileged=True, network_mode="host")
DOCKER_OPTS_GPU = dict(device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])])
# Postgres server + pgbench clients on large hosts: max_connections ×
# max_worker_processes needs tens of thousands of FDs (c3d-highcpu-360 ≈ 59k).
# memlock=-1 and unconfined seccomp unlock huge pages / io_uring.
DB_DOCKER_OPTS = DOCKER_OPTS | dict(
    security_opt=["seccomp=unconfined"],
    ulimits=[
        docker.types.Ulimit(name="nofile", soft=1048576, hard=1048576),
        docker.types.Ulimit(name="memlock", soft=-1, hard=-1),
    ],
)
# filter error_msg which is written to meta.json for these, we don't want to leak information
FILTER_ERROR_MSG = {
    re.compile(r"Submit a request for Quota increase at https.*to succeed\."),
}
# non-retryable Pulumi errors. The message that matches one of these regexes will be saved in meta.json.
PULUMI_ERRORS = {
    re.compile(r"error occurred"),   # AWS permanent error
    re.compile(r"creating .* error"),  # Azure errors
    re.compile(r"creating failed"),  # Azure errors
    re.compile(r"error waiting for.*to create"),   # GCP error
    re.compile(r"Unable to create server"),  # Upcloud error
    re.compile(r"\*\*creating failed\*\*"),  # Upcloud/Pulumi resource failure marker
    # Alicloud errors
    re.compile(r"The instanceType of the specified instance does not support this disk category"),
    re.compile(r"InvalidInstanceType"),  # instance type not available/not supported
    re.compile(r"beyond the permitted range"),  # instance type not permitted
    re.compile(r"QuotaExceeded\.Vpc"),  # hard quota limit for VPC creation
    re.compile(r"OperationDenied\.NoStock"),  # no capacity in selected zone
    re.compile(r"InvalidPeriod\.RegionDiscontinued"),  # region unavailable for ordering
    re.compile(r"InvalidParameter\.NotMatch"),  # instance/image incompatibility
    re.compile(r"only supports some specific images"),  # same as above, message form
    re.compile(r"InvalidInstanceType\.NotSupportDiskCategory"),  # exact provider code form
    re.compile(r"error creating server"),  # Vultr provider errors
    re.compile(r"\* error"),  # Pulumi bullet-list error details
}
# Generic Pulumi summary lines that wrap real provider errors.
PULUMI_ERROR_SUMMARY = re.compile(r"\d+\s+errors?\s+occurred:?\s*$", re.IGNORECASE)
PULUMI_BORING_ERRORS = {
    re.compile(r"^\s*error:\s*update failed\s*$", re.IGNORECASE),
}
_PROVIDER_JSON_ERROR = re.compile(r'\{"error":"([^"]+)"')
# Transient capacity/quota errors: retry on a later start run (possibly another region).
RETRYABLE_START_ERROR_MARKERS = (
    "InsufficientInstanceCapacity",
    "PublicIPCountLimitReached",
    "OperationNotAllowed",
    "exceeding approved",
    "QuotaExceeded",
    # GCP: Quota 'CPUS' / 'N2_CPUS' / 'NETWORKS' / … exceeded
    "Quota '",
    # UpCloud: CPU core quota of N exceeded … (type=SERVER_CORE_LIMIT_REACHED)
    "SERVER_CORE_LIMIT_REACHED",
    "CPU core quota",
)
# Region-level resource limits: unlike instance capacity, these block *every*
# instance type in the region, not just the one being tried. No point trying
# another candidate (e.g. a different DBaaS client SKU) in the same region -
# move on to the next region/zone immediately instead.
REGION_FATAL_ERROR_MARKERS = (
    "VpcLimitExceeded",
    "SubnetLimitExceeded",
    "InternetGatewayLimitExceeded",
    "NetworkAclLimitExceeded",
    "RouteTableLimitExceeded",
    "SecurityGroupLimitExceeded",
    "AddressLimitExceeded",
)
# Same-location Pulumi retries (retry_locked): only errors that may clear in seconds.
# Capacity shortages should fail over to the next region/zone immediately.
RETRYABLE_PULUMI_ERROR_MARKERS = tuple(
    m
    for m in RETRYABLE_START_ERROR_MARKERS
    if m != "InsufficientInstanceCapacity"
)
# provision machines with storage (GiB)
VOLUME_SIZE = 128

# Load USER_DATA script from external file
_USER_DATA_PATH = os.path.join(os.path.dirname(__file__), "user_data.sh")
with open(_USER_DATA_PATH, "r") as _f:
    USER_DATA = _f.read()



class Meta(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    exit_code: int | None = None
    error_msg: str | None = None
    task_hash: str | None = None
    version: str | None = None
    kernel_version: str | None = None
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    outputs: list[str] = []
    start_retries: int = 0
    # AWS only: True when the instance was requested as Spot; used to self-heal
    # unexpected Spot terminations by retrying on-demand after cleanup timeouts.
    spot: bool | None = None


class Task(BaseModel):
    vendors_only: set = set()  # run for these vendors only, empty means all
    servers_only: set[tuple] = set()  # run for these vendor/server pairs only, empty means all
    servers_exclude: set[tuple] = set()  # exclude these servers
    parallel: bool = False  # should we run this task concurrently with other tasks in the same priority group?
    priority: int | float = math.inf  # lower priority runs earlier, missing means last
    rollout: float = 1.0  # rollout percentage (between 0.0 and 1.0)
    version_command: str | list | None = None  # command to run to get the version
    command: str | list | None # command to run
    transform_output: list[Callable] = [transform.raw]  # functions to transform the output on the inspected node, write as raw if missing
    parse_output: list[Callable] = []  # functions to parse the already collected outputs from the repo
    rerun: timedelta | None = None  # re-run the task after a delay on successful execution, None means no re-evaluation
    timeout: timedelta = timedelta(minutes=30)  # timeout for the task
    name: str | None = None  # name of the task
    gpu: bool = False  # requires a machine with GPU(s)
    minimum_memory: float = 0  # minimum memory in GiBs for this test
    maximum_cpus: int = 0  # maximum CPUs for this test; 0 means no limit
    precheck_command: str | list | None = None  # check if we should run this task
    precheck_regex: str | None = None  # regular expression to match the precheck command's stdout
    # Never start an instance for this task alone; may still be included when another task starts.
    start_with_instance: bool = False
    # Always run on inspect when the instance was started for any task; never starts an instance alone.
    always_run: bool = False

    def feasible_on(self, vcpus: float, mem_gib: float, disk_gib: float | None) -> bool:
        return True


class DockerTask(Task):
    image: str
    docker_opts: dict = DOCKER_OPTS
    version_docker_opts: dict = {}


VLLM_PROBE_COMMAND = ["--probe-only"]
# Wall-clock cap per image probe (must exceed harness /health wait; see SERVER_START_TIMEOUT_PROBE_*).
VLLM_PROBE_TIMEOUT = timedelta(minutes=18)
# Current ghcr.io/sparecores/benchmark-vllm-gpu ships Ampere+ CUDA kernels only.
VLLM_MIN_GPU_COMPUTE_CAP: tuple[int, int] = (8, 0)


class VllmDockerTask(DockerTask):
    """Try each image in order (probe startup, then full benchmark on first success)."""

    images: list[str]
    image: str = "vllm/unified"
    probe_timeout: timedelta = VLLM_PROBE_TIMEOUT

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "image", "|".join(self.images))


class MultiVmDbTask(DockerTask):
    """Postgres benchmarks that use a companion client VM (priority band 1.x)."""

    MULTI_VM_PRIORITY_BAND: ClassVar[int] = 1
    command: str | list | None = None
    needs_companion: bool = True
    topology: Literal["multi_vm"] = "multi_vm"
    benchmark_family: str = ""
    workload_proxy: str = "read_heavy"
    durability: Literal["durable", "async"] = "async"

    def client_requirements(self, srv):
        from benchmark_tiers import client_req

        return client_req(srv)

    def disk_gib_required(self, srv) -> float:
        from db_storage import db_storage_plan

        mem_gib = float(srv.memory_amount) / 1024.0
        return float(db_storage_plan(srv.vendor_id, mem_gib, vcpus=int(srv.vcpus)).storage_gib)

    def feasible_on(self, vcpus: float, mem_gib: float, disk_gib: float | None) -> bool:
        from benchmark_tiers import mem_feasible

        del vcpus, disk_gib
        return mem_feasible(mem_gib)


class DbaasDbTask(DockerTask):
    """Postgres benchmarks against a vendor-managed database (priority band 1.x)."""

    command: str | list | None = None
    topology: Literal["dbaas"] = "dbaas"
    benchmark_family: str = ""
    workload_proxy: str = "read_heavy"
    durability: Literal["durable", "async"] = "durable"
    dbaas_only: set[tuple[str, str]] = set()

    def client_requirements(self, target):
        from benchmark_tiers import dbaas_client_req

        del target  # fixed sizing, independent of the DB target
        return dbaas_client_req()

    def disk_gib_required(self, target) -> float:
        from dbaas_tiers import provision_spec

        return float(provision_spec(target)["disk_gib_required"])

    def feasible_on(self, vcpus: float, mem_gib: float, disk_gib: float | None) -> bool:
        from benchmark_tiers import mem_feasible

        del vcpus, disk_gib
        return mem_feasible(mem_gib)

    def supported_on_target(self, target) -> bool:
        """False when this task's durability mode is unavailable on the managed DB."""
        if self.durability == "async" and target.sync_commit_session_settable is False:
            return False
        return True

    def supported_on_runtime(self) -> bool:
        """False when precheck/env shows async durability is unavailable."""
        if self.durability != "async":
            return True
        raw = os.environ.get("SC_PROVISION_SYNC_COMMIT_SETTABLE", "").strip().lower()
        return raw != "false"


def _parse_compute_cap(raw: str) -> tuple[int, int] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        major_s, minor_s = text.split(".", 1)
        return int(major_s), int(minor_s)
    except ValueError:
        return None


def _gpu_compute_capabilities() -> list[tuple[int, int]] | None:
    """Host GPU compute capabilities, or None when unknown.

    Prefers ``GPU_COMPUTE_CAP`` from user_data (inspector container has no GPU),
    then a local ``nvidia-smi`` if present.
    """
    raw = os.environ.get("GPU_COMPUTE_CAP", "").strip()
    if raw:
        caps: list[tuple[int, int]] = []
        for part in raw.replace(";", ",").split(","):
            parsed = _parse_compute_cap(part)
            if parsed is not None:
                caps.append(parsed)
        if caps:
            return caps
        logger.warning("Invalid GPU_COMPUTE_CAP=%r, ignoring", raw)

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    caps = []
    for line in out.splitlines():
        parsed = _parse_compute_cap(line)
        if parsed is not None:
            caps.append(parsed)
    return caps or None


def _vllm_gpu_image_supported(caps: list[tuple[int, int]] | None) -> bool:
    """Skip GPU image only when we know every GPU is below the minimum arch."""
    if not caps:
        return True
    return min(caps) >= VLLM_MIN_GPU_COMPUTE_CAP


def _vllm_image_label(image: str) -> str:
    """Short label for logs (image name without registry/tag)."""
    name = image.rsplit("/", 1)[-1]
    return name.split(":", 1)[0]


def _vllm_image_use_gpu(image: str) -> bool:
    return "benchmark-vllm-gpu" in image


def _vllm_image_amd64_only(image: str) -> bool:
    return "benchmark-vllm-cpu-avx2" in image


def _host_is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def _vllm_image_attempts(
    task: VllmDockerTask, accelerator_count: float
) -> list[tuple[str, str, bool]]:
    attempts: list[tuple[str, str, bool]] = []
    gpu_caps: list[tuple[int, int]] | None = None
    gpu_caps_checked = False
    for image in task.images:
        use_gpu = _vllm_image_use_gpu(image)
        if use_gpu and accelerator_count <= 0:
            continue
        if use_gpu:
            if not gpu_caps_checked:
                gpu_caps = _gpu_compute_capabilities()
                gpu_caps_checked = True
            if not _vllm_gpu_image_supported(gpu_caps):
                logger.info(
                    "Skipping %s: GPU compute capability %s < %s.%s "
                    "(Ampere+/sm_80 required)",
                    _vllm_image_label(image),
                    min(gpu_caps) if gpu_caps else "unknown",
                    VLLM_MIN_GPU_COMPUTE_CAP[0],
                    VLLM_MIN_GPU_COMPUTE_CAP[1],
                )
                continue
        if _vllm_image_amd64_only(image) and _host_is_arm64():
            continue
        attempts.append((_vllm_image_label(image), image, use_gpu))
    return attempts


def load_task_meta(task: Task, data_dir: str | os.PathLike, **kwargs) -> Meta:
    fn = os.path.join(data_dir, task.name, META_NAME)
    if os.path.exists(fn):
        with open(fn, "r") as f:
            meta = json.load(f)
        meta |= kwargs
        return Meta.model_validate(meta)
    else:
        logger.debug(f"{fn} not found, returning an empty Meta object")
        return Meta(**kwargs)


def task_hash(task: Task) -> str:
    h = hashlib.sha1()
    for var in sorted(TASK_HASH_KEYS):
        if not hasattr(task, var):
            continue
        h.update(var.encode("ascii"))
        value = getattr(task, var)
        if var == "transform_output":
            # add functions' source to the hash, so we re-run the task if its transformers change
            value = "".join([inspect.getsource(c) for c in value])
        h.update(str(value).encode("ascii"))

    return h.hexdigest()


@cache
def get_taskgroups(vendor: str) -> dict[tuple[float, bool], list[Task]]:
    import tasks
    taskgroups = defaultdict(list)
    for name, task in inspect.getmembers(tasks):
        if isinstance(task, (Task, DockerTask, VllmDockerTask, MultiVmDbTask)):
            # task name becomes the variable's name
            task.name = name.lower()
            validate_task_name(task.name)
            # only add the task if vendor is listed in vendors_only or if it's empty
            if vendor in task.vendors_only or not task.vendors_only:
                # execute parallel tasks first by negating the parallel option, so it gets forward during sorting
                taskgroups[(task.priority, not task.parallel)].append(task)
    import dbaas_tasks as dbaas_tasks_mod

    for name, task in inspect.getmembers(dbaas_tasks_mod):
        if isinstance(task, DbaasDbTask):
            task.name = name.lower()
            if vendor in task.vendors_only or not task.vendors_only:
                taskgroups[(task.priority, not task.parallel)].append(task)
    return taskgroups


@cache
def get_tasks(vendor: str) -> list[Task]:
    taskgroups = get_taskgroups(vendor)
    return list(chain(*taskgroups.values()))


def _server_in_servers_only(servers_only, vendor, instance, data_dir=None):
    """Check if (vendor, instance) is in servers_only."""
    return (vendor, instance) in servers_only


def _target_in_dbaas_only(dbaas_only, vendor, instance_key):
    return (vendor, instance_key) in dbaas_only


def _dbaas_api_reference_from_env() -> str:
    native = os.environ.get("SC_PROVISION_NATIVE_ID", "").strip()
    if native:
        return native
    key = os.environ.get("MANAGED_DB_INSTANCE_KEY", "").strip()
    return key.split("/", 1)[0] if key else ""


def _keeper_task_reasons(
    vendor: str,
    resource: str,
    task_name: str,
    *,
    kind: str,
    block_reasons: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return keeper block reasons for one task (empty list = allowed).

    ``block_reasons`` is an optional per-resource map from a bulk fetch.
    Raises ``keeper_client.KeeperUnavailable`` on transport/API failure.
    """
    import keeper_client

    if block_reasons is not None:
        return list(block_reasons.get(task_name, []))
    if kind == "database":
        return keeper_client.database_task_block_reasons(vendor, resource, [task_name]).get(
            task_name, []
        )
    return keeper_client.server_task_block_reasons(vendor, resource, [task_name]).get(
        task_name, []
    )


def _task_matches_target_for_start(
    task: Task,
    target,
    data_dir: str | os.PathLike,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> bool:
    import keeper_client

    try:
        reasons = _keeper_task_reasons(
            target.vendor_id,
            target.native_id,
            task.name,
            kind="database",
            block_reasons=block_reasons,
        )
    except keeper_client.KeeperUnavailable as e:
        logger.error("Keeper unavailable for DBaaS eligibility: %s", e)
        return False
    if reasons:
        keeper_client.log_block_reasons(task.name, target.vendor_id, target.native_id, reasons)
        return False
    if isinstance(task, DbaasDbTask):
        if task.dbaas_only and not _target_in_dbaas_only(
            task.dbaas_only, target.vendor_id, target.instance_key
        ):
            return False
        disk_gib = task.disk_gib_required(target)
        if not task.feasible_on(target.cpu_count, target.memory_gib, disk_gib):
            return False
        if not task.supported_on_target(target):
            return False
    return True


def _task_matches_server_for_start(
    task: Task,
    srv,
    data_dir: str | os.PathLike,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> bool:
    import keeper_client

    if isinstance(task, DbaasDbTask):
        return False
    try:
        reasons = _keeper_task_reasons(
            srv.vendor_id,
            srv.api_reference,
            task.name,
            kind="server",
            block_reasons=block_reasons,
        )
    except keeper_client.KeeperUnavailable as e:
        logger.error("Keeper unavailable for server eligibility: %s", e)
        return False
    if reasons:
        keeper_client.log_block_reasons(
            task.name, srv.vendor_id, srv.api_reference, reasons
        )
        return False
    if isinstance(task, MultiVmDbTask):
        from disk import effective_disk_gib

        disk_gib = effective_disk_gib(srv.vendor_id, srv, task.disk_gib_required(srv))
        if not task.feasible_on(srv.vcpus, srv.memory_amount / 1024, disk_gib):
            return False
    return True


def _is_stale_no_output(meta: Meta) -> bool:
    return bool(
        meta.start
        and (datetime.now() - meta.start) >= FAIL_IF_NO_OUTPUT
        and (meta.end is None or meta.exit_code is None)
    )


def boot_meta_for_task(task: Task, data_dir: str | os.PathLike, **fields) -> Meta:
    """Boot-time meta for a task, preserving start_retries from any prior meta."""
    prior = load_task_meta(task, data_dir)
    data = dict(start=datetime.now(), task_hash=task_hash(task), start_retries=prior.start_retries)
    data.update(fields)
    return Meta(**data)


def reconcile_stale_start_retries(
    vendor: str,
    data_dir: str | os.PathLike,
    srv,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> bool:
    """Reset or finalize tasks stuck with no output after FAIL_IF_NO_OUTPUT."""
    candidates = [
        task
        for task in get_tasks(vendor)
        if _task_matches_server_for_start(task, srv, data_dir, block_reasons=block_reasons)
        and _is_stale_no_output(load_task_meta(task, data_dir))
    ]
    if not candidates:
        return False
    repo.pull()
    changed = False
    now = datetime.now()
    for task in candidates:
        meta = load_task_meta(task, data_dir)
        if not _is_stale_no_output(meta):
            continue
        meta_path = os.path.join(data_dir, task.name, META_NAME)
        if meta.start_retries >= MAX_START_RETRIES:
            if meta.exit_code is not None:
                continue
            meta.end = now
            meta.exit_code = EXIT_CODE_STALE_RETRIES_EXHAUSTED
            meta.error_msg = (
                f"No output after {FAIL_IF_NO_OUTPUT.days} days; "
                f"exceeded {MAX_START_RETRIES} stale start retries"
            )
            logger.info(
                f"Task {task.name} exhausted stale start retries ({meta.start_retries}), "
                "will not trigger instance starts"
            )
        else:
            meta.start_retries += 1
            meta.start = None
            meta.end = None
            meta.exit_code = None
            meta.error_msg = None
            logger.info(
                f"Task {task.name} stale with no output; "
                f"retry {meta.start_retries}/{MAX_START_RETRIES}"
            )
        write_meta(meta, meta_path)
        changed = True
    if changed:
        repo.push_path(data_dir, f"Stale start retry from {repo.gha_url()}")
    return changed


def should_start(
    task: Task,
    data_dir: str | os.PathLike,
    srv,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> bool:
    """Return True if we should start a server for this task."""
    if isinstance(task, DbaasDbTask):
        return False
    meta = load_task_meta(task, data_dir)
    # Spot self-heal must be able to trigger a boot even for start_with_instance/always_run
    # tasks (e.g. vllm, llm): they never reach the scheduling checks below on their own, so
    # without this, an unfinished Spot run of one of these tasks past cleanup timeout could
    # never be retried on-demand unless some other, independently-triggering task on the
    # same instance also happened to be due for a run.
    spot_self_heal = bool(
        is_unfinished_spot_meta(meta)
        and force_ondemand_after_spot_timeout(srv.vendor_id, srv.api_reference, data_dir)
    )
    if (task.start_with_instance or task.always_run) and not spot_self_heal:
        logger.info(f"Skipping task {task.name}, does not trigger an instance start on its own")
        return False
    if meta.start and (datetime.now() - meta.start) <= WAIT_SINCE_LAST_START and not spot_self_heal:
        logger.info(f"Skipping task {task.name}, last start: {meta.start}")
        return False
    thash = task_hash(task)
    if not _task_matches_server_for_start(task, srv, data_dir, block_reasons=block_reasons):
        logger.info(
            f"Skipping task {task.name}: not eligible via keeper / multi-vm checks "
            f"for {srv.vendor_id}/{srv.api_reference}"
        )
        return False

    if meta.start:
        if (datetime.now() - meta.start) >= FAIL_ON_ERROR and meta.exit_code is not None and meta.exit_code > 0:
            raise RuntimeError(f"{task.name} was last started at {meta.start} and failed!")
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        # if rerun is set and there's a successful run, run the task again if rerun time interval has passed
        logger.info(f"Task {task.name} should be started: {task.rerun} has passed since last run")
        return True
    if meta.task_hash != thash:
        # This will be triggered when a task is first created or if it has been changed.
        logger.info(f"Task {task.name} should run as its task hash has changed: {meta.task_hash} -> {thash}")
        return True
    if meta.exit_code == -1 and is_retryable_start_error(meta.error_msg):
        logger.info(f"Retrying task {task.name} due to retryable start error, meta: {meta}")
        return True
    if spot_self_heal:
        logger.info(
            f"Retrying task {task.name} on-demand after Spot run passed cleanup timeout, meta: {meta}"
        )
        return True
    if not meta.start:
        logger.info(f"Task {task.name} should run, no start field")
        return True
    # Skip the task. If it was started, but hasn't yet produced output, we won't start a new run, as it would
    # ruin the above checks for failing outputs, and we don't want to constantly and silently restart the tasks.
    if meta.exit_code != 0:
        # don't report succeeded tasks
        logger.info(f"Skipping task {task.name}, meta: {meta}")
    return False


def _task_resource_checks(
    task: Task,
    data_dir: str | os.PathLike,
    vendor: str,
    instance: str,
    accelerator_count: float,
    mem_bytes: int,
    cpu_count: float | None = None,
) -> bool:
    """Live MultiVm / DBaaS feasibility (catalog SKU policy is enforced via keeper)."""
    if isinstance(task, MultiVmDbTask):
        from postgres_multi import multi_vm_supported

        if not multi_vm_supported(vendor):
            logger.info(f"Skipping task {task.name}: multi-VM not supported for {vendor}")
            return False
        provisioned = VOLUME_SIZE
        if os.environ.get("PROVISIONED_DISK_GIB"):
            provisioned = float(os.environ["PROVISIONED_DISK_GIB"])
        if not task.feasible_on(
            os.cpu_count() or 1,
            mem_bytes / 1024**3,
            provisioned,
        ):
            logger.info(f"Skipping task {task.name}: not feasible on live host")
            return False
    if isinstance(task, DbaasDbTask):
        mem_gib = float(os.environ.get("MEM_GIB") or mem_bytes / 1024**3)
        db_vcpus = float(os.environ.get("SC_PROVISION_CPU_COUNT", os.cpu_count() or 1))
        if os.environ.get("SC_PROVISION_STORAGE_GIB"):
            storage = float(os.environ["SC_PROVISION_STORAGE_GIB"])
        else:
            from dbaas_catalog import ManagedDbTarget
            from dbaas_tiers import provision_spec

            stub = ManagedDbTarget(
                vendor_id=vendor,
                engine="postgres",
                native_id=os.environ.get("SC_PROVISION_NATIVE_ID", instance),
                sku_id="",
                engine_version=os.environ.get("SC_PROVISION_ENGINE_VERSION", "18"),
                ha_mode="standalone",
                cpu_count=db_vcpus,
                memory_gib=mem_gib,
            )
            storage = float(provision_spec(stub)["storage_gib"])
        if not task.feasible_on(db_vcpus, mem_gib, storage):
            logger.info(f"Skipping task {task.name}: not feasible on DBaaS host")
            return False
        if not task.supported_on_runtime():
            logger.info(
                f"Skipping task {task.name}: async durability not supported on this managed DB"
            )
            return False
    return True


def task_data_dir(
    task: Task,
    data_dir: str | os.PathLike,
    client_data_dir: str | os.PathLike | None = None,
) -> str | os.PathLike:
    """Return the inspect data directory for a task.

  On multi-VM DB hosts, companion-client characterization may use ``client_data_dir``.
  DBaaS companions only run managed-DB benchmarks under ``data_dir`` (``dbaas/``).
    """
    if client_data_dir is None or isinstance(task, DbaasDbTask):
        return data_dir
    return client_data_dir


def should_run(
    task: Task,
    data_dir: str | os.PathLike,
    vendor: str,
    instance: str,
    accelerator_count: float,
    *,
    cpu_count: float | None = None,
    block_reasons: dict[str, list[str]] | None = None,
    resource_kind: str | None = None,
) -> bool:
    """Return True if we should run a task.

    On the spawned instance (inspect), omit ``cpu_count`` to use the live host.
    From the start orchestrator (GHA), pass catalog vCPUs so checks do not use
    the runner's ``os.cpu_count()``.
    """
    import keeper_client

    if isinstance(task, DbaasDbTask) and os.environ.get("TOPOLOGY") != "dbaas":
        logger.info(f"Skipping task {task.name}: TOPOLOGY is not dbaas")
        return False
    if os.environ.get("TOPOLOGY") == "dbaas" and not isinstance(task, DbaasDbTask):
        logger.info(f"Skipping task {task.name}: characterization tasks don't run on DBaaS topology")
        return False

    kind = resource_kind or (
        "database" if os.environ.get("TOPOLOGY") == "dbaas" else "server"
    )
    resource_ref = instance
    if kind == "database":
        resource_ref = _dbaas_api_reference_from_env() or instance
    try:
        reasons = _keeper_task_reasons(
            vendor,
            resource_ref,
            task.name,
            kind=kind,
            block_reasons=block_reasons,
        )
    except keeper_client.KeeperUnavailable as e:
        logger.error("Keeper unavailable; failing closed for %s: %s", task.name, e)
        return False
    if reasons:
        keeper_client.log_block_reasons(task.name, vendor, resource_ref, reasons)
        return False

    import psutil  # lazy load

    mem_bytes = psutil.virtual_memory().available
    if task.always_run:
        if not _task_resource_checks(
            task, data_dir, vendor, instance, accelerator_count, mem_bytes, cpu_count=cpu_count
        ):
            return False
        logger.info(f"Running {task.name} (always_run)")
        return True

    if not _task_resource_checks(
        task, data_dir, vendor, instance, accelerator_count, mem_bytes, cpu_count=cpu_count
    ):
        return False

    meta = load_task_meta(task, data_dir)
    thash = task_hash(task)
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        return True
    if meta.exit_code != 0 or meta.task_hash != thash:
        if meta.exit_code != 0:
            logger.info(f"Task {task.name} should run as last run has exit code: {meta.exit_code}")
        if meta.task_hash != thash:
            logger.info(f"Task {task.name} should run as its task hash has changed: {meta.task_hash} -> {thash}")
        return True
    logger.info(f"Skipping task {task.name}, {meta.end}, {meta.exit_code}")
    return False


def run_native(meta: Meta, task: Task, data_dir: str | os.PathLike) -> tuple[str | None, bytes, bytes]:
    ver = None
    try:
        if task.version_command:
            res = subprocess.run(task.version_command, capture_output=True, timeout=task.timeout.total_seconds())
            ver = res.stdout.strip().decode("utf-8").replace("\n", ", ")
    except Exception as e:
        meta.error_msg = str(e)
        return ver, b"", b""
    try:
        res = subprocess.run(task.command, capture_output=True, timeout=task.timeout.total_seconds())
        meta.end = datetime.now()
        meta.stdout_bytes = len(res.stdout)
        meta.stderr_bytes = len(res.stderr)
        meta.exit_code = res.returncode
        return ver, res.stdout, res.stderr
    except Exception as e:
        meta.error_msg = str(e)
        return ver, b"", b""


def container_remove(c):
    try:
        c.remove(force=True)
    except Exception:
        # don't fail if we couldn't remove the container
        pass


def docker_prune_after_round() -> None:
    """Reclaim disk under /var/lib/docker between inspect taskgroups."""
    if os.environ.get("DOCKER_PRUNE", "1").lower() in ("0", "false", "no"):
        return
    try:
        d = docker.from_env(timeout=120)
        containers = d.containers.prune()
        images = d.images.prune(filters={"dangling": False})
        networks = d.networks.prune()
        logger.info(
            "Docker prune after taskgroup: containers_deleted=%s images_space_reclaimed=%s networks_deleted=%s",
            containers.get("ContainersDeleted"),
            images.get("SpaceReclaimed"),
            networks.get("NetworksDeleted"),
        )
    except Exception:
        logger.exception("Docker prune failed (non-fatal)")


def _vllm_subtask_docker_opts(
    task: VllmDockerTask, use_gpu: bool
) -> dict:
    opts = copy.deepcopy(task.docker_opts)
    env = dict(opts.get("environment") or {})
    env["BENCHMARK_VLLM_MODE"] = "gpu" if use_gpu else "cpu"
    opts["environment"] = env
    if use_gpu:
        opts |= DOCKER_OPTS_GPU
    return opts


def _vllm_subtask(
    task: VllmDockerTask,
    image: str,
    *,
    command: list[str] | str | None,
    timeout: timedelta,
    use_gpu: bool,
) -> DockerTask:
    return DockerTask(
        image=image,
        command=command,
        docker_opts=_vllm_subtask_docker_opts(task, use_gpu),
        version_command=task.version_command,
        version_docker_opts=task.version_docker_opts,
        timeout=timeout,
    )


def _vllm_probe_image(
    task: VllmDockerTask,
    data_dir: str | os.PathLike,
    image: str,
    mode: str,
    use_gpu: bool,
    accelerator_count: float,
) -> tuple[bool, str]:
    """Return (ok, error_detail). Only tests server startup, not bench serve."""
    probe_meta = Meta(start=datetime.now())
    sub = _vllm_subtask(
        task,
        image,
        command=VLLM_PROBE_COMMAND,
        timeout=task.probe_timeout,
        use_gpu=use_gpu,
    )
    logger.info("vLLM probe mode=%s image=%s", mode, image)
    run_docker(probe_meta, sub, data_dir, accelerator_count if use_gpu else 0.0)
    if probe_meta.exit_code == 0:
        return True, ""
    detail = probe_meta.error_msg or f"exit_code={probe_meta.exit_code}"
    return False, detail


def run_vllm_docker(
    meta: Meta,
    task: VllmDockerTask,
    data_dir: str | os.PathLike,
    accelerator_count: float = 0.0,
) -> tuple[str | None, bytes, bytes]:
    attempts = _vllm_image_attempts(task, accelerator_count)
    probe_errors: list[str] = []
    selected: tuple[str, str, bool] | None = None

    for mode, image, use_gpu in attempts:
        ok, detail = _vllm_probe_image(
            task, data_dir, image, mode, use_gpu, accelerator_count
        )
        if ok:
            selected = (mode, image, use_gpu)
            logger.info("vLLM probe selected mode=%s image=%s", mode, image)
            break
        probe_errors.append(f"probe {mode}: {detail}")
        logger.info("vLLM probe failed mode=%s (%s), trying next", mode, detail)

    if not selected:
        meta.error_msg = "; ".join(probe_errors) or "No vLLM image passed startup probe"
        meta.exit_code = 1
        return None, b"", b""

    mode, image, use_gpu = selected
    meta.end = None
    meta.exit_code = None
    meta.error_msg = None
    meta.stdout_bytes = None
    meta.stderr_bytes = None

    sub = _vllm_subtask(
        task,
        image,
        command=task.command,
        timeout=task.timeout,
        use_gpu=use_gpu,
    )
    logger.info("vLLM full benchmark mode=%s image=%s", mode, image)
    ver, stdout, stderr = run_docker(
        meta, sub, data_dir, accelerator_count if use_gpu else 0.0
    )
    if meta.exit_code != 0:
        meta.error_msg = (
            f"benchmark failed on {mode} after successful probe"
            f" ({meta.error_msg or meta.exit_code})"
        )
    return ver, stdout, stderr


NVBANDWIDTH_BUFFER_MIB_DEFAULT = 512
NVBANDWIDTH_BUFFER_MIB_LOW_VRAM = 256
NVBANDWIDTH_LOW_VRAM_THRESHOLD_MIB = 2048


def nvbandwidth_command(vram_mib: int | None = None, accelerator_count: float = 0.0) -> str:
    """nvbandwidth CLI; bidirectional CE needs 3x buffer MiB on GPU (size + 2*size)."""
    buffer_mib = NVBANDWIDTH_BUFFER_MIB_DEFAULT
    if vram_mib is not None and vram_mib <= NVBANDWIDTH_LOW_VRAM_THRESHOLD_MIB:
        buffer_mib = NVBANDWIDTH_BUFFER_MIB_LOW_VRAM
    elif vram_mib is None and accelerator_count > 0 and accelerator_count != int(accelerator_count):
        # Fractional GPUs are typically low-VRAM; avoid -b 512 OOM when nvidia-smi fails.
        buffer_mib = NVBANDWIDTH_BUFFER_MIB_LOW_VRAM
    return f"nvbandwidth -j -b {buffer_mib}"


def _gpu_vram_mib() -> int | None:
    """VRAM from host user_data (GPU_VRAM_MIB); inspector container has no GPU."""
    raw = os.environ.get("GPU_VRAM_MIB", "").strip()
    if not raw:
        return None
    try:
        vram = int(raw)
        if vram <= 0:
            raise ValueError("non-positive VRAM")
        return vram
    except ValueError:
        logger.warning("Invalid GPU_VRAM_MIB=%r, ignoring host VRAM hint", raw)
        return None


def _resolve_docker_command(task: DockerTask, accelerator_count: float = 0.0) -> str | list | None:
    if task.name == "nvbandwidth":
        return nvbandwidth_command(_gpu_vram_mib(), accelerator_count=accelerator_count)
    return task.command


def run_docker(meta: Meta, task: DockerTask, data_dir: str | os.PathLike, accelerator_count: float = 0.0) -> tuple[str | None, bytes, bytes]:
    ver = None
    stdout = stderr = b""
    command = _resolve_docker_command(task, accelerator_count=accelerator_count)
    
    # Define the different docker options to try
    docker_options = []
    if accelerator_count > 0:
        # First try with GPU if available
        docker_opts = copy.deepcopy(task.docker_opts) | DOCKER_OPTS_GPU
        version_docker_opts = copy.deepcopy(task.version_docker_opts) | DOCKER_OPTS_GPU
        docker_options.append((docker_opts, version_docker_opts, True))
    
    # Then try with original options (with GPU if task.gpu is True)
    docker_opts = copy.deepcopy(task.docker_opts)
    version_docker_opts = copy.deepcopy(task.version_docker_opts)
    if task.gpu:
        docker_opts |= DOCKER_OPTS_GPU
        version_docker_opts |= DOCKER_OPTS_GPU
    docker_options.append((docker_opts, version_docker_opts, False))
    
    for docker_opts, version_docker_opts, is_gpu_attempt in docker_options:
        c = None
        try:
            d = docker.from_env(timeout=1800)
            image = d.images.pull(task.image)
            # Prefer immutable image reference (repo@sha256:...) over tags like :main/:latest.
            image_ref = next(iter(image.attrs.get("RepoDigests") or []), task.image)
            env = dict(docker_opts.get("environment") or {})
            env["TRACKER_CONTAINER_IMAGE"] = image_ref
            if hf_token := os.environ.get("HF_TOKEN"):
                env["HF_TOKEN"] = hf_token
            docker_opts["environment"] = env
            version_env = dict(version_docker_opts.get("environment") or {})
            version_env["TRACKER_CONTAINER_IMAGE"] = image_ref
            if hf_token := os.environ.get("HF_TOKEN"):
                version_env["HF_TOKEN"] = hf_token
            version_docker_opts["environment"] = version_env
            if task.version_command:
                ver = d.containers.run(task.image, task.version_command, **version_docker_opts).strip().decode("utf-8").replace("\n", ", ")
            docker_opts = configure_resource_tracker_docker_opts(docker_opts, data_dir)
            c = d.containers.run(task.image, command, **docker_opts)
        except Exception as e:
            if is_gpu_attempt:
                logger.info(f"GPU run failed, retrying without GPU: {e!s}")
                if c is not None:
                    container_remove(c)
                continue
            meta.error_msg = str(e)
            return ver, b"", b""

        ts = time.time() + task.timeout.total_seconds()
        while time.time() < ts:
            time.sleep(0.1)
            c.reload()
            if c.status == "exited":
                break
        else:
            # timed out, stop container, set error message
            c.stop()
            meta.error_msg = f"Execution timed out after {task.timeout.total_seconds()}s"
            if is_gpu_attempt:
                logger.info("GPU run timed out, retrying without GPU")
                container_remove(c)
                continue
            return ver, b"", b""
            
        meta.end = datetime.now()
        try:
            # wait for container exit/get output with 60s of docker timeout
            res = c.wait(timeout=60)
        except Exception as e:
            if is_gpu_attempt:
                logger.info(f"GPU run failed during wait, retrying without GPU: {e!s}")
                container_remove(c)
                continue
            meta.error_msg = str(e)
            return ver, b"", b""

        meta.exit_code = res["StatusCode"]
        stdout = c.logs(stdout=True, stderr=False)
        stderr = c.logs(stdout=False, stderr=True)
        meta.stdout_bytes = len(stdout)
        meta.stderr_bytes = len(stderr)
        container_remove(c)
        return ver, stdout, stderr
    
    # If we get here, all attempts failed
    meta.error_msg = "All docker run attempts failed"
    return ver, b"", b""


def write_meta(meta: Meta, file: str | os.PathLike) -> None:
    task_dir = os.path.dirname(file)
    os.makedirs(task_dir, exist_ok=True)
    with open(file, "w") as f:
        f.write(meta.model_dump_json())


def schedule_root(instance_root: str | os.PathLike) -> str:
    """Scheduler meta/timing root: ``{instance}/schedule``."""
    return os.path.join(instance_root, SCHEDULE_DIRNAME)


def is_variant_dirname(name: str) -> bool:
    return name.isdigit()


def variant_root(instance_root: str | os.PathLike, index: int) -> str:
    if index < 0:
        raise ValueError(f"variant index must be >= 0, got {index}")
    return os.path.join(instance_root, str(index))


def variant_roots(instance_root: str | os.PathLike) -> list[str]:
    """Sorted numeric variant directories under an instance root."""
    if not os.path.isdir(instance_root):
        return []
    roots = []
    for name in os.listdir(instance_root):
        path = os.path.join(instance_root, name)
        if os.path.isdir(path) and is_variant_dirname(name):
            roots.append(path)
    return sorted(roots, key=lambda p: int(os.path.basename(p)))


def validate_task_name(name: str) -> None:
    if name in RESERVED_INSTANCE_DIRNAMES or is_variant_dirname(name):
        raise ValueError(f"task name {name!r} collides with reserved instance path component")


def task_meta_path(task_root: str | os.PathLike, task: Task | str) -> str:
    name = task if isinstance(task, str) else task.name
    return os.path.join(task_root, name, META_NAME)


def load_schedule_meta(task: Task, instance_root: str | os.PathLike, **kwargs) -> Meta:
    return load_task_meta(task, schedule_root(instance_root), **kwargs)


def load_variant_meta(task: Task, instance_root: str | os.PathLike, index: int, **kwargs) -> Meta:
    return load_task_meta(task, variant_root(instance_root, index), **kwargs)


def _meta_sort_key(meta: Meta) -> tuple:
    """Newest-first key: prefer end, then start, then empty."""
    return (
        meta.end or datetime.min,
        meta.start or datetime.min,
    )


def select_schedule_meta(
    task: Task,
    instance_root: str | os.PathLike,
    *,
    active_root: str | os.PathLike | None = None,
    current_hash: str | None = None,
) -> Meta | None:
    """Materialized-union pick across variant metas for one task.

    Only ``active_root`` (the variant this VM is running on) may contribute an
    in-progress meta; unfinished metas in other variants are leftovers of dead
    runs and must not pin the schedule as running. Otherwise: current-hash
    success, else newest terminal current-hash, else newest available.
    """
    thash = current_hash if current_hash is not None else task_hash(task)
    active = os.path.normpath(active_root) if active_root is not None else None
    candidates: list[Meta] = []
    for root in variant_roots(instance_root):
        if not os.path.exists(task_meta_path(root, task)):
            continue
        meta = load_task_meta(task, root)
        if meta.start and meta.end is None:
            if os.path.normpath(root) == active:
                return meta
            continue
        candidates.append(meta)
    if not candidates:
        return None

    success = [m for m in candidates if m.exit_code == 0 and m.task_hash == thash and m.end]
    if success:
        return max(success, key=_meta_sort_key)

    terminal_hash = [m for m in candidates if m.task_hash == thash and m.end is not None]
    if terminal_hash:
        return max(terminal_hash, key=_meta_sort_key)

    return max(candidates, key=_meta_sort_key)


def refresh_schedule_meta(
    instance_root: str | os.PathLike,
    task: Task,
    *,
    active_root: str | os.PathLike | None = None,
    current_hash: str | None = None,
) -> Meta | None:
    """Recompute ``schedule/{task}/meta.json`` from all variant metas.

    Leaves the schedule meta untouched when no variant has a usable meta: start
    failures, stale-retry counters and ``-5`` live only in ``schedule/``.
    """
    selected = select_schedule_meta(
        task, instance_root, active_root=active_root, current_hash=current_hash
    )
    if selected is None:
        return None
    write_meta(selected, task_meta_path(schedule_root(instance_root), task))
    return selected


def refresh_all_schedule_metas(
    instance_root: str | os.PathLike,
    vendor: str,
    *,
    active_root: str | os.PathLike | None = None,
) -> None:
    for task in get_tasks(vendor):
        if isinstance(task, DbaasDbTask):
            continue
        refresh_schedule_meta(instance_root, task, active_root=active_root)


def _schedule_placeholder(task: Task, instance_root: str | os.PathLike) -> Meta | None:
    """This boot's unfinished schedule meta (boot placeholder or mirrored real start)."""
    sched = load_schedule_meta(task, instance_root)
    if sched.start and sched.end is None:
        return sched
    return None


def vm_task_meta(
    task: Task,
    task_dir_root: str | os.PathLike,
    instance_root: str | os.PathLike | None,
) -> Meta:
    """Meta the VM should transition for a task.

    Boot placeholders are written to ``schedule/`` only, so when the selected
    variant has no newer unfinished meta, start from the placeholder. That keeps
    ``spot``/``start_retries`` and lets ``-2``/``-4`` finalize the placeholder.
    """
    meta = load_task_meta(task, task_dir_root)
    if instance_root is None or (meta.start and meta.end is None):
        return meta
    placeholder = _schedule_placeholder(task, instance_root)
    if placeholder is not None and (not meta.start or placeholder.start > meta.start):
        return placeholder
    return meta


def write_variant_task_meta(
    meta: Meta,
    task: Task,
    variant_root_dir: str | os.PathLike,
    instance_root: str | os.PathLike | None,
) -> None:
    """Write task meta under the variant and refresh schedule when applicable."""
    if instance_root is not None:
        placeholder = _schedule_placeholder(task, instance_root)
        if placeholder is not None:
            if meta.spot is None:
                meta.spot = placeholder.spot
            meta.start_retries = max(meta.start_retries, placeholder.start_retries)
    write_meta(meta, task_meta_path(variant_root_dir, task))
    if instance_root is not None:
        refresh_schedule_meta(instance_root, task, active_root=variant_root_dir)


def probe_lscpu() -> tuple[dict[str, str], str | None, datetime]:
    """Run the lscpu task's image/command once; return (fields, lscpu version, end time)."""
    from variant_id import parse_lscpu_stdout

    client = docker.from_env()
    container = client.containers.run(
        "ghcr.io/sparecores/hwinfo:main",
        command=["bash", "-c", "lscpu --version | awk '{print $4}' >&2; lscpu -JB"],
        **DOCKER_OPTS,
    )
    try:
        result = container.wait()
        end = datetime.now()
        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
        if result.get("StatusCode", 1) != 0:
            raise RuntimeError(f"lscpu probe failed: {stderr[:500]}")
        version = stderr.strip().splitlines()[-1].strip() if stderr.strip() else None
        return parse_lscpu_stdout(stdout), version, end
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def select_or_create_variant(
    instance_root: str | os.PathLike,
    fields: dict[str, str] | None = None,
    *,
    lscpu_version: str | None = None,
    first_seen: datetime | None = None,
) -> tuple[int, str, str]:
    """Pick or create a numeric variant dir. Returns (index, root, variant_id)."""
    from variant_id import (
        build_variant_json,
        match_variant,
        variant_id_from_fields,
        write_variant_json,
    )

    if fields is None:
        fields, lscpu_version, first_seen = probe_lscpu()
    vid = variant_id_from_fields(fields)

    def _create(index: int) -> str:
        root = variant_root(instance_root, index)
        os.makedirs(root, exist_ok=True)
        data = build_variant_json(
            fields,
            source="observed",
            first_seen=(first_seen or datetime.now()).isoformat(),
            lscpu_version=lscpu_version,
        )
        write_variant_json(os.path.join(root, VARIANT_JSON_NAME), data)
        return root

    if not os.path.exists(os.path.join(variant_root(instance_root, 0), VARIANT_JSON_NAME)):
        return 0, _create(0), vid

    roots = variant_roots(instance_root)
    for root in roots:
        vpath = os.path.join(root, VARIANT_JSON_NAME)
        if not os.path.exists(vpath):
            continue
        with open(vpath, "r") as f:
            stored = json.load(f)
        if match_variant(fields, stored):
            return int(os.path.basename(root)), root, vid

    nxt = max(int(os.path.basename(r)) for r in roots) + 1
    return nxt, _create(nxt), vid


def parse_variant_outputs(instance_root: str | os.PathLike, tasks: list[Task]) -> list[str]:
    """Run parsers for every numeric variant's task outputs; return variant roots walked."""
    roots = variant_roots(instance_root)
    for root in roots:
        for task in tasks:
            if isinstance(task, DbaasDbTask):
                continue
            meta = load_task_meta(task, data_dir=root)
            if not meta.outputs:
                continue
            for parse_func in task.parse_output:
                parse_func(meta, task, os.path.join(root, task.name))
    return roots


def write_variant_marker(
    run_dir: str | os.PathLike,
    index: int,
    variant_id: str,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "variant")
    with open(path, "w") as f:
        f.write(json.dumps({"n": index, "id": variant_id}) + "\n")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_timing_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def github_run_id() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if not run_id:
        logger.warning("GITHUB_RUN_ID not set, using 'unknown' for timing paths")
        return "unknown"
    return run_id


def host_timing_dir(vendor: str, instance: str, run_id: str | None = None) -> str:
    return os.path.join(HOST_TIMING_BASE, vendor, instance, run_id or github_run_id())


def timing_dir(data_dir: str | os.PathLike) -> str:
    return os.path.join(data_dir, TIMING_TASK_NAME)


def timing_run_dir(data_dir: str | os.PathLike, run_id: str | None = None) -> str:
    return os.path.join(timing_dir(data_dir), run_id or github_run_id())


def timing_file_path(data_dir: str | os.PathLike, filename: str, run_id: str | None = None) -> str:
    return os.path.join(timing_run_dir(data_dir, run_id), filename)


def prepare_variant_timing(
    instance_root: str | os.PathLike,
    variant_root_dir: str | os.PathLike,
    index: int,
    variant_id: str,
    *,
    run_id: str | None = None,
) -> None:
    """Copy schedule timing api_* into the variant run dir; write variant markers."""
    rid = run_id or github_run_id()
    src = timing_run_dir(schedule_root(instance_root), rid)
    dst = timing_run_dir(variant_root_dir, rid)
    os.makedirs(dst, exist_ok=True)
    os.makedirs(src, exist_ok=True)
    if os.path.isdir(src):
        for name in (TIMING_API_START, TIMING_API_END):
            sp = os.path.join(src, name)
            if os.path.isfile(sp):
                with open(sp, "rb") as f:
                    data = f.read()
                with open(os.path.join(dst, name), "wb") as f:
                    f.write(data)
    write_variant_marker(dst, index, variant_id)
    write_variant_marker(src, index, variant_id)


def write_timing_file(
    data_dir: str | os.PathLike,
    filename: str,
    when: datetime | None = None,
    *,
    run_id: str | None = None,
) -> None:
    """Write a single UTC timestamp (ISO-8601 Z) under timing/<GITHUB_RUN_ID>/."""
    path = timing_file_path(data_dir, filename, run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(format_timing_utc(when or utc_now()))
        f.write("\n")


def machine_boot_time_utc() -> datetime:
    with open("/proc/uptime") as f:
        uptime_seconds = int(float(f.read().split()[0]))
    return utc_now() - timedelta(seconds=uptime_seconds)


def record_timing_machine_start(data_dir: str | os.PathLike, *, run_id: str | None = None) -> None:
    write_timing_file(data_dir, TIMING_MACHINE_START, machine_boot_time_utc(), run_id=run_id)


def record_timing_api(
    data_dir: str | os.PathLike,
    api_start: datetime,
    api_end: datetime,
    *,
    run_id: str | None = None,
) -> None:
    """Write instance resource create start/end (Pulumi ResourcePre/Outputs events)."""
    write_timing_file(data_dir, TIMING_API_START, api_start, run_id=run_id)
    write_timing_file(data_dir, TIMING_API_END, api_end, run_id=run_id)


def record_timing_inspector_start(data_dir: str | os.PathLike, *, run_id: str | None = None) -> None:
    write_timing_file(data_dir, TIMING_INSPECTOR_START, run_id=run_id)


def record_timing_inspector_end(data_dir: str | os.PathLike, *, run_id: str | None = None) -> None:
    write_timing_file(data_dir, TIMING_INSPECTOR_END, run_id=run_id)


def parse_timing_utc(text: str) -> datetime:
    text = text.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def record_timing_from_host(
    data_dir: str | os.PathLike,
    host_dir: str | os.PathLike | None = None,
    *,
    run_id: str | None = None,
) -> None:
    """Copy user_data timestamps from the host mount into timing/<run_id>/."""
    host_dir = host_dir or HOST_TIMING_MOUNT
    for filename in (TIMING_USER_DATA_START, TIMING_USER_DATA_END):
        src = os.path.join(host_dir, filename)
        if not os.path.isfile(src):
            logger.info(f"Host timing file not found: {src}")
            continue
        with open(src) as f:
            ts = f.read().strip()
        try:
            when = parse_timing_utc(ts)
        except ValueError:
            logger.exception(f"Invalid timestamp in {src}: {ts!r}")
            continue
        write_timing_file(data_dir, filename, when, run_id=run_id)


def record_timing_metrics(data_dir: str | os.PathLike, *, run_id: str | None = None) -> None:
    """Collect remote timing checkpoints for this GHA run; always overwrites."""
    run_id = run_id or github_run_id()
    record_timing_from_host(data_dir, run_id=run_id)
    record_timing_machine_start(data_dir, run_id=run_id)


def tasks_to_start(
    vendor: str,
    data_dir: str | os.PathLike,
    srv,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> list[Task]:
    """Tasks that should trigger starting an instance.

    Tasks with start_with_instance or always_run never start an instance on their own, but are
    added here when another task is already starting the machine.
    """
    reconcile_stale_start_retries(vendor, data_dir, srv, block_reasons=block_reasons)
    tasks = [
        task
        for task in get_tasks(vendor)
        if should_start(task, data_dir, srv, block_reasons=block_reasons)
    ]
    if not tasks:
        return []
    started = {id(task) for task in tasks}
    for task in get_tasks(vendor):
        if id(task) in started:
            continue
        if isinstance(task, DbaasDbTask):
            continue
        if task.always_run:
            logger.info(f"Adding {task.name} (always_run) for co-start")
            tasks.append(task)
            started.add(id(task))
        elif task.start_with_instance and should_run(
            task,
            data_dir,
            srv.vendor_id,
            srv.api_reference,
            srv.accelerator_count,
            cpu_count=srv.vcpus,
            block_reasons=block_reasons,
            resource_kind="server",
        ):
            logger.info(f"Adding {task.name} (start_with_instance) for co-start")
            tasks.append(task)
            started.add(id(task))
    return tasks


def should_start_dbaas(
    task: Task,
    data_dir: str | os.PathLike,
    target,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> bool:
    """Return True if we should start a DBaaS stack for this task."""
    if not isinstance(task, DbaasDbTask):
        return False
    if task.start_with_instance or task.always_run:
        return False
    meta = load_task_meta(task, data_dir)
    if meta.start and (datetime.now() - meta.start) <= WAIT_SINCE_LAST_START:
        return False
    if not _task_matches_target_for_start(
        task, target, data_dir, block_reasons=block_reasons
    ):
        return False
    thash = task_hash(task)
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        return True
    if meta.task_hash != thash:
        return True
    if meta.exit_code is None or meta.exit_code != 0:
        return True
    if _is_stale_no_output(meta):
        return True
    return False


def tasks_to_start_dbaas(
    vendor: str,
    data_dir: str | os.PathLike,
    target,
    *,
    block_reasons: dict[str, list[str]] | None = None,
) -> list[Task]:
    """DBaaS tasks that should trigger starting a managed DB + client stack."""
    tasks = [
        task
        for task in get_tasks(vendor)
        if should_start_dbaas(task, data_dir, target, block_reasons=block_reasons)
    ]
    if not tasks:
        return []
    started = {id(task) for task in tasks}
    for task in get_tasks(vendor):
        if id(task) in started:
            continue
        if isinstance(task, DbaasDbTask) and task.always_run:
            tasks.append(task)
            started.add(id(task))
        elif isinstance(task, DbaasDbTask) and task.start_with_instance and should_run(
            task,
            data_dir,
            vendor,
            target.native_id,
            0.0,
            cpu_count=target.cpu_count,
            block_reasons=block_reasons,
            resource_kind="database",
        ):
            tasks.append(task)
            started.add(id(task))
    return tasks


def run_task(
    q: Queue,
    data_dir: str | os.PathLike,
    accelerator_count: float = 0.0,
    client_data_dir: str | os.PathLike | None = None,
    instance_root: str | os.PathLike | None = None,
) -> None:
    while True:
        try:
            task = q.get()
            if not task:
                break
            task_dir_root = task_data_dir(task, data_dir, client_data_dir)
            meta = boot_meta_for_task(task, task_dir_root, kernel_version=platform.release())
            failed = False
            task_dir = os.path.join(task_dir_root, task.name)
            os.makedirs(task_dir, exist_ok=True)
            if task.name == TIMING_TASK_NAME:
                try:
                    record_timing_metrics(task_dir_root)
                    meta.end = datetime.now()
                    meta.exit_code = 0
                    meta.version = "n/a"
                except Exception as e:
                    failed = True
                    meta.exit_code = 256
                    meta.error_msg = str(e)
                write_variant_task_meta(meta, task, task_dir_root, instance_root)
                continue
            try:
                if isinstance(task, VllmDockerTask):
                    ver, stdout, stderr = run_vllm_docker(
                        meta, task, task_dir, accelerator_count
                    )
                elif isinstance(task, MultiVmDbTask):
                    from postgres_multi import run_multi_vm_task

                    ver, stdout, stderr = run_multi_vm_task(
                        meta, task, task_dir_root, accelerator_count
                    )
                elif isinstance(task, DbaasDbTask):
                    from postgres_dbaas import run_dbaas_task

                    ver, stdout, stderr = run_dbaas_task(
                        meta, task, task_dir_root, accelerator_count
                    )
                elif isinstance(task, DockerTask):
                    ver, stdout, stderr = run_docker(meta, task, task_dir, accelerator_count)
                else:
                    ver, stdout, stderr = run_native(meta, task, task_dir)
                meta.version = ver
            except Exception as e:
                failed = True
                # return something positive (negative will be inspector start errors) and outside normal return codes
                meta.exit_code = 256
                meta.error_msg = str(e)
            if not failed:
                for t in task.transform_output:
                    meta.outputs.extend(t(meta, task, task_dir, stdout, stderr))
            # S3-only bulky artifacts: upload+delete before the taskgroup git push.
            upload_resource_tracker_metrics(task.name, task_dir)
            from s3_runs import upload_postgres_log

            upload_postgres_log(task.name, task_dir)
            write_variant_task_meta(meta, task, task_dir_root, instance_root)
        except Exception:
            raise
        finally:
            # ack the task, so run_tasks won't wait forever
            q.task_done()


def _finalize_multi_vm_band_if_done(
    ordered_groups: list[tuple[float, bool]],
    group_index: int,
    data_dir: str | os.PathLike,
) -> None:
    """Power off the companion client after the last task in the multi-VM priority band."""
    if os.environ.get("TOPOLOGY") == "dbaas":
        return
    priority = ordered_groups[group_index][0]
    if not math.isfinite(priority) or math.floor(priority) != MultiVmDbTask.MULTI_VM_PRIORITY_BAND:
        return
    next_priority = (
        ordered_groups[group_index + 1][0]
        if group_index + 1 < len(ordered_groups)
        else None
    )
    next_floor = (
        math.floor(next_priority)
        if next_priority is not None and math.isfinite(next_priority)
        else None
    )
    if next_floor is not None and next_floor <= MultiVmDbTask.MULTI_VM_PRIORITY_BAND:
        return
    try:
        from postgres_multi import finalize_multi_vm, has_companion_client

        if not has_companion_client():
            return
        finalize_multi_vm(data_dir)
    except Exception:
        logger.exception("multi-VM companion shutdown failed")


def _push_data_dirs(data_dirs: list[str | os.PathLike], message: str) -> None:
    for data_dir in dict.fromkeys(data_dirs):
        for i in range(3):
            try:
                repo.push_path(data_dir, message)
                break
            except Exception:
                logger.exception("push failed")
                if i < 2:
                    time.sleep(random.randint(1, 10))


def run_tasks(
    vendor,
    data_dir: str | os.PathLike,
    instance: str,
    accelerator_count: float = 0.0,
    nthreads: int = 8,
    client_data_dir: str | os.PathLike | None = None,
    instance_root: str | os.PathLike | None = None,
):
    taskgroups = get_taskgroups(vendor)

    def _push_roots(roots: list[str | os.PathLike], message: str) -> None:
        push = list(roots)
        if instance_root is not None:
            push.append(schedule_root(instance_root))
        _push_data_dirs(push, message)

    # initialize thread pool
    q: Queue = Queue(maxsize=nthreads)
    threads = []
    for _ in range(nthreads):
        threads.append(
            threading.Thread(
                target=run_task,
                args=(q, data_dir, accelerator_count, client_data_dir, instance_root),
                daemon=True,
            )
        )
        threads[-1].start()

    # iterate over tasks, sorted by task key (running parallel tasks in a group first, then
    # non-parallel ones)
    ordered_groups = sorted(taskgroups.keys())
    for group_index, taskgroup in enumerate(ordered_groups):
        changed_dirs: list[str | os.PathLike] = []
        for task in taskgroups[taskgroup]:
            task_dir_root = task_data_dir(task, data_dir, client_data_dir)
            meta = vm_task_meta(task, task_dir_root, instance_root)
            if not should_run(task, task_dir_root, vendor, instance, accelerator_count):
                if meta.start and meta.exit_code is None:
                    # update meta, if it doesn't yet have an exit code, so the monitoring won't fail on this
                    meta.end = datetime.now()
                    meta.exit_code = -2
                    meta.task_hash=task_hash(task)
                    meta.error_msg = "Task doesn't need to run on this instance"
                    write_variant_task_meta(meta, task, task_dir_root, instance_root)
                    changed_dirs.append(task_dir_root)
                continue
            if task.precheck_command and task.precheck_regex:
                check_res = subprocess.run(task.precheck_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if not re.search(task.precheck_regex, check_res.stdout, re.IGNORECASE):
                    logger.info("Task precheck_regex didn't match, skipping")
                    meta.end = datetime.now()
                    meta.exit_code = -3
                    meta.task_hash=task_hash(task)
                    meta.error_msg = "Task precheck_regex didn't match"
                    write_variant_task_meta(meta, task, task_dir_root, instance_root)
                    changed_dirs.append(task_dir_root)
                    continue

            logger.info(f"Starting {task.name}")
            if not task.parallel:
                # Persist the real start time before blocking on this task, and push it
                # now rather than waiting for the taskgroup-end push below. Otherwise the
                # committed meta.json keeps showing the boot-time placeholder start for as
                # long as the task runs, which is indistinguishable from a genuinely
                # abandoned boot placeholder to is_abandoned_boot_meta() once any earlier
                # task in the same run has finished — cleanup would destroy the instance
                # out from under a task that is still actively running.
                meta.start = datetime.now()
                meta.end = None
                meta.exit_code = None
                meta.error_msg = None
                meta.task_hash = task_hash(task)
                write_variant_task_meta(meta, task, task_dir_root, instance_root)
                _push_roots([task_dir_root], f"Starting {task.name} from {repo.gha_url()}")
            q.put(task)
            changed_dirs.append(task_dir_root)
            if not task.parallel:
                q.join()
        # wait at the end of the taskgroup
        q.join()
        docker_prune_after_round()
        # do a push at the end of each round if we made changes
        if changed_dirs:
            _push_roots(changed_dirs, f"Inspecting server from {repo.gha_url()}")
        _finalize_multi_vm_band_if_done(ordered_groups, group_index, data_dir)
    q.join()


def is_abandoned_boot_meta(meta: Meta, last_activity: datetime | None) -> bool:
    """Boot-time placeholder meta left when inspect exits before the task actually runs."""
    return bool(
        meta.start
        and not meta.end
        and last_activity is not None
        and meta.start < last_activity
        and meta.version is None
        and meta.kernel_version is None
    )


def cleanup_destroy_reason(
    task_metas: list[tuple[Task, Meta]],
    now: datetime | None = None,
    *,
    force: bool = False,
) -> str | None:
    """Return a destroy reason if cleanup should tear down the instance, else None.

    Shared by ``cleanup_task`` and Spot self-heal (retry unfinished Spot runs
    on-demand once the machine would have been removed by cleanup).
    """
    now = now or datetime.now()
    start_times: list[datetime] = []
    end_times: list[datetime] = []
    already_ended: list[bool] = []
    sum_timeout = timedelta()
    max_unfinished_timeout = timedelta()
    last_activity = max((meta.end for _, meta in task_metas if meta.end), default=None)
    for task, meta in task_metas:
        if not meta.start:
            continue
        if meta.end:
            end_times.append(meta.end)
            already_ended.append(True)
            continue
        stale = now > meta.start + task.timeout + DESTROY_AFTER
        abandoned = is_abandoned_boot_meta(meta, last_activity)
        if stale or abandoned:
            already_ended.append(True)
            continue
        start_times.append(meta.start)
        already_ended.append(False)
        sum_timeout += task.timeout
        max_unfinished_timeout = max(max_unfinished_timeout, task.timeout)

    if start_times and now >= (wait_time := max(start_times) + max_unfinished_timeout + DESTROY_AFTER):
        return f"last_start: {max(start_times)}, last timeout: {wait_time}"
    if start_times and now >= (wait_time := max(start_times) + sum_timeout + DESTROY_AFTER):
        return f"last_start: {max(start_times)}, wait time: {wait_time}"
    if already_ended and all(already_ended):
        return "all tasks have finished"
    if not start_times and end_times and now >= max(end_times) + DESTROY_AFTER:
        return f"last end: {max(end_times)}"
    if not start_times and force:
        return "forced cleanup"
    return None


def instance_task_metas(
    vendor: str,
    server: str,
    data_dir: str | os.PathLike,
    tasks: list[Task] | None = None,
) -> list[tuple[Task, Meta]]:
    """Applicable (task, meta) pairs for cleanup / Spot heal decisions."""
    if tasks is None:
        tasks = [task for task in get_tasks(vendor) if not isinstance(task, DbaasDbTask)]
    return [(task, load_task_meta(task, data_dir=data_dir)) for task in tasks]


def is_unfinished_spot_meta(meta: Meta) -> bool:
    return bool(meta.spot is True and meta.start and meta.end is None)


def force_ondemand_after_spot_timeout(
    vendor: str,
    server: str,
    data_dir: str | os.PathLike,
    tasks: list[Task] | None = None,
    now: datetime | None = None,
) -> bool:
    """True when an unfinished Spot run is past cleanup destroy time (self-heal with on-demand)."""
    if vendor != "aws":
        return False
    task_metas = instance_task_metas(vendor, server, data_dir, tasks)
    if not any(is_unfinished_spot_meta(meta) for _, meta in task_metas):
        return False
    return cleanup_destroy_reason(task_metas, now) is not None


AWS_ONDEMAND = False
AWS_SPOT = True


def aws_instance_market_modes(*, force_ondemand: bool = False) -> list[bool]:
    """Spot-first, then on-demand after the full regional ladder fails; heal uses on-demand only."""
    if force_ondemand:
        return [AWS_ONDEMAND]
    return [AWS_SPOT, AWS_ONDEMAND]


def update_task_metas_spot(data_dir: str | os.PathLike, tasks: list[Task], spot: bool) -> None:
    """Update spot flag on existing boot metas (e.g. Spot ladder exhausted → on-demand)."""
    for task in tasks:
        meta = load_task_meta(task, data_dir=data_dir)
        if meta.spot == spot:
            continue
        meta.spot = spot
        write_meta(meta, os.path.join(data_dir, task.name, META_NAME))


def finalize_task_metas(
    vendor: str,
    data_dir: str | os.PathLike,
    instance: str,
    accelerator_count: float = 0.0,
    client_data_dir: str | os.PathLike | None = None,
    instance_root: str | os.PathLike | None = None,
) -> None:
    """Mark unfinished task metas when inspect ends before every queued task ran."""
    changed_dirs: list[str | os.PathLike] = []
    last_activity = None
    for task in get_tasks(vendor):
        task_dir_root = task_data_dir(task, data_dir, client_data_dir)
        meta = vm_task_meta(task, task_dir_root, instance_root)
        if meta.end:
            last_activity = max(last_activity, meta.end) if last_activity else meta.end
    for task in get_tasks(vendor):
        task_dir_root = task_data_dir(task, data_dir, client_data_dir)
        meta = vm_task_meta(task, task_dir_root, instance_root)
        if not meta.start or meta.end is not None:
            continue
        if should_run(task, task_dir_root, vendor, instance, accelerator_count):
            meta.end = datetime.now()
            meta.exit_code = -4
            meta.task_hash = task_hash(task)
            meta.error_msg = "Inspect ended before task completed"
            write_variant_task_meta(meta, task, task_dir_root, instance_root)
            changed_dirs.append(task_dir_root)
        elif meta.exit_code is None:
            meta.end = datetime.now()
            meta.exit_code = -2
            meta.task_hash = task_hash(task)
            meta.error_msg = "Task doesn't need to run on this instance"
            write_variant_task_meta(meta, task, task_dir_root, instance_root)
            changed_dirs.append(task_dir_root)
    # Refresh schedule for every task so co-start boot placeholders clear when
    # another variant already has a successful union result.
    if instance_root is not None:
        refresh_all_schedule_metas(instance_root, vendor, active_root=data_dir)
        changed_dirs.append(schedule_root(instance_root))
    if changed_dirs:
        _push_data_dirs(changed_dirs, f"Finalized task metas from {repo.gha_url()}")


def _applicable_tasks(vendor: str, server: str):
    return [task for task in get_tasks(vendor) if not isinstance(task, DbaasDbTask)]


def _applicable_dbaas_tasks(vendor: str, instance_key: str):
    return [
        task
        for task in get_tasks(vendor)
        if isinstance(task, DbaasDbTask)
        and (not task.dbaas_only or (vendor, instance_key) in task.dbaas_only)
    ]


def max_dbaas_runtime(vendor: str, instance_key: str) -> timedelta:
    """Upper bound on DBaaS inspect wall time for one managed DB instance_key."""
    from collections import defaultdict

    taskgroups = defaultdict(list)
    for task in _applicable_dbaas_tasks(vendor, instance_key):
        taskgroups[task.priority].append(task)
    runtime = timedelta()
    for priority in sorted(taskgroups):
        runtime += max((task.timeout for task in taskgroups[priority]), default=timedelta())
    return runtime


def max_server_runtime(vendor: str, server: str) -> timedelta:
    """Upper bound on inspect wall time: sum of per-priority max task timeouts."""
    from collections import defaultdict

    taskgroups = defaultdict(list)
    for task in _applicable_tasks(vendor, server):
        taskgroups[task.priority].append(task)
    runtime = timedelta()
    for priority in sorted(taskgroups):
        runtime += max((task.timeout for task in taskgroups[priority]), default=timedelta())
    return runtime


@cache
def get_last_start(data_dir, vendor, server):
    tasks = list(get_tasks(vendor))
    if not tasks:
        # if there are no tasks, return a low value which can be used as a sort key
        return datetime.min
    # Scheduler metas live under data/{vendor}/{server}/schedule/
    server_data_dir = schedule_root(os.path.join(data_dir, vendor, server))
    tasks = _applicable_tasks(vendor, server)
    meta_starts = [load_task_meta(task, data_dir=server_data_dir).start for task in tasks]
    meta_starts = [start for start in meta_starts if start]
    if not meta_starts:
        # put it to the back
        return datetime.min
    return max(meta_starts)


def get_last_end(data_dir, vendor, server):
    server_data_dir = schedule_root(os.path.join(data_dir, vendor, server))
    meta_ends = [
        load_task_meta(task, data_dir=server_data_dir).end
        for task in _applicable_tasks(vendor, server)
    ]
    meta_ends = [end for end in meta_ends if end]
    if not meta_ends:
        return None
    return max(meta_ends)


def get_last_start_dbaas(data_dir, vendor, instance_key):
    instance_data_dir = os.path.join(data_dir, vendor, instance_key)
    tasks = _applicable_dbaas_tasks(vendor, instance_key)
    if not tasks:
        return datetime.min
    meta_starts = [load_task_meta(task, data_dir=instance_data_dir).start for task in tasks]
    meta_starts = [start for start in meta_starts if start]
    if not meta_starts:
        return datetime.min
    return max(meta_starts)


def get_last_end_dbaas(data_dir, vendor, instance_key):
    instance_data_dir = os.path.join(data_dir, vendor, instance_key)
    meta_ends = [
        load_task_meta(task, data_dir=instance_data_dir).end
        for task in _applicable_dbaas_tasks(vendor, instance_key)
    ]
    meta_ends = [end for end in meta_ends if end]
    if not meta_ends:
        return None
    return max(meta_ends)


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def active_run_blocks_s3_cleanup(
    vendor: str,
    instance: str,
    records: list[Any],
    data_dir: str | os.PathLike,
    *,
    instance_key: str | None = None,
) -> str | None:
    """Return a reason to skip S3-driven destroy when git shows a newer active run.

    For multi-VM, ``data_dir`` is the repo ``data/`` root and metas live under
    ``data/<vendor>/<instance>/``. For DBaaS, pass ``instance_key`` and set
    ``data_dir`` to the managed-DB directory
    (``dbaas/<vendor>/<instance_key>/``); do not join vendor/client again.
    """
    terminated = [_as_utc(r.terminated_at) for r in records if r.terminated_at]
    if not terminated:
        return None
    latest_terminated = max(terminated)
    if instance_key:
        task_data_dir = data_dir
        tasks = _applicable_dbaas_tasks(vendor, instance_key)
    else:
        task_data_dir = schedule_root(os.path.join(data_dir, vendor, instance))
        tasks = _applicable_tasks(vendor, instance)
    if not os.path.isdir(task_data_dir):
        return None
    for task in tasks:
        meta = load_task_meta(task, data_dir=task_data_dir)
        started = _as_utc(meta.start)
        if started and meta.end is None and started > latest_terminated:
            return (
                f"{task.name} started at {meta.start} after stale run record(s) "
                f"ended at {latest_terminated.isoformat()}"
            )
    return None


def should_scan_for_cleanup(
    data_dir,
    vendor,
    server,
    *,
    lookback_mins: int | None = None,
    data_only: bool = False,
) -> bool:
    """Return True if this server should be checked by a cleanup run."""
    server_data_dir = os.path.join(data_dir, vendor, server)
    if data_only and not os.path.isdir(server_data_dir):
        return False
    last_start = get_last_start(data_dir, vendor, server)
    if last_start == datetime.min:
        return False
    now = datetime.now()
    if lookback_mins is not None:
        return last_start >= now - timedelta(minutes=lookback_mins)
    last_end = get_last_end(data_dir, vendor, server)
    active_deadline = last_start + max_server_runtime(vendor, server) + DESTROY_AFTER + CLEANUP_BOOT_SLACK
    if last_end:
        return now <= max(active_deadline, last_end + CLEANUP_DESTROY_RETRY)
    return now <= active_deadline


def should_scan_dbaas_for_cleanup(
    data_dir,
    vendor,
    instance_key,
    *,
    lookback_mins: int | None = None,
    data_only: bool = False,
) -> bool:
    """Return True if this managed DB instance_key should be checked by cleanup-sweep."""
    instance_data_dir = os.path.join(data_dir, vendor, instance_key)
    if data_only and not os.path.isdir(instance_data_dir):
        return False
    last_start = get_last_start_dbaas(data_dir, vendor, instance_key)
    if last_start == datetime.min:
        return False
    now = datetime.now()
    if lookback_mins is not None:
        return last_start >= now - timedelta(minutes=lookback_mins)
    last_end = get_last_end_dbaas(data_dir, vendor, instance_key)
    active_deadline = (
        last_start + max_dbaas_runtime(vendor, instance_key) + DESTROY_AFTER + CLEANUP_BOOT_SLACK
    )
    if last_end:
        return now <= max(active_deadline, last_end + CLEANUP_DESTROY_RETRY)
    return now <= active_deadline


def instance_start_order_key(vendor: str, instance: str) -> int:
    """Stable pseudo-random order for start iteration (mixes instance sizes)."""
    return crc32(f"{vendor}/{instance}".encode("utf-8"))


def sort_servers_for_start(available_servers: dict) -> list[tuple[tuple[str, str], list]]:
    """Return servers in deterministic hash order for the start command.

    Catalog price order would start small instances first and heavy ones last,
    which burns through quota on a new vendor before larger SKUs get a turn.
    Hashing vendor/instance mixes sizes while staying predictable across runs.
    """
    return sorted(
        available_servers.items(),
        key=lambda item: instance_start_order_key(item[0][0], item[0][1]),
    )


def sort_managed_dbs_for_start(available_managed_dbs: dict) -> list[tuple[tuple[str, str], list]]:
    """Return managed DB targets in deterministic hash order for start-dbaas."""
    return sorted(
        available_managed_dbs.items(),
        key=lambda item: instance_start_order_key(item[0][0], item[0][1]),
    )


def sort_available_managed_dbs(available_managed_dbs: dict, data_dir, reverse=True, max_start=None):
    if max_start:
        available_managed_dbs = {
            k: v
            for k, v in available_managed_dbs.items()
            if get_last_start_dbaas(data_dir, k[0], k[1]) >= max_start
        }
    return sorted(
        available_managed_dbs.items(),
        key=lambda item: get_last_start_dbaas(data_dir, item[0][0], item[0][1]),
        reverse=reverse,
    )


def sort_available_servers(available_servers: dict, data_dir, reverse=True, max_start=None):
    if max_start:
        available_servers = {
            k: v for k, v in available_servers.items()
            if get_last_start(data_dir, k[0], k[1]) >= max_start
        }
    return sorted(
        available_servers.items(),
        key=lambda item: get_last_start(data_dir, item[0][0], item[0][1]),
        reverse=reverse,
    )


def inspector_user_data_replacements(
    vendor: str,
    server: str,
    srv_data,
    region: str,
    zone: str | None,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    *,
    role: str = "server",
    mp_authkey_b64: str = "",
    mp_port: int = 18765,
    client_private_ip: str = "",
    client_instance: str = "",
    client_cpu_arch: str = "",
    client_vcpus: int = 0,
    provisioned_disk_gib: int | None = None,
    client_disk_gib: int = 30,
    db_disk_type: str = "",
    db_disk_iops: str = "",
    db_disk_throughput: str = "",
    include_run_upload: bool | None = None,
) -> dict[str, str]:
    import s3_runs

    if include_run_upload is None:
        include_run_upload = role == "server"
    log_url, run_url = s3_runs.presigned_urls_for_instance(vendor, server)
    if not include_run_upload:
        run_url = ""
    task_logs_post_b64 = s3_runs.presigned_task_logs_post_b64(vendor, server)
    cdn_dataset_post_b64 = s3_runs.presigned_cdn_dataset_post_b64()
    replacements = {
        "SSH_DEPLOY_KEY_B64": ssh_deploy_key_b64,
        "REPO_URL": repo_url_ssh,
        "GITHUB_SERVER_URL": os.environ.get("GITHUB_SERVER_URL", ""),
        "GITHUB_REPOSITORY": os.environ.get("GITHUB_REPOSITORY", ""),
        "GITHUB_RUN_ID": os.environ.get("GITHUB_RUN_ID", ""),
        "GITHUB_WORKFLOW": s3_runs.workflow_slug(),
        "BENCHMARK_SECRETS_PASSPHRASE": os.environ.get("BENCHMARK_SECRETS_PASSPHRASE", ""),
        "SENTINEL_API_TOKEN": os.environ.get("SENTINEL_API_TOKEN", ""),
        "KEEPER_API_BASE": os.environ.get(
            "KEEPER_API_BASE", "https://keeper.sparecores.net"
        ),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "VENDOR": vendor,
        "INSTANCE": server,
        "REGION": region or "",
        "ZONE": zone or "",
        "ACCELERATOR_COUNT": srv_data.accelerator_count,
        "SHUTDOWN_MINS": timeout_mins + 30,  # give enough time to set up the machine
        "HOST_TIMING_DIR": host_timing_dir(vendor, server, github_run_id()),
        "LOG_UPLOAD_URL": log_url,
        "RUN_UPLOAD_URL": run_url,
        "TASK_LOGS_S3_POST_B64": task_logs_post_b64,
        "SC_CDN_DATASET_POST_B64": cdn_dataset_post_b64,
        "INSPECTOR_ROLE": role,
        "MP_AUTHKEY_B64": mp_authkey_b64,
        "MP_PORT": str(mp_port),
        "CLIENT_PRIVATE_IP": client_private_ip,
        "MULTI_VM_CLIENT_INSTANCE": client_instance,
        "MULTI_VM_CLIENT_CPU_ARCH": client_cpu_arch,
        "MULTI_VM_CLIENT_VCPUS": str(client_vcpus or ""),
        "PROVISIONED_DISK_GIB": str(provisioned_disk_gib or VOLUME_SIZE),
        "CLIENT_DISK_GIB": str(client_disk_gib),
        "MULTI_VM_DB_DISK_TYPE": db_disk_type,
        "MULTI_VM_DB_DISK_IOPS": db_disk_iops,
        "MULTI_VM_DB_DISK_THROUGHPUT": db_disk_throughput,
        "TOPOLOGY": "multi_vm" if role != "dbaas_client" else "dbaas",
        "MANAGED_DB_INSTANCE_KEY": "",
        "SC_DB_HOST": "",
        "SC_DB_PORT": "5432",
        "SC_DB_USER": "",
        "SC_DB_PASSWORD": "",
        "SC_DB_NAME": "bench",
        "DB_WAIT_TIMEOUT_SEC": os.environ.get("DB_WAIT_TIMEOUT_SEC", "1200"),
    }
    replacements["SC_CDN_BASE_URL"] = os.environ.get(
        "SC_CDN_BASE_URL",
        "https://cdn.sparecores.net/sc-inspector",
    )
    return replacements


def build_inspector_user_data(
    vendor: str,
    server: str,
    srv_data,
    region: str,
    zone: str | None,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    *,
    role: str = "server",
    mp_authkey_b64: str = "",
    mp_port: int = 18765,
    client_private_ip: str = "",
    client_instance: str = "",
    client_cpu_arch: str = "",
    client_vcpus: int = 0,
    provisioned_disk_gib: int | None = None,
    client_disk_gib: int = 30,
    include_run_upload: bool | None = None,
) -> tuple[str, str]:
    replacements = inspector_user_data_replacements(
        vendor,
        server,
        srv_data,
        region,
        zone,
        timeout_mins,
        ssh_deploy_key_b64,
        repo_url_ssh,
        role=role,
        mp_authkey_b64=mp_authkey_b64,
        mp_port=mp_port,
        client_private_ip=client_private_ip,
        client_instance=client_instance,
        client_cpu_arch=client_cpu_arch,
        client_vcpus=client_vcpus,
        provisioned_disk_gib=provisioned_disk_gib,
        client_disk_gib=client_disk_gib,
        include_run_upload=include_run_upload,
    )
    return user_data_pack.publish_user_data_script(vendor, server, USER_DATA, replacements)


def build_server_user_data_replacements(
    vendor: str,
    server: str,
    srv_data,
    region: str,
    zone: str | None,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    *,
    mp_authkey_b64: str,
    mp_port: int,
    client_instance: str,
    client_cpu_arch: str,
    client_vcpus: int,
    provisioned_disk_gib: int,
    client_disk_gib: int,
    db_disk_type: str = "",
    db_disk_iops: str = "",
    db_disk_throughput: str = "",
) -> dict:
    """Replacements for server user-data rendered inside Pulumi via Output.apply."""
    replacements = inspector_user_data_replacements(
        vendor,
        server,
        srv_data,
        region,
        zone,
        timeout_mins,
        ssh_deploy_key_b64,
        repo_url_ssh,
        role="server",
        mp_authkey_b64=mp_authkey_b64,
        mp_port=mp_port,
        client_private_ip="{CLIENT_PRIVATE_IP}",
        client_instance=client_instance,
        client_cpu_arch=client_cpu_arch,
        client_vcpus=client_vcpus,
        provisioned_disk_gib=provisioned_disk_gib,
        client_disk_gib=client_disk_gib,
        db_disk_type=db_disk_type,
        db_disk_iops=db_disk_iops,
        db_disk_throughput=db_disk_throughput,
    )
    bootstrap, _ = user_data_pack.publish_user_data_script(vendor, server, USER_DATA, replacements)
    return {
        "USER_DATA_TEMPLATE": bootstrap,
    }


def delayed_destroy(vendor, server, resource_opts):
    from sc_runner import runner

    # to be run in the background
    time.sleep(180)

    # change the thread name for logging
    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{server}"
    instance_logger = logging.getLogger(f"{vendor}/{server}")
    try:
        runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
    except Exception:
        logger.exception("Failed to destroy")


def _order_location_candidates(candidates: set[str], *preference_lists: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for preference in preference_lists:
        for location in preference:
            if location in candidates and location not in seen:
                ordered.append(location)
                seen.add(location)
    for location in sorted(candidates - seen):
        ordered.append(location)
    return ordered


def vultr_deployable_regions(server: str) -> list[str]:
    """ACTIVE ONDEMAND regions for the deployable Vultr plan and its catalog instance."""
    from sc_runner import data as sc_data
    from sc_runner.resources import vultr as vultr_resources

    plan = server if vultr_resources._is_bare_metal(server) else vultr_resources.resolve_plan(server, VOLUME_SIZE)
    regions = sc_data.plan_regions("vultr", plan)
    if plan != server:
        regions = list(dict.fromkeys([*regions, *sc_data.plan_regions("vultr", server)]))
    return regions


def vultr_cleanup_regions(server: str, regions: list[str]) -> list[str]:
    """Regions to scan when destroying Vultr stacks."""
    return list(dict.fromkeys([*regions, *vultr_deployable_regions(server)]))


def candidate_regions(vendor: str, server: str, regions: list[str]) -> list[str]:
    """Return deployable regions for an instance, cheapest sc-data price first."""
    from sc_runner import data as sc_data

    candidates = list(regions)
    if vendor == "vultr":
        available = set(vultr_deployable_regions(server))
        if available:
            candidates = _order_location_candidates(available, candidates, list(available))
    prices = sc_data.server_region_prices(vendor, server)
    ordered = sc_data.sort_by_price(candidates, prices)
    logger.info(f"Region order for {vendor}/{server}: {ordered}")
    return ordered


def candidate_zones(
    vendor: str,
    server: str,
    zones: list[str],
    zone_to_region: dict[str, str] | None = None,
) -> list[str]:
    """Return deployable zones for an instance, cheapest sc-data price first."""
    from sc_runner import data as sc_data

    prices = sc_data.server_zone_prices(vendor, server)
    if vendor == "alicloud":
        # Keep cn- zones as a last resort because of slow network connectivity.
        ordered = sorted(
            zones,
            key=lambda zone: (
                (zone_to_region or {}).get(zone, "").startswith("cn-"),
                prices.get(zone, float("inf")),
                zone,
            ),
        )
    else:
        ordered = sc_data.sort_by_price(zones, prices)
    logger.info(f"Zone order for {vendor}/{server}: {ordered}")
    return ordered


# Pulumi engine logs instance creation as e.g. "aws:ec2:Instance m9g.large created (15s)".
# Use that line (end time + duration) as the source of truth; engine event timestamps lag behind.
INSTANCE_RESOURCE_SUFFIXES = (
    ":Instance",
    ":Server",
    ":VirtualMachine",
    ":BareMetalServer",
)
PULUMI_INSTANCE_CREATED_RE = re.compile(
    r"([\w:.-]+)\s+(\S+)\s+created\s+\((\d+(?:\.\d+)?)s\)"
)


def strip_pulumi_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)
    return re.sub(r"<\{%[^%]+%\}>", "", text)


_SECRET_OUTPUT_KEYS = (
    "db_admin_password",
    "administrator_login_password",
    "admin_password",
)


def redact_pulumi_output(text: str) -> str:
    """Mask credential-like values before Pulumi stdout reaches CI logs."""
    text = strip_pulumi_output(text)
    for key in _SECRET_OUTPUT_KEYS:
        text = re.sub(
            rf'({re.escape(key)}\s*:\s*)"[^"]*"',
            r'\1"[secret]"',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"({re.escape(key)}\s*:\s*)'[^']*'",
            r"\1'[secret]'",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"({re.escape(key)})=([^\s,]+)",
            r"\1=[secret]",
            text,
            flags=re.IGNORECASE,
        )
    return text


def pulumi_log_output(message: str) -> None:
    logger.info(redact_pulumi_output(message))


def pulumi_on_output(logger):
    """Return an ``on_output`` callback that redacts secrets before logging."""

    def on_output(message: str) -> None:
        logger.info(redact_pulumi_output(message))

    return on_output


def is_instance_resource_type(resource_type: str) -> bool:
    return any(resource_type.endswith(suffix) for suffix in INSTANCE_RESOURCE_SUFFIXES)


class InstanceCreationTiming:
    """Tracks cloud instance resource create start/end from Pulumi progress output."""

    def __init__(self) -> None:
        self.start: datetime | None = None
        self.end: datetime | None = None

    def reset(self) -> None:
        self.start = None
        self.end = None

    def complete(self) -> bool:
        return self.start is not None and self.end is not None

    def record_from_log(self, message: str, server: str) -> None:
        clean = strip_pulumi_output(message)
        match = PULUMI_INSTANCE_CREATED_RE.search(clean)
        if not match:
            return
        resource_type, name, duration_s = match.group(1), match.group(2), float(match.group(3))
        if name != server or not is_instance_resource_type(resource_type):
            return
        # Pulumi prints "created (Xs)" when the resource finishes; duration is its stopwatch.
        end = utc_now()
        self.end = end
        self.start = end - timedelta(seconds=duration_s)


def pulumi_output_filter(message, error_msgs, output, logger=logging):
    # print output to the console with logger, so we have the dates
    safe = redact_pulumi_output(message)
    logger.info(safe)
    output.append(safe)
    if any(regex.search(message) for regex in PULUMI_ERRORS):
        error_msgs.append(safe)


def pulumi_event_filter(event, error_msgs):
    try:
        if event.diagnostic_event.severity == "error" and any([regex.search(event.diagnostic_event.message) for regex in PULUMI_ERRORS]):
            error_msgs.append(event.diagnostic_event.message)
    except Exception:
        pass


def pulumi_instance_timing_event(event, error_msgs):
    pulumi_event_filter(event, error_msgs)


def pulumi_stack_opts(
    error_msgs,
    output,
    logger,
    instance_timing: InstanceCreationTiming | None = None,
    server: str | None = None,
):
    """Capture Pulumi errors and instance-creation timing from stdout and engine events."""

    def on_output(message):
        pulumi_output_filter(message, error_msgs, output, logger)
        if instance_timing is not None and server is not None:
            instance_timing.record_from_log(message, server)

    def on_event(event):
        pulumi_instance_timing_event(event, error_msgs)

    return dict(on_output=on_output, on_event=on_event)


def is_retryable_start_error(error_msg: str | None) -> bool:
    if not error_msg:
        return False
    return any(marker in error_msg for marker in RETRYABLE_START_ERROR_MARKERS)


def pulumi_error_text(exc: BaseException, error_msgs: list[str] | None = None) -> str:
    parts = [str(exc)]
    if error_msgs:
        parts.extend(error_msgs)
    for attr in ("stdout", "stderr", "message"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    if exc.__cause__:
        parts.append(pulumi_error_text(exc.__cause__))
    return "\n".join(parts)


def is_retryable_pulumi_error(exc: BaseException, error_msgs: list[str] | None = None) -> bool:
    text = pulumi_error_text(exc, error_msgs)
    return any(marker in text for marker in RETRYABLE_PULUMI_ERROR_MARKERS)


def is_region_fatal_error(exc: BaseException, error_msgs: list[str] | None = None) -> bool:
    text = pulumi_error_text(exc, error_msgs)
    return any(marker in text for marker in REGION_FATAL_ERROR_MARKERS)


def is_gcp_host_maintenance_policy_error(error_text: str) -> bool:
    """True if GCP rejected the instance for an invalid onHostMaintenance value."""
    lowered = error_text.lower()
    return "onhostmaintenance" in lowered or "on_host_maintenance" in lowered


def is_pulumi_error_summary(message: str) -> bool:
    stripped = message.strip()
    if PULUMI_ERROR_SUMMARY.search(stripped):
        return True
    return any(regex.search(stripped) for regex in PULUMI_BORING_ERRORS)


def best_pulumi_error(error_msgs: list[str]) -> str | None:
    """Pick the most informative Pulumi/provider error from captured output."""
    if not error_msgs:
        return None
    candidates = [msg for msg in error_msgs if not is_pulumi_error_summary(msg)]
    if not candidates:
        candidates = error_msgs
    for msg in reversed(candidates):
        if match := _PROVIDER_JSON_ERROR.search(msg):
            return match.group(1)
    detailed = [
        msg for msg in candidates
        if any(token in msg for token in ("sdk-v2/", "error creating", "Unable to create", "InsufficientInstanceCapacity"))
    ]
    if detailed:
        return detailed[-1]
    return candidates[-1]


def record_instance_start_failure(lock, data_dir, tasks, error_msgs, fallback_msg=None):
    """Write exit_code=-1 to task meta when the cloud instance could not be started."""
    now = datetime.now()
    if error_msgs:
        error_msg = remove_matches(FILTER_ERROR_MSG, best_pulumi_error(error_msgs))
    elif fallback_msg:
        error_msg = fallback_msg
    else:
        error_msg = "Failed to start instance"
    logger.info(f"Failed to start instance, uploading error messages: {error_msg}")
    with lock:
        repo.pull()
        for task in tasks:
            meta = load_task_meta(task, data_dir=data_dir)
            if not meta.start:
                meta.start = now
            meta.end = now
            meta.exit_code = -1
            meta.error_msg = error_msg
            meta.task_hash = task_hash(task)
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
        repo.push_path(data_dir, f"Failed to start instance from {repo.gha_url()}")


def delayed_destroy(vendor, server, resource_opts):
    from sc_runner import runner

    # to be run in the background
    time.sleep(180)

    # change the thread name for logging
    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{server}"
    instance_logger = logging.getLogger(f"{vendor}/{server}")
    try:
        runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
    except Exception:
        logger.exception("Failed to destroy")


def remove_matches(regexes, input_string):
    for regex in regexes:
        input_string = re.sub(regex, '', input_string)
    return input_string


STALE_LOCK_AGE = timedelta(days=1)
_LOCK_CREATED_RE = re.compile(r" at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)")


class StackLockedError(Exception):
    """Pulumi stack is locked by another process and cleanup should be retried later."""


def lock_created_at(exc: BaseException) -> datetime | None:
    match = _LOCK_CREATED_RE.search(str(exc))
    if not match:
        return None
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))


def retry_locked_cleanup(func: Callable[[], Any], cancel_func: Callable[[], None] | None = None) -> Any:
    """Retry a Pulumi cleanup operation; cancel and retry when the lock is older than a day."""
    import pulumi

    stale_cancelled = False
    for i in range(3):
        try:
            return func()
        except pulumi.automation.errors.ConcurrentUpdateError as exc:
            created_at = lock_created_at(exc)
            now = datetime.now(timezone.utc)
            if (
                cancel_func
                and created_at
                and now - created_at > STALE_LOCK_AGE
                and not stale_cancelled
            ):
                logger.warning(
                    "Cancelling stale Pulumi lock created at %s (older than %s)",
                    created_at,
                    STALE_LOCK_AGE,
                )
                cancel_func()
                stale_cancelled = True
                continue
            if i < 2:
                logger.warning("ConcurrentUpdateError during cleanup, retry #%d", i + 1)
                time.sleep(random.randint(1, 5))
                continue
            raise StackLockedError(str(exc)) from exc


def retry_locked(
    func,
    *args,
    instance_timing: InstanceCreationTiming | None = None,
    error_msgs: list[str] | None = None,
    **kwargs,
):
    """Retry a pulumi function with random backoff for locking and transient quota errors."""
    import pulumi

    for i in range(3):
        if instance_timing is not None:
            instance_timing.reset()
        try:
            return func(*args, **kwargs)
        except pulumi.automation.errors.ConcurrentUpdateError:
            logger.exception(f"ConcurrentUpdateError, retry #{i}")
            time.sleep(random.randint(1, 5))
        except Exception as exc:
            if is_retryable_pulumi_error(exc, error_msgs) and i < 2:
                logger.warning("Retryable Pulumi error, retry #%d: %s", i + 1, exc)
                time.sleep(random.randint(1, 5))
                continue
            raise


def _try_start_multi_vm_inspect(
    executor,
    lock,
    data_dir,
    vendor,
    server,
    tasks,
    srv_data,
    regions,
    zones,
    zone_to_region,
    timeout_mins,
    ssh_deploy_key_b64,
    repo_url_ssh,
    instance_logger,
    instance_timing,
    error_msgs,
    aws_market_modes: list[bool] | None = None,
) -> bool:
    """Provision a multi-VM stack when tasks need a companion client."""
    import secrets

    from benchmark_tiers import merge_client_requirements
    from companion_picker import pick_client_instance
    from db_storage import db_storage_plan
    from sc_runner import runner
    from sc_runner.resources.multi_vm import MultiVmStackSpec

    multi_tasks = [t for t in tasks if isinstance(t, MultiVmDbTask)]
    if not multi_tasks:
        return False

    # Size disk for every multi-VM task on this instance, not only those triggering this start.
    planned_multi = [t for t in get_tasks(vendor) if isinstance(t, MultiVmDbTask)]
    disk_sources = planned_multi or multi_tasks
    client_req = merge_client_requirements([t.client_requirements(srv_data) for t in disk_sources])
    mem_gib = float(srv_data.memory_amount) / 1024.0
    storage_plan = db_storage_plan(
        vendor, mem_gib, vcpus=int(srv_data.vcpus), machine_type=server
    )
    db_disk = storage_plan.storage_gib
    db_disk_opts = storage_plan.multi_vm_disk_opts()
    authkey = secrets.token_bytes(32)
    authkey_b64 = base64.b64encode(authkey).decode("ascii")
    mp_port = 18765

    def location_candidates():
        if vendor == "gcp":
            return [("zone", z) for z in candidate_zones(vendor, server, zones)]
        if vendor == "alicloud":
            return [
                ("zone", z)
                for z in candidate_zones(vendor, server, zones, zone_to_region)
            ]
        return [("region", r) for r in candidate_regions(vendor, server, regions)]

    market_modes = aws_market_modes if vendor == "aws" else [None]
    if vendor == "aws" and market_modes is None:
        market_modes = aws_instance_market_modes()

    for mode_idx, use_spot in enumerate(market_modes):
        if vendor == "aws" and mode_idx > 0:
            logger.info(
                f"{vendor}/{server}: multi-VM Spot exhausted all regions; falling back to on-demand"
            )
            with lock:
                repo.pull()
                update_task_metas_spot(data_dir, tasks, spot=False)
                repo.push_path(data_dir, f"Falling back to on-demand from {repo.gha_url()}")
            error_msgs.clear()
        if vendor == "aws":
            logger.info(
                f"{vendor}/{server}: multi-VM trying market={'spot' if use_spot else 'on-demand'}"
            )

        for kind, location in location_candidates():
            if kind == "zone":
                client = pick_client_instance(vendor, location, client_req)
                region = (zone_to_region or {}).get(location, "")
                zone = location
            else:
                client = pick_client_instance(vendor, location, client_req)
                region = location
                zone = None
            if client is None:
                logger.info(f"No companion client for {vendor}/{location}")
                continue

            logger.info(
                f"Multi-VM: {vendor}/{server} + client {client.api_reference} in {location}"
            )
            _, client_ud_b64 = build_inspector_user_data(
                vendor,
                client.api_reference,
                client,
                region,
                zone,
                timeout_mins,
                ssh_deploy_key_b64,
                repo_url_ssh,
                role="client",
                mp_authkey_b64=authkey_b64,
                mp_port=mp_port,
                include_run_upload=False,
            )
            server_replacements = build_server_user_data_replacements(
                vendor,
                server,
                srv_data,
                region,
                zone,
                timeout_mins,
                ssh_deploy_key_b64,
                repo_url_ssh,
                mp_authkey_b64=authkey_b64,
                mp_port=mp_port,
                client_instance=client.api_reference,
                client_cpu_arch=client.cpu_architecture or "",
                client_vcpus=int(client.vcpus or 0),
                provisioned_disk_gib=db_disk,
                client_disk_gib=30,
                db_disk_type=str(db_disk_opts.get("disk_type") or ""),
                db_disk_iops=str(db_disk_opts.get("disk_iops") or ""),
                db_disk_throughput=str(db_disk_opts.get("disk_throughput") or ""),
            )
            spec = MultiVmStackSpec.two_vm(
                primary_instance=server,
                client_instance=client.api_reference,
                primary_disk_gib=db_disk,
                client_disk_gib=30,
                primary_disk_type=db_disk_opts.get("disk_type"),
                primary_disk_iops=db_disk_opts.get("disk_iops"),
                primary_disk_throughput=db_disk_opts.get("disk_throughput"),
                client_user_data_b64=client_ud_b64,
                primary_user_data_template=server_replacements["USER_DATA_TEMPLATE"],
                extra_exports={"mp_port": mp_port},
            )
            resource_opts = dict(
                public_key=os.environ.get("SSH_PUBLIC_KEY", ""),
                instance=server,
                disk_size=db_disk,
                multi_vm=spec,
            )
            if vendor == "aws" and use_spot is not None:
                resource_opts["spot"] = bool(use_spot)
            if vendor == "azure":
                resource_opts["region"] = region
            elif vendor == "gcp":
                resource_opts["zone"] = zone
            elif vendor == "alicloud":
                resource_opts["region"] = region
                resource_opts["availability_zone"] = zone
            else:
                resource_opts["region"] = region

            runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
            pulumi_output = []
            stack_opts = pulumi_stack_opts(
                error_msgs, pulumi_output, instance_logger, instance_timing, server
            )
            extra = {}
            if vendor == "azure":
                image_sku = (
                    "server-arm64"
                    if "arm" in (srv_data.cpu_architecture or "").lower()
                    else "server"
                )
                extra["image_sku"] = image_sku
            try:
                retry_locked(
                    runner.create,
                    vendor,
                    {},
                    resource_opts | extra,
                    stack_opts=stack_opts,
                    instance_timing=instance_timing,
                    error_msgs=error_msgs,
                )
                return True
            except Exception as exc:
                logger.exception("Multi-VM create failed for %s", location)
                if not error_msgs:
                    error_msgs.append(str(exc))
                executor.submit(delayed_destroy, vendor, server, copy.deepcopy(resource_opts))
    return False


def start_inspect(executor, lock, data_dir, vendor, server, tasks, srv_data, regions, zones, zone_to_region=None):
    import sc_runner.resources
    from sc_runner import runner
    from sc_runner.resources import default

    # change the thread name for logging
    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{server}"
    instance_logger = logging.getLogger(f"{vendor}/{server}")

    error_msgs = []
    instance_started = False
    instance_timing = InstanceCreationTiming()
    sum_timeout = timedelta()
    force_ondemand = force_ondemand_after_spot_timeout(vendor, server, data_dir, tasks)
    aws_market_modes = aws_instance_market_modes(force_ondemand=force_ondemand) if vendor == "aws" else None
    if force_ondemand:
        logger.info(f"{vendor}/{server}: Spot self-heal → on-demand only")
    initial_spot = (
        bool(aws_market_modes[0]) if vendor == "aws" and aws_market_modes else None
    )
    with lock:
        repo.pull()
        for task in tasks:
            fields = {}
            if vendor == "aws":
                fields["spot"] = initial_spot
            meta = boot_meta_for_task(task, data_dir, **fields)
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
            sum_timeout += task.timeout
        repo.push_path(data_dir, f"Starting server from {repo.gha_url()}")
    timeout_mins = int(sum_timeout.total_seconds()/60)
    logger.info(f"Starting {vendor}/{server} with {timeout_mins}m timeout")
    # Construct SSH repo URL from GitHub Actions context, or use fallback
    github_repo = os.environ.get("GITHUB_REPOSITORY", "SpareCores/sc-inspector-data")
    repo_url_ssh = f"git@github.com:{github_repo}.git"
    # Base64 encode SSH key to avoid shell interpretation issues
    ssh_deploy_key = os.environ.get("SSH_DEPLOY_KEY", "")
    ssh_deploy_key_b64 = base64.b64encode(ssh_deploy_key.encode("utf-8")).decode("ascii") if ssh_deploy_key else ""
    multi_tasks = [t for t in tasks if isinstance(t, MultiVmDbTask)]
    if multi_tasks:
        instance_started = _try_start_multi_vm_inspect(
            executor,
            lock,
            data_dir,
            vendor,
            server,
            tasks,
            srv_data,
            regions,
            zones,
            zone_to_region,
            timeout_mins,
            ssh_deploy_key_b64,
            repo_url_ssh,
            instance_logger,
            instance_timing,
            error_msgs,
            aws_market_modes=aws_market_modes,
        )
        if instance_started and instance_timing.complete() and any(task.always_run for task in tasks):
            with lock:
                repo.pull()
                record_timing_api(data_dir, instance_timing.start, instance_timing.end)
                repo.push_path(data_dir, f"Instance creation timing from {repo.gha_url()}")
        if not instance_started:
            record_instance_start_failure(lock, data_dir, tasks, error_msgs)
        return
    pulumi_tempdir = tempfile.TemporaryDirectory(prefix=f"pulumi-{vendor}-{server}-")
    pulumi_opts = dict(work_dir=pulumi_tempdir.name)
    # start instance
    if vendor in ("aws", "gcp", "hcloud", "upcloud", "ovh", "alicloud", "vultr"):
        # get the copy (so we don't modify the original) of the default instance opts for the vendor and add ours
        instance_opts = copy.deepcopy(default(getattr(sc_runner.resources, vendor).DEFAULTS, "instance_opts"))
    if vendor in ["hcloud", "upcloud", "ovh"]:
        resource_opts = dict(instance=server)
        if vendor == "hcloud":
            # allows only one key with the same fingerprint, so we need to use the already existing one
            instance_opts |= dict(ssh_keys=["info@sparecores.com"])
        if vendor == "ovh":
            # also reuse already existing SSH key
            instance_opts |= dict(ssh_key={"name": "spare-cores"})
        if vendor == "upcloud":
            # explicitly set SSH key from envvar
            resource_opts |= dict(public_key=os.environ.get("SSH_PUBLIC_KEY"))
            resource_opts |= dict(disk_size=VOLUME_SIZE)
        for region in candidate_regions(vendor, server, regions):
            logger.info(f"Trying {region}")
            resource_opts["region"] = region
            user_data, b64_user_data = build_inspector_user_data(
                vendor, server, srv_data, region, None, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
            )

            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
            pulumi_output = []
            stack_opts = pulumi_stack_opts(error_msgs, pulumi_output, instance_logger, instance_timing, server)
            try:
                retry_locked(runner.create, vendor, pulumi_opts,
                             resource_opts | dict(instance_opts=instance_opts, user_data=user_data),
                             stack_opts=stack_opts, instance_timing=instance_timing,
                             error_msgs=error_msgs)
                # empty it if create succeeded, just in case
                error_msgs = []
                instance_started = True
                break
            except Exception as e:
                # on failure, try the next one
                logger.exception("Couldn't start instance")
                if not error_msgs:
                    error_msgs.append(str(e))

    if vendor == "vultr":
        resource_opts = dict(
            public_key=os.environ.get("SSH_PUBLIC_KEY"),
            instance=server,
            disk_size=VOLUME_SIZE,
        )
        for region in candidate_regions(vendor, server, regions):
            logger.info(f"Trying {region}")
            resource_opts["region"] = region
            user_data, b64_user_data = build_inspector_user_data(
                vendor, server, srv_data, region, None, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
            )

            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
            pulumi_output = []
            stack_opts = pulumi_stack_opts(error_msgs, pulumi_output, instance_logger, instance_timing, server)
            try:
                retry_locked(runner.create, vendor, pulumi_opts,
                             resource_opts | dict(instance_opts=instance_opts, user_data=user_data),
                             stack_opts=stack_opts, instance_timing=instance_timing,
                             error_msgs=error_msgs)
                # empty it if create succeeded, just in case
                error_msgs = []
                instance_started = True
                break
            except Exception as e:
                # on failure, try the next one
                logger.exception("Couldn't start instance")
                if not error_msgs:
                    error_msgs.append(str(e))

    if vendor == "aws":
        # we use the key_name in instance_opts instead of creating a new key
        resource_opts = dict(public_key="", instance=server, disk_size=VOLUME_SIZE)
        instance_opts |= dict(key_name="spare-cores")
        for mode_idx, use_spot in enumerate(aws_market_modes or aws_instance_market_modes()):
            if mode_idx > 0:
                logger.info(
                    f"{vendor}/{server}: Spot exhausted all regions; falling back to on-demand"
                )
                with lock:
                    repo.pull()
                    update_task_metas_spot(data_dir, tasks, spot=False)
                    repo.push_path(data_dir, f"Falling back to on-demand from {repo.gha_url()}")
                error_msgs.clear()
            logger.info(
                f"{vendor}/{server}: trying market={'spot' if use_spot else 'on-demand'}"
            )
            # AWS rejects ModifyInstanceAttribute(InstanceInitiatedShutdownBehavior) for
            # spot instances ("not supported for spot instances"); the provider issues
            # that call whenever the field is set, regardless of value, so it must be
            # left out entirely for spot rather than just matching AWS's own default.
            mode_instance_opts = dict(instance_opts)
            if not use_spot:
                mode_instance_opts["instance_initiated_shutdown_behavior"] = "terminate"
            for region in candidate_regions(vendor, server, regions):
                logger.info(f"Trying {region}")
                resource_opts["region"] = region
                user_data, b64_user_data = build_inspector_user_data(
                    vendor, server, srv_data, region, None, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
                )

                # before starting, destroy everything to make sure the user-data will run (this is the first boot)
                runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
                pulumi_output = []
                stack_opts = pulumi_stack_opts(error_msgs, pulumi_output, instance_logger, instance_timing, server)
                create_opts = resource_opts | dict(
                    instance_opts=mode_instance_opts,
                    user_data=b64_user_data,
                    spot=bool(use_spot),
                )
                try:
                    retry_locked(runner.create, vendor, pulumi_opts,
                                 create_opts,
                                 stack_opts=stack_opts, instance_timing=instance_timing,
                                 error_msgs=error_msgs)
                    # empty it if create succeeded, just in case
                    error_msgs = []
                    instance_started = True
                    break
                except Exception as e:
                    # on failure, try the next one
                    logger.exception("Couldn't start instance")
                    if not error_msgs:
                        error_msgs.append(str(e))
            if instance_started:
                break

    if vendor == "alicloud":
        # we use the key_name in instance_opts instead of creating a new key
        resource_opts = dict(public_key="", instance=server, disk_size=VOLUME_SIZE)
        instance_opts |= dict(
            key_name="spare-cores",
        )
        done = False
        for zone in candidate_zones(vendor, server, zones, zone_to_region):
            # Get region from the zone_to_region mapping (from database)
            region = zone_to_region.get(zone)
            if not region:
                logger.warning(f"Unknown region for zone {zone}, skipping")
                continue
            if region.startswith("cn-"):
                # Chinese regions have weak network connectivity, warn but try as fallback
                logger.warning(f"Trying cn- zone {zone} as fallback (slow network expected)")
            logger.info(f"Trying zone {zone} (region {region})")
            resource_opts["region"] = region
            resource_opts["availability_zone"] = zone
            user_data, b64_user_data = build_inspector_user_data(
                vendor, server, srv_data, region, zone, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
            )

            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))
            error_msgs = []
            output = []
            stack_opts = pulumi_stack_opts(error_msgs, output, instance_logger, instance_timing, server)
            # try with cloud_auto first, then retry without system_disk_category if needed
            current_instance_opts = copy.deepcopy(instance_opts)
            # Ensure first attempt has cloud_auto
            current_instance_opts["system_disk_category"] = "cloud_auto"
            for attempt in range(2):
                logger.info(f"Attempt {attempt + 1} for zone {zone} with instance_opts: {current_instance_opts}")
                try:
                    retry_locked(runner.create, vendor, pulumi_opts,
                                 resource_opts | dict(instance_opts=current_instance_opts, user_data=b64_user_data),
                                 stack_opts=stack_opts, instance_timing=instance_timing,
                                 error_msgs=error_msgs)
                    # empty it if create succeeded, just in case
                    error_msgs = []
                    done = True
                    instance_started = True
                    break
                except Exception as e:
                    # Check if the error is about disk category not being supported
                    error_text = str(e) + " " + " ".join(error_msgs) + " " + " ".join(output)
                    if "specified instance does not support this disk category" in error_text:
                        logger.exception(f"Disk category error, retrying without system_disk_category for zone {zone}")
                        # Remove system_disk_category for second attempt
                        current_instance_opts.pop("system_disk_category", None)
                        # clear error_msgs before retry
                        error_msgs = []
                        output = []
                        continue
                    # try the next zone; do not schedule delayed_destroy here — the stack
                    # name is alicloud.{region}.{instance} (zone is not part of it), so a
                    # background destroy from a failed zone would delete a later success
                    # in the same region; the next iteration's sync destroy handles cleanup
                    logger.exception(f"Couldn't start instance in zone {zone}")
                    break
            if done:
                break

    if vendor == "azure":
        # explicitly set SSH key from envvar
        resource_opts = dict(public_key=os.environ.get("SSH_PUBLIC_KEY"), instance=server, disk_size=VOLUME_SIZE)
        image_sku = "server"
        if "arm" in (srv_data.cpu_architecture or "").lower():
            image_sku = "server-arm64"
        done = False
        for region in candidate_regions(vendor, server, regions):
            logger.info(f"Trying {region}")
            resource_opts["region"] = region
            user_data, b64_user_data = build_inspector_user_data(
                vendor, server, srv_data, region, None, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
            )
            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))

            error_msgs = []
            output = []
            stack_opts = pulumi_stack_opts(error_msgs, output, instance_logger, instance_timing, server)
            for _ in range(2):
                # try normal images first, then gen1 if we get Hypervisor Generation '2' error
                try:
                    retry_locked(runner.create, vendor, pulumi_opts,
                                 resource_opts | dict(user_data=b64_user_data, image_sku=image_sku),
                                 stack_opts=stack_opts, instance_timing=instance_timing,
                                 error_msgs=error_msgs)
                    # empty it if create succeeded, just in case
                    error_msgs = []
                    done = True
                    instance_started = True
                    break
                except Exception:
                    if image_sku.endswith("-gen1"):
                        # we already know it's a gen1 instance, don't try to create twice with the same options
                        break
                    # The selected VM size 'Standard_A0' cannot boot Hypervisor Generation '2'. If this was a
                    # Create operation please check that the Hypervisor Generation of the Image matches the
                    # Hypervisor Generation of the selected VM Size. If this was an Update operation please select
                    # a Hypervisor Generation '2' VM Size. For more information, see https://aka.ms/azuregen2vm
                    if any(["cannot boot Hypervisor Generation '2'" in s for s in output]):
                        logger.exception(f"Hypervisor Generation error, image_sku={image_sku}, adding -gen1")
                        if "gen1" not in image_sku:
                            image_sku += "-gen1"
                        # The NIC will be blocked for 180s, so wait until we retry
                        # Nic(s) in request is reserved for another Virtual Machine for 180 seconds.
                        logger.info("Sleeping 180s to make the NIC free again")
                        time.sleep(180)
                        continue
                    logger.exception(f"Couldn't start instance, deleting the stack for {region} in the background")
                    # on failure, destroy the stack in the background (as we have to wait 180s for the NIC), so we're
                    # not blocking further tries
                    executor.submit(delayed_destroy, vendor, server, copy.deepcopy(resource_opts))
                    break
            if done:
                break

    if vendor == "gcp":
        resource_opts = dict(instance=server, disk_size=VOLUME_SIZE)
        # select the first zone from the list, work on a copy as we modify it
        bootdisk_init_opts = copy.deepcopy(default(getattr(sc_runner.resources, vendor).DEFAULTS, "bootdisk_init_opts"))
        # image/architecture are left unset here: sc_runner's resources_gcp()
        # always calls apply_gcp_boot_disk_defaults() before creating the
        # instance, which fills them in from the same Server.cpu_architecture
        # catalog field (now backed by GCP's real MachineType.architecture API
        # field) — duplicating that lookup here would be redundant.
        # -416 machines won't boot with ubuntu-2404 (or any other ubuntu images), so use Debian for them
        if server.endswith("-416"):
            if "arm" in srv_data.cpu_architecture:
                bootdisk_init_opts |= dict(image="debian-12-arm64")
            else:
                bootdisk_init_opts |= dict(image="debian-12")

        # e2 needs to be spot; we only have spot quota for GPU instances.
        # srv_data.accelerator_count is reliably populated by the crawler for every
        # GPU family (A2/A3/A4/A4X/G2/G4), unlike cpu_architecture above, so no
        # hardcoded series list is needed here — accelerators cannot live-migrate.
        # https://cloud.google.com/compute/docs/gpus/gpu-host-maintenance
        # https://cloud.google.com/compute/docs/instances/setting-vm-host-options
        has_gpu = (srv_data.accelerator_count or 0) > 0
        is_preemptible = server.startswith("e2") or has_gpu
        # a3-highgpu-{1,2,4}g (and other <8-GPU A3 sizes) reject the legacy
        # preemptible=true flag alone: without an explicit provisioning_model,
        # GCP silently keeps it STANDARD, which then conflicts with the
        # GPU-mandated onHostMaintenance=TERMINATE and fails with a confusing
        # "must have onHostMaintenance be [TERMINATE]. But was MIGRATE" error
        # (that's GCP's own internal recompute, not what we actually sent).
        # https://cloud.google.com/compute/docs/accelerator-optimized-machines
        # MachineType has no supported-onHostMaintenance field. Default to the
        # common policy (MIGRATE); on a policy reject, fall back to TERMINATE
        # (H4D, bare metal, large Z3, …). Spot/preemptible only allows TERMINATE.
        if is_preemptible:
            maintenance_attempts = ("TERMINATE",)
        else:
            maintenance_attempts = ("MIGRATE", "TERMINATE")
        scheduling_opts = dict(
            preemptible=is_preemptible,
            automatic_restart=False if is_preemptible else True,
            on_host_maintenance=maintenance_attempts[0],
            provisioning_model="SPOT" if is_preemptible else "STANDARD",
        )
        resource_opts |= dict(
            bootdisk_init_opts=bootdisk_init_opts,
            scheduling_opts=scheduling_opts,
        )
        # enable nested virtualization
        for zone in candidate_zones(vendor, server, zones):
            logger.info(f"Trying {zone}")
            resource_opts["zone"] = zone
            user_data, b64_user_data = build_inspector_user_data(
                vendor, server, srv_data, "", zone, timeout_mins, ssh_deploy_key_b64, repo_url_ssh
            )
            instance_opts = copy.deepcopy(default(getattr(sc_runner.resources, vendor).DEFAULTS, "instance_opts"))
            instance_opts |= dict(
                metadata_startup_script=user_data,
                advanced_machine_features=dict(enable_nested_virtualization=True),
            )
            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, pulumi_opts, resource_opts, stack_opts=dict(on_output=pulumi_on_output(instance_logger)))

            for attempt, maintenance in enumerate(maintenance_attempts):
                scheduling_opts["on_host_maintenance"] = maintenance
                logger.info(
                    f"GCP create attempt {attempt + 1} in {zone} with on_host_maintenance={maintenance}"
                )
                error_msgs = []
                pulumi_output = []
                stack_opts = pulumi_stack_opts(error_msgs, pulumi_output, instance_logger, instance_timing, server)
                try:
                    retry_locked(runner.create, vendor, pulumi_opts,
                                 resource_opts | dict(instance_opts=instance_opts),
                                 stack_opts=stack_opts, instance_timing=instance_timing,
                                 error_msgs=error_msgs)
                    # empty it if create succeeded, just in case
                    error_msgs = []
                    instance_started = True
                    break
                except Exception as e:
                    error_text = pulumi_error_text(e, error_msgs) + "\n" + "\n".join(pulumi_output)
                    can_fallback = (
                        attempt + 1 < len(maintenance_attempts)
                        and is_gcp_host_maintenance_policy_error(error_text)
                    )
                    if can_fallback:
                        nxt = maintenance_attempts[attempt + 1]
                        logger.exception(
                            f"on_host_maintenance={maintenance} rejected for {server}, "
                            f"retrying with {nxt}"
                        )
                        runner.destroy(
                            vendor,
                            pulumi_opts,
                            resource_opts,
                            stack_opts=dict(on_output=pulumi_on_output(instance_logger)),
                        )
                        continue
                    # on failure, try the next zone
                    logger.exception("Couldn't start instance")
                    if not error_msgs:
                        error_msgs.append(str(e))
                    break
            if instance_started:
                break

    if instance_started and instance_timing.complete() and any(task.always_run for task in tasks):
        with lock:
            repo.pull()
            record_timing_api(data_dir, instance_timing.start, instance_timing.end)
            repo.push_path(data_dir, f"Instance creation timing from {repo.gha_url()}")
    elif instance_started and any(task.always_run for task in tasks):
        logger.warning(
            f"Instance started but creation timing not captured for {vendor}/{server} "
            f"(start={instance_timing.start}, end={instance_timing.end})"
        )

    if not instance_started:
        record_instance_start_failure(lock, data_dir, tasks, error_msgs)


def thread_monitor(executor, interval=60):
    while True:
        running_tasks = [task for task in executor._threads if task.is_alive()]
        if not running_tasks:
            logger.info("Executor shut down and no tasks left")
            break
        frames = sys._current_frames()
        for t in running_tasks:
            logger.info(f"thread {t.name}: {frames.get(t.ident)}")
        time.sleep(interval)

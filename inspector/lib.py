from collections import defaultdict
from datetime import datetime, timedelta
from functools import cache
from itertools import chain
from pydantic import BaseModel
from queue import Queue
from typing import Callable
import copy
import docker
import hashlib
import inspect
import json
import logging
import math
import os
import repo
import subprocess
import threading
import time
import transform


META_NAME = "meta.json"
# add options to the task hash, whose function is to signal any
# changes in the tasks' runtime parameters, which might alter the output
TASK_HASH_KEYS = {"command", "transform_output", "image"}
# don't start task if it has already been started less than 2 hours ago
WAIT_SINCE_LAST_START = timedelta(hours=2)
# fail if a job has already started, but didn't produce output for 2 days
FAIL_IF_NO_OUTPUT = timedelta(days=7)
FAIL_ON_ERROR = timedelta(days=7)
# destroy the instance one hour after it has been started
DESTROY_AFTER = timedelta(hours=1)
DOCKER_OPTS = dict(detach=True, privileged=True)
DOCKER_OPTS_GPU = dict(device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])])


mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


class Meta(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    exit_code: int | None = None
    error_msg: str | None = None
    task_hash: str | None = None
    version: str | None = None
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    outputs: list[str] = []


class Task(BaseModel):
    vendors_only: set = set()  # run for these vendors only, empty means all
    servers_only: set[tuple] = set()  # run for these vendor/server pairs only, empty means all
    parallel: bool = False  # should we run this task concurrently with other tasks in the same priority group?
    priority: int | float = math.inf  # lower priority runs earlier, missing means last
    version_command: str | list | None = None  # command to run to get the version
    command: str | list  # command to run
    transform_output: list[Callable] = [transform.raw]  # functions to transform the output on the inspected node, write as raw if missing
    parse_output: list[Callable] = []  # functions to parse the already collected outputs from the repo
    rerun: timedelta | None = None  # re-run the task after a delay on successful execution, None means no re-evaluation
    timeout: timedelta = timedelta(minutes=30)  # timeout for the task
    name: str | None = None  # name of the task
    gpu: bool = False  # requires a machine with GPU(s)
    minimum_memory: float = 0  # minimum memory in GiBs for this test


class DockerTask(Task):
    image: str
    docker_opts: dict = DOCKER_OPTS
    version_docker_opts: dict = {}


def load_task_meta(task: Task, data_dir: str | os.PathLike, **kwargs) -> Meta:
    fn = os.path.join(data_dir, task.name, META_NAME)
    if os.path.exists(fn):
        with open(fn, "r") as f:
            meta = json.load(f)
        meta |= kwargs
        return Meta.model_validate(meta)
    else:
        return Meta(**kwargs)


def task_hash(task: Task) -> str:
    h = hashlib.sha1()
    for var in sorted(TASK_HASH_KEYS):
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
        if isinstance(task, Task) or isinstance(task, DockerTask):
            # task name becomes the variable's name
            task.name = name.lower()
            # only add the task if vendor is listed in vendors_only or if it's empty
            if vendor in task.vendors_only or not task.vendors_only:
                # execute parallel tasks first by negating the parallel option, so it gets forward during sorting
                taskgroups[(task.priority, not task.parallel)].append(task)
    return taskgroups


@cache
def get_tasks(vendor: str) -> list[Task]:
    taskgroups = get_taskgroups(vendor)
    return list(chain(*taskgroups.values()))


def should_start(task: Task, data_dir: str | os.PathLike, srv) -> bool:
    """Return True if we should start a server for this task."""
    meta = load_task_meta(task, data_dir)
    if meta.start and (datetime.now() - meta.start) <= WAIT_SINCE_LAST_START:
        logging.info(f"Skipping task {task.name}, last start: {meta.start}")
        return False
    thash = task_hash(task)
    if task.gpu and not srv.gpu_count:
        # skip tasks which require GPUs on a server which doesn't have one
        logging.info(f"Skipping task {task.name} because it requires GPU, but gpu_count is {srv.gpu_count}")
        return False
    # srv.memory_amount is MiB, minimum_memory is GiB
    if srv.memory_amount < task.minimum_memory * 1024:
        mem_gib = srv.memory_amount / 1024
        logging.info(f"Skipping task {task.name} because it requires {task.minimum_memory} GiB RAM, but this machine has only {mem_gib:.03}")
        return False
    if task.servers_only and (srv.vendor_id, srv.api_reference) not in task.servers_only:
        logging.info(f"Skipping task {task.name} because it is not enabled for {srv.vendor_id}/{srv.api_reference}")
        return False

    if meta.start:
        if (datetime.now() - meta.start) >= FAIL_IF_NO_OUTPUT and (meta.end is None or meta.exit_code is None):
            raise RuntimeError(f"{task.name} was started at {meta.start}, but didn't produce output!")
        if (datetime.now() - meta.start) >= FAIL_ON_ERROR and meta.exit_code != 0:
            raise RuntimeError(f"{task.name} was last started at {meta.start} and failed!")
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        # if rerun is set and there's a successful run, run the task again if rerun time interval has passed
        logging.info(f"Task {task.name} should be started: {task.rerun} has passed since last run")
        return True
    if meta.task_hash != thash:
        # This will be triggered when a task is first created or if it has been changed.
        logging.info(f"Task {task.name} should run as its task hash has changed: {meta.task_hash} -> {thash}")
        return True
    # Skip the task. If it was started, but hasn't yet produced output, we won't start a new run, as it would
    # ruin the above checks for failing outputs, and we don't want to constantly and silently restart the tasks.
    if meta.exit_code != 0:
        # don't report succeeded tasks
        logging.info(f"Skipping task {task.name}, meta: {meta}")
    return False


def should_run(task: Task, data_dir: str | os.PathLike, vendor: str, instance: str, gpu_count: int) -> bool:
    """Return True if we should run a task."""
    meta = load_task_meta(task, data_dir)
    thash = task_hash(task)
    # minimum_memory is GiB
    if mem_bytes < task.minimum_memory * 1024 ** 2:
        mem_gib = mem_bytes / 1024 ** 2
        logging.info(f"Skipping task {task.name} because it requires {task.minimum_memory} GiB RAM, but this machine has only {mem_gib:.03}")
        return False
    if task.gpu and not gpu_count:
        logging.info(f"Skipping task {task.name} because it requires GPU, but gpu_count is {gpu_count}")
        # skip tasks which require GPUs on a server which doesn't have one
        return False
    if task.servers_only and (vendor, instance) not in task.servers_only:
        logging.info(f"Skipping task {task.name} because it is not enabled for {vendor}/{instance}")
        return False
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        return True
    if meta.exit_code != 0 or meta.task_hash != thash:
        if meta.exit_code != 0:
            logging.info(f"Task {task.name} should run as last run has exit code: {meta.exit_code}")
        if meta.task_hash != thash:
            logging.info(f"Task {task.name} should run as its task hash has changed: {meta.task_hash} -> {thash}")
        return True
    logging.info(f"Skipping task {task.name}, {meta.end}, {meta.exit_code}")
    return False


def run_native(meta: Meta, task: Task, data_dir: str | os.PathLike) -> tuple[str | None, bytes, bytes]:
    ver = None
    try:
        if task.version_command:
            res = subprocess.run(task.version_command, capture_output=True, timeout=task.timeout.total_seconds())
            ver = res.stdout.strip().decode("utf-8")
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


def run_docker(meta: Meta, task: DockerTask, data_dir: str | os.PathLike) -> tuple[str | None, bytes, bytes]:
    ver = None
    stdout = stderr = b""
    docker_opts = copy.deepcopy(task.docker_opts)
    version_docker_opts = copy.deepcopy(task.version_docker_opts)
    if task.gpu:
        docker_opts |= DOCKER_OPTS_GPU
        version_docker_opts |= DOCKER_OPTS_GPU
    try:
        d = docker.from_env()
        if task.version_command:
            ver = d.containers.run(task.image, task.version_command, **version_docker_opts).strip().decode("utf-8")
        c = d.containers.run(task.image, task.command, **docker_opts)
    except Exception as e:
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
        c.stop(c)
        meta.error_msg = f"Execution timed out after {task.timeout.total_seconds()}s"
    meta.end = datetime.now()
    try:
        # wait for container exit/get output with 60s of docker timeout
        res = c.wait(timeout=60)
    except Exception as e:
        meta.error_msg = str(e)
        return ver, b"", b""

    meta.exit_code = res["StatusCode"]
    stdout = c.logs(stdout=True, stderr=False)
    stderr = c.logs(stdout=False, stderr=True)
    meta.stdout_bytes = len(stdout)
    meta.stderr_bytes = len(stderr)
    container_remove(c)
    return ver, stdout, stderr


def write_meta(meta: Meta, file: str | os.PathLike) -> None:
    task_dir = os.path.dirname(file)
    os.makedirs(task_dir, exist_ok=True)
    with open(file, "w") as f:
        f.write(meta.model_dump_json())


def run_task(q: Queue, data_dir: str | os.PathLike) -> None:
    while True:
        try:
            task = q.get()
            if not task:
                break
            meta = Meta(start=datetime.now(), task_hash=task_hash(task))
            failed = False
            try:
                if isinstance(task, DockerTask):
                    ver, stdout, stderr = run_docker(meta, task, os.path.join(data_dir, task.name))
                else:
                    ver, stdout, stderr = run_native(meta, task, os.path.join(data_dir, task.name))
                meta.version = ver
            except Exception as e:
                failed = True
                meta.exit_code = -1
                meta.error_msg = str(e)
            task_dir = os.path.join(data_dir, task.name)
            os.makedirs(task_dir, exist_ok=True)
            if not failed:
                for t in task.transform_output:
                    meta.outputs.extend(t(meta, task, task_dir, stdout, stderr))
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
        except Exception:
            raise
        finally:
            # ack the task, so run_tasks won't wait forever
            q.task_done()


def run_tasks(vendor, data_dir: str | os.PathLike, instance: str, gpu_count: int = 0, nthreads: int = 8):
    taskgroups = get_taskgroups(vendor)

    # initialize thread pool
    q: Queue = Queue(maxsize=nthreads)
    threads = []
    for _ in range(nthreads):
        threads.append(threading.Thread(target=run_task, args=(q, data_dir), daemon=True))
        threads[-1].start()

    # iterate over tasks, sorted by task key (running parallel tasks in a group first, then
    # non-parallel ones)
    for taskgroup in sorted(taskgroups.keys()):
        for task in taskgroups[taskgroup]:
            if not should_run(task, data_dir, vendor, instance, gpu_count):
                continue
            logging.info(f"Starting {task.name}")
            q.put(task)
            if not task.parallel:
                q.join()
        # wait at the end of the taskgroup
        q.join()
        # do a push at the end of each round
        if os.environ.get("GITHUB_TOKEN"):
            repo.push_path(data_dir, f"Inspecting server from {repo.gha_url()}")
    q.join()

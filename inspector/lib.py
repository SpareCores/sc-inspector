from collections import defaultdict
from datetime import datetime, timedelta
from functools import cache
from itertools import chain
from pydantic import BaseModel
from queue import Queue
from typing import Callable
import docker
import hashlib
import inspect
import json
import math
import os
import repo
import subprocess
import threading
import time
import transform


META_NAME = "meta.json"
# exclude these task options from the task hash, whose function is to signal any
# changes in the tasks' runtime parameters, which might alter the output
HASH_EXCLUDE = {"vendors_only", "parallel", "priority"}
# fail if a job has already started, but didn't produce output for 2 days
FAIL_IF_NO_OUTPUT = timedelta(days=2)
FAIL_ON_ERROR = timedelta(days=2)
# wait this much to start a task again if a server has already been started with the task, but not yet produced output
WAIT_BETWEEN_TASKS = timedelta(hours=1)
DOCKER_OPTS = dict(detach=True, privileged=True)
DOCKER_OPTS_GPU = dict(device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])


class Meta(BaseModel):
    start: datetime | None = None
    end: datetime | None = None
    exit_code: int | None = None
    error_msg: str | None = None
    task_hash: str | None = None
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    outputs: list[str] = []


class Task(BaseModel):
    vendors_only: set = set()  # run for these vendors only, empty means all
    parallel: bool = False  # should we run this task concurrently with other tasks in the same priority group?
    priority: int | float = math.inf  # lower priority runs earlier, missing means last
    command: str | list  # command to run
    transform_output: list[Callable] = [transform.raw]  # functions to transform the output on the inspected node, write as raw if missing
    parse_output: list[Callable] = []  # functions to parse the already collected outputs from the repo
    rerun: timedelta | None = None  # re-run the task after a delay on successful execution, None means no re-evaluation
    timeout: timedelta = timedelta(minutes=30)  # timeout for the task
    name: str  # name of the task
    gpu: bool = False  # requires a machine with GPU(s)


class DockerTask(Task):
    image: str
    docker_opts: dict = DOCKER_OPTS


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
    task_vars = sorted(set(task.model_fields.keys()).difference(HASH_EXCLUDE))
    h = hashlib.sha1()
    for var in task_vars:
        if var in ["parse_output"]:
            # leave these out from the hash, so a change in them won't trigger a re-run
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
    for name, model in inspect.getmembers(tasks):
        if inspect.isclass(model) and issubclass(model, Task) and model not in (Task, DockerTask):
            # I couldn't find a better way to have the model's class name inside the instance other than to
            # explicitly pass it as an argument
            obj = model(name=model.__name__.lower())
            # only add the task if vendor is listed in vendors_only or if it's empty
            if vendor in obj.vendors_only or not obj.vendors_only:
                # execute parallel tasks first by negating the parallel option, so it gets forward during sorting
                taskgroups[(obj.priority, not obj.parallel)].append(obj)
    return taskgroups


@cache
def get_tasks(vendor: str) -> list[Task]:
    taskgroups = get_taskgroups(vendor)
    return list(chain(*taskgroups.values()))


def should_start(task: Task, data_dir: str | os.PathLike, gpu_count: int) -> bool:
    """Return True if we should start a server for this task."""
    meta = load_task_meta(task, data_dir)
    thash = task_hash(task)
    if task.gpu and not gpu_count:
        # skip tasks which require GPUs on a server which doesn't have one
        return False
    if meta.start:
        if (datetime.now() - meta.start) >= FAIL_IF_NO_OUTPUT and (meta.end is None or meta.exit_code is None):
            raise RuntimeError(f"{task.name} was started at {meta.start}, but didn't produce output!")
        if (datetime.now() - meta.start) >= FAIL_ON_ERROR and meta.exit_code != 0:
            raise RuntimeError(f"{task.name} was last started at {meta.start} and failed!")
        if (datetime.now() - meta.start) < WAIT_BETWEEN_TASKS:
            return False
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        # if rerun is set and there's a successful run, run the task again if rerun time interval has passed
        return True
    if meta.task_hash != thash:
        return True
    return False


def should_run(task: Task, data_dir: str | os.PathLike, gpu_count: int) -> bool:
    """Return True if we should run a task."""
    meta = load_task_meta(task, data_dir)
    thash = task_hash(task)
    if task.gpu and not gpu_count:
        # skip tasks which require GPUs on a server which doesn't have one
        return False
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        return True
    if meta.exit_code != 0 or meta.task_hash != thash:
        return True
    return False


def run_native(meta: Meta, task: Task, data_dir: str | os.PathLike) -> tuple[bytes, bytes]:
    try:
        res = subprocess.run(task.command, capture_output=True, timeout=task.timeout.total_seconds())
        meta.end = datetime.now()
        meta.stdout_bytes = len(res.stdout)
        meta.stderr_bytes = len(res.stderr)
        meta.exit_code = res.returncode
        return res.stdout, res.stderr
    except Exception as e:
        meta.error_msg = str(e)
        return b"", b""


def container_remove(c):
    try:
        c.remove(force=True)
    except Exception:
        # don't fail if we couldn't remove the container
        pass


def run_docker(meta: Meta, task: DockerTask, data_dir: str | os.PathLike) -> tuple[bytes, bytes]:
    stdout = stderr = b""
    docker_opts = task.docker_opts
    if task.gpu:
        docker_opts |= DOCKER_OPTS_GPU
    try:
        d = docker.from_env()
        c = d.containers.run(task.image, task.command, **docker_opts)
    except Exception as e:
        meta.error_msg = str(e)
        return b"", b""

    ts = time.time() + task.timeout.total_seconds()
    while time.time() < ts:
        time.sleep(0.1)
        c.reload()
        if c.status == "exited":
            break
    else:
        # timed out, kill container, return with error
        container_remove(c)
        meta.error_msg = f"Execution timed out after {task.timeout.total_seconds()}s"
        return b"", b""
    meta.end = datetime.now()
    try:
        # wait for container exit/get output with 60s of docker timeout
        res = c.wait(timeout=60)
    except Exception as e:
        meta.error_msg = str(e)
        return b"", b""

    meta.exit_code = res["StatusCode"]
    stdout = c.logs(stdout=True, stderr=False)
    stderr = c.logs(stdout=False, stderr=True)
    meta.stdout_bytes = len(stdout)
    meta.stderr_bytes = len(stderr)
    container_remove(c)
    return stdout, stderr


def write_meta(meta: Meta, file: str | os.PathLike) -> None:
    task_dir = os.path.dirname(file)
    os.makedirs(task_dir, exist_ok=True)
    with open(file, "w") as f:
        f.write(meta.model_dump_json())
    if os.environ.get("GITHUB_TOKEN"):
        repo.push_path(task_dir, f"Inspecting server from {repo.gha_url()}")


def run_task(q: Queue, data_dir: str | os.PathLike) -> None:
    while True:
        task = q.get()
        if not task:
            break
        meta = Meta(start=datetime.now(), task_hash=task_hash(task))
        failed = False
        try:
            if isinstance(task, DockerTask):
                stdout, stderr = run_docker(meta, task, os.path.join(data_dir, task.name))
            else:
                stdout, stderr = run_native(meta, task, os.path.join(data_dir, task.name))
        except Exception as e:
            failed = True
            meta.exit_code = -1
            meta.error_msg = str(e)
        task_dir = os.path.join(data_dir, task.name)
        os.makedirs(task_dir, exist_ok=True)
        if not failed:
            for t in task.transform_output:
                meta.outputs.extend(t(meta, task, task_dir, stdout, stderr))
        try:
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
        except Exception:
            raise
        finally:
            q.task_done()


def run_tasks(vendor, data_dir: str | os.PathLike, gpu_count: int = 0, nthreads: int = 8):
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
            if not should_run(task, data_dir, gpu_count):
                continue
            q.put(task)
            if not task.parallel:
                q.join()
        # wait at the end of the taskgroup
        q.join()
    q.join()

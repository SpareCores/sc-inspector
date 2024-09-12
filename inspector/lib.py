from collections import defaultdict
from datetime import datetime, timedelta
from functools import cache
from itertools import chain
from pydantic import BaseModel
from queue import Queue
from typing import Callable
import base64
import copy
import docker
import hashlib
import inspect
import json
import logging
import math
import os
import psutil
import random
import re
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
# destroy the instance 15 mins after the last task has timed out
DESTROY_AFTER = timedelta(minutes=15)
DOCKER_OPTS = dict(detach=True, privileged=True, network_mode="host")
DOCKER_OPTS_GPU = dict(device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])])
# filter error_msg which is written to meta.json for these, we don't want to leak information
FILTER_ERROR_MSG = {
    re.compile(r"Submit a request for Quota increase at https.*to succeed\."),
}
# non-retryable Pulumi errors. The message that matches one of these regexes will be saved in meta.json.
PULUMI_ERRORS = {
    re.compile(r"error occurred"),   # AWS permanent error
    re.compile(r"creating .* error"),  # Azure errors
    re.compile(r"error waiting for.*to create"),   # GCP error
}
USER_DATA = """#!/bin/sh

# just to be sure, schedule a shutdown early
shutdown --no-wall +{SHUTDOWN_MINS}

export DEBIAN_FRONTEND=noninteractive
. /etc/os-release
apt-get update -y
# Add the required repositories to Apt sources:
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
# docker
curl -fsSL https://download.docker.com/linux/$ID/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
# nvidia drivers/toolkit in GPU_COUNT != 0
NVIDIA_PKGS=""
if [ {GPU_COUNT} -ne 0 ]; then
    add-apt-repository ppa:graphics-drivers/ppa -y
    # nvidia container toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    NVIDIA_PKGS="nvidia-driver-525 nvidia-container-toolkit"
fi
apt-get update -y >> /tmp/output 2>&1
apt-get install -y $NVIDIA_PKGS docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >> /tmp/output 2>&1
systemctl restart docker
# stop some services to preserve memory
snap stop amazon-ssm-agent >> /tmp/output 2>&1
systemctl stop chrony acpid cron multipathd snapd systemd-timedated unattended-upgrades polkit packagekit systemd-udevd >> /tmp/output 2>&1
# remove unwanted packages
apt-get autoremove -y apport unattended-upgrades snapd packagekit >> /tmp/output 2>&1
# https://github.com/NVIDIA/nvidia-container-toolkit/issues/202
# on some machines docker initialization times out with a lot of GPUs. Enable persistence mode to overcome that.
nvidia-smi -pm 1
docker run --rm --network=host --privileged -v /var/run/docker.sock:/var/run/docker.sock \
    -e GITHUB_TOKEN={GITHUB_TOKEN} \
    -e GITHUB_SERVER_URL={GITHUB_SERVER_URL} \
    -e GITHUB_REPOSITORY={GITHUB_REPOSITORY} \
    -e GITHUB_RUN_ID={GITHUB_RUN_ID} \
    -e BENCHMARK_SECRETS_PASSPHRASE={BENCHMARK_SECRETS_PASSPHRASE} \
    ghcr.io/sparecores/sc-inspector:main inspect --vendor {VENDOR} --instance {INSTANCE} --gpu-count {GPU_COUNT} >> /tmp/output 2>&1
poweroff
"""

mem_bytes = psutil.virtual_memory().available


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
    precheck_command: str | list | None = None  # check if we should run this task
    precheck_regex: str | None = None  # regular expression to match the precheck command's stdout


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
        logging.debug(f"{fn} not found, returning an empty Meta object")
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
        if (datetime.now() - meta.start) >= FAIL_ON_ERROR and meta.exit_code > 0:
            raise RuntimeError(f"{task.name} was last started at {meta.start} and failed!")
    if meta.end and task.rerun and (datetime.now() - meta.end) >= task.rerun and meta.exit_code == 0:
        # if rerun is set and there's a successful run, run the task again if rerun time interval has passed
        logging.info(f"Task {task.name} should be started: {task.rerun} has passed since last run")
        return True
    if meta.task_hash != thash:
        # This will be triggered when a task is first created or if it has been changed.
        logging.info(f"Task {task.name} should run as its task hash has changed: {meta.task_hash} -> {thash}")
        return True
    if meta.exit_code == -1 and "InsufficientInstanceCapacity" in meta.error_msg:
        # Retry insufficient instance capacity startup errors
        logging.info(f"Retrying task {task.name} due to insufficient capacity, meta: {meta}")
        return True
    if meta.exit_code == -1 and "PublicIPCountLimitReached" in meta.error_msg:
        # Retry Azure public IP count limit reached errors
        logging.info(f"Retrying task {task.name} due to PublicIPCountLimitReached, meta: {meta}")
        return True
    if not meta.start:
        logging.info(f"Task {task.name} should run, no start field")
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
    if mem_bytes < task.minimum_memory * 1024 ** 3:
        mem_gib = mem_bytes / 1024 ** 3
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
            ver = d.containers.run(task.image, task.version_command, **version_docker_opts).strip().decode("utf-8").replace("\n", ", ")
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
                # return something positive (negative will be inspector start errors) and outside normal return codes
                meta.exit_code = 256
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
            meta = load_task_meta(task, data_dir)
            if not should_run(task, data_dir, vendor, instance, gpu_count):
                if meta.start and meta.exit_code is None:
                    # update meta, if it doesn't yet have an exit code, so the monitoring won't fail on this
                    meta.end = datetime.now()
                    meta.exit_code = -2
                    meta.task_hash=task_hash(task)
                    meta.error_msg = "Task doesn't need to run on this instance"
                    write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
                continue
            if task.precheck_command and task.precheck_regex:
                check_res = subprocess.run(task.precheck_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if not re.search(task.precheck_regex, check_res.stdout, re.IGNORECASE):
                    logging.info("Task precheck_regex didn't match, skipping")
                    meta.end = datetime.now()
                    meta.exit_code = -3
                    meta.task_hash=task_hash(task)
                    meta.error_msg = "Task precheck_regex didn't match"
                    write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
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


@cache
def get_last_start(data_dir, vendor, server):
    tasks = list(get_tasks(vendor))
    if not tasks:
        # if there are no tasks, return a low value which can be used as a sort key
        return datetime.min
    tasks = [task for task in tasks if not (task.servers_only and (vendor, server) not in task.servers_only)]
    data_dir = os.path.join(data_dir, vendor, server)
    meta_starts = [load_task_meta(task, data_dir=data_dir).start for task in tasks]
    meta_starts = [start for start in meta_starts if start]
    if not meta_starts:
        # put it to the back
        last_start = datetime.min
    else:
        last_start = max(meta_starts)
    return last_start


def sort_available_servers(available_servers: dict, data_dir, reverse=True, max_start=None):
    if max_start:
        available_servers = {k: v for k, v in available_servers.items() if get_last_start(data_dir, k[0], k[1]) >= max_start}
    sorted_servers = sorted(available_servers.items(), key=lambda item: get_last_start(data_dir, item[0][0], item[0][1]), reverse=reverse)
    return sorted_servers


def custom_sort(lst, key):
    """Shuffles a list, but always returns `key` as the first element."""
    if key in lst:
        lst.remove(key)

    random.shuffle(lst)
    lst.insert(0, key)

    return lst


def pulumi_output_filter(message, error_msgs, output):
    # print output to the console with logger, so we have the dates
    logging.info(message)
    output.append(message)
    if any([regex.search(message) for regex in PULUMI_ERRORS]):
        error_msgs.append(message)


def pulumi_event_filter(event, error_msgs):
    try:
        if event.diagnostic_event.severity == "error" and any([regex.search(event.diagnostic_event.message) for regex in PULUMI_ERRORS]):
            error_msgs.append(event.diagnostic_event.message)
    except Exception:
        pass


def delayed_destroy(vendor, server, resource_opts):
    from sc_runner import runner

    # to be run in the background
    time.sleep(180)

    # change the thread name for logging
    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{server}"
    try:
        runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=logging.info))
    except Exception:
        logging.exception("Failed to destroy")


def remove_matches(regexes, input_string):
    for regex in regexes:
        input_string = re.sub(regex, '', input_string)
    return input_string


def retry_locked(func, *args, **kwargs):
    """Retry a pulumi function with random backoff for locking issues"""
    import pulumi

    for i in range(3):
        try:
            return func(*args, **kwargs)
        except pulumi.automation.errors.ConcurrentUpdateError:
            logging.exception(f"ConcurrentUpdateError, retry #{i}")
            time.sleep(random.randint(1, 5))
        except Exception:
            raise


def start_inspect(executor, lock, data_dir, vendor, server, tasks, srv_data, regions, zones):
    from sc_runner.resources import default
    from sc_runner import runner
    import sc_runner.resources

    # change the thread name for logging
    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{server}"

    error_msgs = []
    sum_timeout = timedelta()
    for task in tasks:
        meta = Meta(start=datetime.now(), task_hash=task_hash(task))
        write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
        sum_timeout += task.timeout
    timeout_mins = int(sum_timeout.total_seconds()/60)
    logging.info(f"Starting {vendor}/{server} with {timeout_mins}m timeout")
    if os.environ.get("GITHUB_TOKEN"):
        with lock:
            # don't run multiple pushes
            repo.push_path(data_dir, f"Starting server from {repo.gha_url()}")
    # start instance
    user_data = USER_DATA.format(
        GITHUB_TOKEN=os.environ.get("GITHUB_TOKEN"),
        GITHUB_SERVER_URL=os.environ.get("GITHUB_SERVER_URL"),
        GITHUB_REPOSITORY=os.environ.get("GITHUB_REPOSITORY"),
        GITHUB_RUN_ID=os.environ.get("GITHUB_RUN_ID"),
        BENCHMARK_SECRETS_PASSPHRASE=os.environ.get("BENCHMARK_SECRETS_PASSPHRASE"),
        VENDOR=vendor,
        INSTANCE=server,
        GPU_COUNT=srv_data.gpu_count,
        SHUTDOWN_MINS=timeout_mins,
    )
    b64_user_data = base64.b64encode(user_data.encode("utf-8")).decode("ascii")
    if vendor in ("aws", "gcp"):
        # get the copy (so we don't modify the original) of the default instance opts for the vendor and add ours
        instance_opts = copy.deepcopy(default(getattr(sc_runner.resources, vendor).DEFAULTS, "instance_opts"))
    if vendor == "aws":
        # we use the key_name in instance_opts instead of creating a new key
        resource_opts = dict(public_key="", instance=server)
        instance_opts |= dict(
            key_name="spare-cores",
            instance_initiated_shutdown_behavior="terminate",
        )
        for region in custom_sort(regions, "us-west-2"):
            logging.info(f"Trying {region}")
            resource_opts["region"] = region

            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=logging.info))
            error_msgs = []
            stack_opts = dict(on_output=logging.info, on_event=lambda event: pulumi_event_filter(event, error_msgs))
            try:
                retry_locked(runner.create,vendor, {},
                             resource_opts | dict(instance_opts=instance_opts, user_data=b64_user_data, disk_size=16),
                             stack_opts=stack_opts)
                # empty it if create succeeded, just in case
                error_msgs = []
                break
            except Exception:
                # on failure, try the next one
                logging.exception("Couldn't start instance")

    if vendor == "azure":
        # explicitly set SSH key from envvar
        resource_opts = dict(public_key=os.environ.get("SSH_PUBLIC_KEY"), instance=server)
        image_sku = "server"
        if "arm" in srv_data.cpu_architecture:
            image_sku = "server-arm64"
        done = False
        # we have larger quota in centralus, so prefer that
        for region in custom_sort(regions, "centralus"):
            logging.info(f"Trying {region}")
            resource_opts["region"] = region
            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=logging.info))

            error_msgs = []
            output = []
            # Azure native doesn't give sensible error events, use its output
            stack_opts = dict(on_output=lambda message: pulumi_output_filter(message, error_msgs, output))
            for _ in range(2):
                # try normal images first, then gen1 if we get Hypervisor Generation '2' error
                try:
                    retry_locked(runner.create, vendor, {},
                                 resource_opts | dict(user_data=b64_user_data, image_sku=image_sku),
                                 stack_opts=stack_opts)
                    # empty it if create succeeded, just in case
                    error_msgs = []
                    done = True
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
                        logging.exception(f"Hypervisor Generation error, image_sku={image_sku}, adding -gen1")
                        if "gen1" not in image_sku:
                            image_sku += "-gen1"
                        # The NIC will be blocked for 180s, so wait until we retry
                        # Nic(s) in request is reserved for another Virtual Machine for 180 seconds.
                        logging.info("Sleeping 180s to make the NIC free again")
                        time.sleep(180)
                        continue
                    logging.exception("Couldn't start instance, deleting the stack in the background")
                    # on failure, destroy the stack in the background (as we have to wait 180s for the NIC), so we're
                    # not blocking further tries
                    executor.submit(delayed_destroy, vendor, server, resource_opts)
                    break
            if done:
                break

    if vendor == "gcp":
        resource_opts = dict(instance=server)
        # select the first zone from the list, work on a copy as we modify it
        bootdisk_init_opts = copy.deepcopy(default(getattr(sc_runner.resources, vendor).DEFAULTS, "bootdisk_init_opts"))
        if "arm" in srv_data.cpu_architecture:
            bootdisk_init_opts |= dict(image="ubuntu-2404-lts-arm64")
        else:
            bootdisk_init_opts |= dict(image="ubuntu-2404-lts-amd64")
        # -416 machines won't boot with ubuntu-2404 (or any other ubuntu images), so use Debian for them
        if server.endswith("-416"):
            if "arm" in srv_data.cpu_architecture:
                bootdisk_init_opts |= dict(image="debian-12-arm64")
            else:
                bootdisk_init_opts |= dict(image="debian-12")

        # e2 needs to be spot, also, we have only spot quotas for selected GPU instances
        is_preemptible = server.startswith("e2") or srv_data.gpu_count > 0
        resource_opts |= dict(bootdisk_init_opts=bootdisk_init_opts,
                              scheduling_opts=dict(
                                  preemptible=is_preemptible,
                                  automatic_restart=False if is_preemptible else True,
                                  on_host_maintenance="TERMINATE")
                              )
        instance_opts |= dict(metadata_startup_script=user_data)

        for zone in zones:
            logging.info(f"Trying {zone}")
            resource_opts["zone"] = zone
            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, {}, resource_opts, stack_opts=dict(on_output=logging.info))

            error_msgs = []
            stack_opts = dict(on_output=logging.info, on_event=lambda event: pulumi_event_filter(event, error_msgs))
            try:
                retry_locked(runner.create, vendor, {},
                             resource_opts | dict(instance_opts=instance_opts),
                             stack_opts=stack_opts)
                # empty it if create succeeded, just in case
                error_msgs = []
                break
            except Exception:
                # on failure, try the next one
                logging.exception("Couldn't start instance")

    if error_msgs and os.environ.get("GITHUB_TOKEN"):
        # upload error message if we couldn't start the instance
        now = datetime.now()
        logging.info("Failed to start instance, uploading error messages")
        for task in tasks:
            meta = Meta(
                start=now,
                end=now,
                exit_code=-1,
                error_msg=remove_matches(FILTER_ERROR_MSG, error_msgs[-1]),
                task_hash=task_hash(task),
            )
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
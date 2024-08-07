"""
Import only the necessary modules here, because the `inspect` will run on small memory machines, so we want to
preserve memory for running the benchmarks.
All other modules should be imported lazily, where needed.
"""
from concurrent.futures import ThreadPoolExecutor
import click
import copy
import itertools
import lib
import logging
import os
import pulumi_aws as aws
import repo
import sys
import tempfile

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# We can't (yet) start these
EXCLUDE_INSTANCES: list[tuple[str, str]] = []

RESOURCE_OPTS = {
    "aws": dict(region="us-west-2"),
}
USER_DATA = """#!/bin/sh

# just to be sure, schedule a shutdown early
shutdown --no-wall +{SHUTDOWN_MINS}

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
# Add the required repositories to Apt sources:
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
# docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
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
systemctl stop chrony acpid cron multipathd snapd systemd-timedated >> /tmp/output 2>&1
# we don't want to submit crash dumps anywhere
apt-get autoremove -y apport >> /tmp/output 2>&1
docker run --rm --network=host -v /var/run/docker.sock:/var/run/docker.sock \
    -e GITHUB_TOKEN={GITHUB_TOKEN} \
    -e GITHUB_SERVER_URL={GITHUB_SERVER_URL} \
    -e GITHUB_REPOSITORY={GITHUB_REPOSITORY} \
    -e GITHUB_RUN_ID={GITHUB_RUN_ID} \
    -e BENCHMARK_SECRETS_PASSPHRASE={BENCHMARK_SECRETS_PASSPHRASE} \
    ghcr.io/sparecores/sc-inspector:main inspect --vendor {VENDOR} --instance {INSTANCE} --gpu-count {GPU_COUNT} >> /tmp/output 2>&1
poweroff
"""


def servers():
    from sc_crawler.tables import Server
    from sqlmodel import create_engine, Session, select
    import sc_data

    path = sc_data.db.path
    engine = create_engine(f"sqlite:///{path}")

    session = Session(engine)
    return session.exec(select(Server)).all()


def available_servers(vendor: str | None = None, region: str | None = None):
    """Return servers with the regions in which they are available."""
    from sc_crawler.tables import ServerPrice, Server, Region, Zone
    from sqlmodel import create_engine, Session, select
    import sc_data
    path = sc_data.db.path
    engine = create_engine(f"sqlite:///{path}")
    session = Session(engine)
    stmt = (select(
                   ServerPrice.vendor_id,
                   Region.api_reference,
                   Zone.api_reference,
                   Server
            )
            .where(ServerPrice.status == "ACTIVE")
            .where(ServerPrice.allocation == "ONDEMAND")
            .join(Region, Region.region_id == ServerPrice.region_id)
            .join(Zone, Zone.zone_id == ServerPrice.zone_id)
            .join(Server).order_by(ServerPrice.price)
    )
    if vendor:
        stmt = stmt.where(ServerPrice.vendor_id == vendor)
    if region:
        stmt = stmt.where(ServerPrice.region_id == region)
    servers = {}
    for vendor, region, zone, server in session.exec(stmt.distinct()).all():
        if (vendor, server.api_reference) in servers:
            if region not in servers[(vendor, server.api_reference)][1]:
                servers[(vendor, server.api_reference)][1].append(region)
            if zone not in servers[(vendor, server.api_reference)][2]:
                servers[(vendor, server.api_reference)][2].append(zone)
        else:
            servers[(vendor, server.api_reference)] = [server, [region], [zone]]
    return servers


@click.group()
@click.option("--repo-path", default=os.environ.get("REPO_PATH", os.getcwd()), help="Directory which contains the repository")
def cli(repo_path):
    pass


def pulumi_event_filter(event, error_msgs):
    try:
        if event.diagnostic_event.severity == "error" and "error occurred" in event.diagnostic_event.message:
            error_msgs.append(event.diagnostic_event.message)
    except Exception:
        pass


@cli.command()
@click.pass_context
@click.option("--exclude", type=(str, str), default=EXCLUDE_INSTANCES, multiple=True, help="Exclude $vendor $instance")
@click.option("--start-only", type=(str, str), multiple=True, help="Start only $vendor $instance")
def start(ctx, exclude, start_only):
    from datetime import datetime, timedelta
    from sc_runner import runner
    from sc_runner.resources import default, supported_vendors
    import base64
    import sc_runner.resources

    count = 0
    error_msgs = []
    for (vendor, server), (srv, regions, zones) in available_servers().items():
        if vendor not in supported_vendors:
            # sc-runner can't yet handle this vendor
            continue
        resource_opts = RESOURCE_OPTS.get(vendor, {})
        if vendor in {"aws"}:
            if resource_opts and RESOURCE_OPTS.get(vendor, {}).get("region") and RESOURCE_OPTS.get(vendor, {}).get("region") not in regions:
                # if this server is unavailable in the default region, use a different one
                resource_opts = copy.deepcopy(RESOURCE_OPTS.get(vendor))
                resource_opts["region"] = regions.pop(0)
            if resource_opts and RESOURCE_OPTS.get(vendor, {}).get("zone") and RESOURCE_OPTS.get(vendor, {}).get("zone") not in zones:
                # if this server is unavailable in the default zone, use a different one
                resource_opts = copy.deepcopy(RESOURCE_OPTS.get(vendor))
                resource_opts["zone"] = zones.pop(0)

        gpu_count = srv.gpu_count
        logging.info(f"Evaluating {vendor}/{server} with {gpu_count} GPUs")
        if (vendor, server) in exclude:
            logging.info(f"Excluding {vendor}/{server}")
            continue
        if start_only and (vendor, server) not in start_only:
            logging.info(f"Excluding {vendor}/{server} as --start-only {start_only} is given")
            continue
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(filter(lambda task: lib.should_start(task, data_dir, srv), lib.get_tasks(vendor)))
        if not tasks:
            logging.info(f"No tasks for {vendor}/{server}")
            continue
        sum_timeout = timedelta()
        for task in tasks:
            meta = lib.Meta(start=datetime.now(), task_hash=lib.task_hash(task))
            lib.write_meta(meta, os.path.join(data_dir, task.name, lib.META_NAME))
            sum_timeout += task.timeout
        timeout_mins = int(sum_timeout.total_seconds()/60)
        logging.info(f"Starting {vendor}/{server} with {timeout_mins}m timeout")
        if os.environ.get("GITHUB_TOKEN"):
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
            GPU_COUNT=gpu_count,
            SHUTDOWN_MINS=timeout_mins,
        )
        b64_user_data = base64.b64encode(user_data.encode("utf-8")).decode("ascii")
        # get default instance opts for the vendor and add ours
        instance_opts = default(getattr(sc_runner.resources, vendor).DEFAULTS, "instance_opts")
        if vendor == "aws":
            instance_opts |= dict(
                user_data_base64=b64_user_data,
                key_name="spare-cores",
                instance_initiated_shutdown_behavior="terminate",
                # increase root volume size
                root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(volume_size=16),
            )
            # before starting, destroy everything to make sure the user-data will run (this is the first boot)
            runner.destroy(vendor, {}, resource_opts | dict(instance=server))
            error_msgs = []
            stack_opts = dict(on_output=print, on_event=lambda event: pulumi_event_filter(event, error_msgs))
            try:
                runner.create(
                    vendor,
                    {},
                    resource_opts | dict(instance=server, instance_opts=instance_opts),
                    stack_opts=stack_opts,
                )
                # empty it if create succeeded, just in case
                error_msgs = []
            except Exception:
                # on failure, try the next one
                logging.exception("Couldn't start instance")

        if vendor == "gcp":
            # select the first zone from the list
            bootdisk_init_opts = default(getattr(sc_runner.resources, vendor).DEFAULTS, "bootdisk_init_opts")
            if "arm" in srv.cpu_architecture:
                bootdisk_init_opts |= dict(image="ubuntu-2404-lts-arm64", size=16)
            else:
                bootdisk_init_opts |= dict(image="ubuntu-2404-lts-amd64", size=16)

            # e2 needs to be spot, also, we have only spot quotas for selected GPU instances
            is_preemptible = server.startswith("e2") or gpu_count > 0
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
                runner.destroy(vendor, {}, resource_opts | dict(instance=server))

                error_msgs = []
                stack_opts = dict(on_output=print, on_event=lambda event: pulumi_event_filter(event, error_msgs))
                try:
                    runner.create(
                        vendor,
                        {},
                        resource_opts | dict(instance=server, instance_opts=instance_opts),
                        stack_opts=stack_opts,
                    )
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
                meta = lib.Meta(
                    start=now,
                    end=now,
                    exit_code=-1,
                    error_msg=error_msgs[-1],
                    task_hash=lib.task_hash(task),
                )
                lib.write_meta(meta, os.path.join(data_dir, task.name, lib.META_NAME))
            repo.push_path(data_dir, f"Failed to start server from {repo.gha_url()}")
        break
        # count += 1
        # if count == 3:
        #     break


def cleanup_task(vendor, server, data_dir, regions=[], zones=[]):
    """
    Some vendors support creating resources in regions, without explicitly specifying the zone, some don't,
    so we support both of them. We'll go through all regions or zones whatever is specified.
    """
    from datetime import datetime, timedelta
    from sc_runner import runner

    tasks = list(lib.get_tasks(vendor))
    if not tasks:
        return

    # see if we have to destroy the resources in the Pulumi stack
    destroy = ""

    start_times = []
    end_times = []
    already_ended = []
    sum_timeout = timedelta()
    # get the maximum possible timeout for this server
    max_timeout = sum([task.timeout for task in tasks if not (task.servers_only and (vendor, server) not in task.servers_only)], timedelta())
    for task in tasks:
        if task.servers_only and (vendor, server) not in task.servers_only:
            # this task doesn't run on this server, leave it out
            continue
        meta = lib.load_task_meta(task, data_dir=data_dir)
        if not meta.start:
            continue
        start_times.append(meta.start)
        if meta.end:
            end_times.append(meta.end)
            # the task has already finished
            already_ended.append(True)
        elif datetime.now() <= meta.start + max_timeout + lib.DESTROY_AFTER:
            # only count tasks which might already be running and leave out those, which have started before the maximum
            # timeout has passed to exclude hung tasks from the past
            already_ended.append(False)
            logging.info(f"{vendor}/{server} Adding task {task.name} timeout: {task.timeout}")
            sum_timeout += task.timeout

    if start_times and datetime.now() >= (wait_time := max(start_times) + max_timeout + lib.DESTROY_AFTER):
        destroy = f"Destroying {vendor}/{server}, last_start: {max(start_times)}, last timeout: {wait_time}"
    if end_times and datetime.now() >= (wait_time := max(end_times) + sum_timeout + lib.DESTROY_AFTER):
        # We can only estimate the time by which all tasks should have been completed, as the start date is added
        # to the git repository before the machine starts up, the machine startup can take a long time, and the
        # tasks do not necessarily run sequentially.
        # So here, we are using the sum_timeout, which is the sum of all timeouts for unfinished jobs.
        destroy = f"Destroying {vendor}/{server}, last_end: {max(end_times)}, wait time: {wait_time}"

    # if all tasks have already finished, we can destroy the stack
    if already_ended and all(already_ended):
        destroy = f"Destroying {vendor}/{server}, all tasks have finished"

    if destroy:
        resource_opts = copy.deepcopy(RESOURCE_OPTS.get(vendor, {}))
        with tempfile.TemporaryDirectory() as tempdir:
            pulumi_opts = dict(work_dir=tempdir)
            # use either regions or zones for cleaning up the stacks
            for opt_name, value in itertools.chain(zip(["region"] * len(regions), regions), zip(["zone"] * len(zones), zones)):
                resource_opts[opt_name] = value
                # In order not to cause unnecessary locks in Pulumi, we first get the stack's resources to see if
                # it's already empty, and in that case, we don't destroy it.
                try:
                    stack = runner.get_stack(vendor, pulumi_opts, resource_opts | dict(instance=server))
                except AttributeError:
                    # this vendor is not yet supported
                    return
                resources = stack.export_stack().deployment.get("resources", [])
                if len(resources) <= 1:
                    # a non-existent stack will have zero, a clean (already destroyed) stack should have exactly one
                    # resource (the Pulumi Stack itself). If we can see either of these, we have nothing to clean up.
                    logging.debug(f"Pulumi stack for {vendor}/{server} has {len(resources)} resources, no cleanup needed")
                    return
                logging.info(destroy)
                runner.destroy(vendor, pulumi_opts, resource_opts | dict(instance=server))


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=32, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
def cleanup(ctx, threads):
    from sc_runner import runner
    from sc_runner.resources import supported_vendors
    futures = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for (vendor, server), (_, regions, zones) in available_servers().items():
            if vendor not in supported_vendors:
                # sc-runner can't yet handle this vendor
                continue
            data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
            # process the cleanup in a thread as getting Pulumi state is very slow
            if vendor in {"aws"}:
                # with these vendors we use region to create resources, so clean those up
                futures.append([vendor, server, executor.submit(cleanup_task, vendor, server, data_dir, regions=regions)])
            else:
                # the others use zones
                futures.append([vendor, server, executor.submit(cleanup_task, vendor, server, data_dir, zones=zones)])
    for vendor, server, f in futures:
        try:
            f.result()
        except Exception:
            logging.exception(f"Error in processing {vendor}/{server}")
            raise


@cli.command()
@click.pass_context
def parse(ctx):
    """Parse already written outputs from the repo and write them back."""
    if os.environ.get("GITHUB_TOKEN"):
        # we must clone the repo before writing anything to it
        repo.get_repo()
    for srv in servers():
        vendor = srv.vendor_id
        server = srv.api_reference
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(lib.get_tasks(vendor))
        if not tasks:
            continue
        for task in tasks:
            meta = lib.load_task_meta(task, data_dir=data_dir)
            if not meta.outputs:
                continue
            for parse_func in task.parse_output:
                parse_func(meta, task, os.path.join(data_dir, task.name))
        if os.path.exists(data_dir) and os.environ.get("GITHUB_TOKEN"):
            repo.push_path(data_dir, f"Parsed outputs in {repo.gha_url()}")


@cli.command()
@click.pass_context
@click.option("--vendor", required=True, help="Vendor ID for this machine")
@click.option("--instance", required=True, help="Instance ID for this machine")
@click.option("--gpu-count", default=0, help="Number of GPUs")
@click.option("--threads", default=8, show_default=True, help="Parallelism in a given task group")
def inspect(ctx, vendor, instance, gpu_count, threads):
    """Run inspection on this machine."""
    if os.environ.get("GITHUB_TOKEN"):
        logging.info("Updating the git repo")
        # we must clone the repo before writing anything to it
        repo.get_repo()
    data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, instance)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lib.run_tasks(
        vendor,
        data_dir,
        instance=instance,
        gpu_count=gpu_count,
        nthreads=threads,
    )


if __name__ == "__main__":
    cli()

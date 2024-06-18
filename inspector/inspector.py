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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


EXCLUDE_INSTANCES: list[tuple[str, str]] = [
    ("aws", "a1.metal"),
    ("aws", "c7i-flex.4xlarge"),
    ("aws", "c7i-flex.8xlarge"),
    ("aws", "dl1.24xlarge"),
    ("aws", "f1.16xlarge"),
    ("aws", "f1.4xlarge"),
    ("aws", "g3.16xlarge"),
    ("aws", "g3.4xlarge"),
    ("aws", "g3.8xlarge"),
    ("aws", "g3s.xlarge"),
    ("aws", "g4ad.16xlarge"),
    ("aws", "g4ad.2xlarge"),
    ("aws", "g4ad.4xlarge"),
    ("aws", "g4ad.8xlarge"),
    ("aws", "g4ad.xlarge"),
    ("aws", "g4dn.12xlarge"),
    ("aws", "g4dn.16xlarge"),
    ("aws", "g4dn.2xlarge"),
    ("aws", "g4dn.4xlarge"),
    ("aws", "g4dn.8xlarge"),
    ("aws", "g4dn.metal"),
    ("aws", "g4dn.xlarge"),
    ("aws", "g5.12xlarge"),
    ("aws", "g5.16xlarge"),
    ("aws", "g5.24xlarge"),
    ("aws", "g5.2xlarge"),
    ("aws", "g5.48xlarge"),
    ("aws", "g5.4xlarge"),
    ("aws", "g5.8xlarge"),
    ("aws", "g5g.16xlarge"),
    ("aws", "g5g.4xlarge"),
    ("aws", "g5g.8xlarge"),
    ("aws", "g5g.metal"),
    ("aws", "g6.12xlarge"),
    ("aws", "g6.16xlarge"),
    ("aws", "g6.24xlarge"),
    ("aws", "g6.2xlarge"),
    ("aws", "g6.48xlarge"),
    ("aws", "g6.4xlarge"),
    ("aws", "g6.8xlarge"),
    ("aws", "g6.xlarge"),
    ("aws", "gr6.4xlarge"),
    ("aws", "gr6.8xlarge"),
    ("aws", "h1.16xlarge"),
    ("aws", "h1.4xlarge"),
    ("aws", "h1.8xlarge"),
    ("aws", "hpc6a.48xlarge"),
    ("aws", "hpc6id.32xlarge"),
    ("aws", "hpc7a.12xlarge"),
    ("aws", "hpc7a.24xlarge"),
    ("aws", "hpc7a.48xlarge"),
    ("aws", "hpc7a.96xlarge"),
    ("aws", "hpc7g.16xlarge"),
    ("aws", "hpc7g.4xlarge"),
    ("aws", "hpc7g.8xlarge"),
    ("aws", "i4i.32xlarge"),
    ("aws", "inf1.24xlarge"),
    ("aws", "inf1.2xlarge"),
    ("aws", "inf1.6xlarge"),
    ("aws", "inf1.xlarge"),
    ("aws", "inf2.24xlarge"),
    ("aws", "inf2.48xlarge"),
    ("aws", "inf2.8xlarge"),
    ("aws", "inf2.xlarge"),
    ("aws", "p2.16xlarge"),
    ("aws", "p2.8xlarge"),
    ("aws", "p2.xlarge"),
    ("aws", "p3.16xlarge"),
    ("aws", "p3.2xlarge"),
    ("aws", "p3.8xlarge"),
    ("aws", "p3dn.24xlarge"),
    ("aws", "p4d.24xlarge"),
    ("aws", "p4de.24xlarge"),
    ("aws", "p5.48xlarge"),
    ("aws", "r5a.24xlarge"),
    ("aws", "r6a.2xlarge"),
    ("aws", "r6a.32xlarge"),
    ("aws", "r6a.48xlarge"),
    ("aws", "r7a.24xlarge"),
    ("aws", "r7a.48xlarge"),
    ("aws", "r7a.medium"),
    ("aws", "r7a.metal-48xl"),
    ("aws", "r7iz.32xlarge"),
    ("aws", "r7iz.4xlarge"),
    ("aws", "r7iz.xlarge"),
    ("aws", "t1.micro"),
    ("aws", "t2.micro"),
    ("aws", "t2.nano"),
    ("aws", "t3a.micro"),
    ("aws", "t3a.nano"),
    ("aws", "t3.nano"),
    ("aws", "t4g.nano"),
    ("aws", "trn1.2xlarge"),
    ("aws", "trn1.32xlarge"),
    ("aws", "trn1n.32xlarge"),
    ("aws", "u-12tb1.112xlarge"),
    ("aws", "u-18tb1.112xlarge"),
    ("aws", "u-24tb1.112xlarge"),
    ("aws", "u-3tb1.56xlarge"),
    ("aws", "u-6tb1.112xlarge"),
    ("aws", "u-6tb1.56xlarge"),
    ("aws", "u7i-12tb.224xlarge"),
    ("aws", "u7in-16tb.224xlarge"),
    ("aws", "u7in-24tb.224xlarge"),
    ("aws", "u7in-32tb.224xlarge"),
    ("aws", "u-9tb1.112xlarge"),
    ("aws", "vt1.24xlarge"),
    ("aws", "vt1.3xlarge"),
    ("aws", "vt1.6xlarge"),
    ("aws", "x1.16xlarge"),
    ("aws", "x1.32xlarge"),
    ("aws", "x1e.16xlarge"),
    ("aws", "x1e.2xlarge"),
    ("aws", "x1e.32xlarge"),
    ("aws", "x1e.4xlarge"),
    ("aws", "x1e.8xlarge"),
    ("aws", "x1e.xlarge"),
    ("aws", "x2gd.12xlarge"),
    ("aws", "x2gd.16xlarge"),
    ("aws", "x2gd.2xlarge"),
    ("aws", "x2gd.4xlarge"),
    ("aws", "x2gd.8xlarge"),
    ("aws", "x2gd.large"),
    ("aws", "x2gd.medium"),
    ("aws", "x2gd.metal"),
    ("aws", "x2gd.xlarge"),
    ("aws", "x2idn.16xlarge"),
    ("aws", "x2idn.24xlarge"),
    ("aws", "x2idn.32xlarge"),
    ("aws", "x2idn.metal"),
    ("aws", "x2iedn.16xlarge"),
    ("aws", "x2iedn.24xlarge"),
    ("aws", "x2iedn.2xlarge"),
    ("aws", "x2iedn.32xlarge"),
    ("aws", "x2iedn.4xlarge"),
    ("aws", "x2iedn.8xlarge"),
    ("aws", "x2iedn.metal"),
    ("aws", "x2iedn.xlarge"),
    ("aws", "x2iezn.12xlarge"),
    ("aws", "x2iezn.2xlarge"),
    ("aws", "x2iezn.4xlarge"),
    ("aws", "x2iezn.6xlarge"),
    ("aws", "x2iezn.8xlarge"),
    ("aws", "x2iezn.metal"),
]

RESOURCE_OPTS = {
    "aws": dict(
        region="us-west-2",
    )
}
USER_DATA = """#!/bin/sh

# just to be sure, schedule a shutdown in 60 minutes
shutdown --no-wall +60

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y >> /tmp/output 2>&1
# https://ubuntu.com/server/docs/nvidia-drivers-installation
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin ubuntu-drivers-common >> /tmp/output 2>&1
ubuntu-drivers install >> /tmp/output 2>&1
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >> /tmp/output 2>&1
apt-get update -y >> /tmp/output 2>&1
apt-get install -y nvidia-container-toolkit >> /tmp/output 2>&1
systemctl restart docker
# stop some services to preserve memory
snap stop amazon-ssm-agent >> /tmp/output 2>&1
systemctl stop chrony acpid cron multipathd snapd systemd-timedated >> /tmp/output 2>&1
# we don't want to submit crash dumps anywhere
apt-get autoremove -y apport >> /tmp/output 2>&1
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
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
        Server)
            .join(Region, Region.region_id == ServerPrice.region_id)
            .join(Zone, Zone.zone_id == ServerPrice.zone_id)
            .join(Server)
    )
    if vendor:
        stmt = stmt.where(ServerPrice.vendor_id == vendor)
    if region:
        stmt = stmt.where(ServerPrice.region_id == region)
    servers = {}
    for vendor, region, zone, server in session.exec(stmt.distinct()).all():
        if (vendor, server.api_reference) in servers:
            servers[(vendor, server.api_reference)][1].add(region)
            servers[(vendor, server.api_reference)][2].add(zone)
        else:
            servers[(vendor, server.api_reference)] = [server, {region}, {zone}]
    return servers


@click.group()
@click.option("--repo-path", default=os.environ.get("REPO_PATH", os.getcwd()), help="Directory which contains the repository")
def cli(repo_path):
    pass


@cli.command()
@click.pass_context
@click.option("--exclude", type=(str, str), default=EXCLUDE_INSTANCES, multiple=True, help="Exclude $vendor $instance")
@click.option("--start-only", type=(str, str), multiple=True, help="Start only $vendor $instance")
def start(ctx, exclude, start_only):
    from datetime import datetime
    from sc_runner import runner
    from sc_runner.resources import default, supported_vendors
    import base64
    import sc_runner.resources

    count = 0
    for (vendor, server), (srv, regions, zones) in available_servers().items():
        if vendor not in supported_vendors:
            # sc-runner can't yet handle this vendor
            continue
        resource_opts = RESOURCE_OPTS.get(vendor, {})
        if resource_opts and RESOURCE_OPTS.get(vendor, {}).get("region") not in regions:
            # if this server is unavailable in the default region, use a different one
            resource_opts = copy.deepcopy(RESOURCE_OPTS.get(vendor))
            resource_opts["region"] = regions.pop()

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
        logging.info(f"Starting {vendor}/{server}")
        for task in tasks:
            meta = lib.Meta(start=datetime.now(), task_hash=lib.task_hash(task))
            lib.write_meta(meta, os.path.join(data_dir, task.name, lib.META_NAME))
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
        if vendor == "gcp":
            # select the first zone from the list
            bootdisk_init_opts = default(getattr(sc_runner.resources, vendor).DEFAULTS, "bootdisk_init_opts")
            if "arm" in srv.cpu_architecture:
                bootdisk_init_opts |= dict(image="ubuntu-2404-lts-arm64", size=16)
            else:
                bootdisk_init_opts |= dict(image="ubuntu-2404-lts-amd64", size=16)
            resource_opts |= dict(zone=zones.pop(), bootdisk_init_opts=bootdisk_init_opts)
            instance_opts |= dict(metadata_startup_script=user_data)
        # before starting, destroy everything to make sure the user-data will run (this is the first boot)
        runner.destroy(vendor, {}, resource_opts | dict(instance=server))
        try:
            runner.create(vendor, {}, resource_opts | dict(instance=server, instance_opts=instance_opts))
        except Exception:
            # on failure, try the next one
            logging.exception("Couldn't start instance")
            break
            continue

        # XXX temporary
        break
        count += 1
        if count == 3:
            # start three per round
            break


def cleanup_task(vendor, server, data_dir, regions=[], zones=[]):
    """
    Some vendors support creating resources in regions, without explicitly specifying the zone, some don't,
    so we support both of them. We'll go through all regions or zones whatever is specified.
    """
    from datetime import datetime
    from sc_runner import runner

    tasks = list(lib.get_tasks(vendor))
    if not tasks:
        return

    # see if we have to destroy the resources in the Pulumi stack
    destroy = ""

    start_times = []
    already_ended = []
    for task in tasks:
        meta = lib.load_task_meta(task, data_dir=data_dir)
        if not meta.start:
            continue
        start_times.append(meta.start)
        if meta.end:
            if meta.end >= meta.start:
                already_ended.append(True)
        else:
            already_ended.append(False)

    # if all tasks have already finished, we can destroy the stack
    if already_ended and all(already_ended):
        destroy = f"Destroying {vendor}/{server}, all tasks have finished"

    # if DESTROY_AFTER has already passed since the newest start time, we should destroy the stack/instance
    if start_times:
        last_start = max(start_times)
        if datetime.now() - lib.DESTROY_AFTER >= last_start:
            destroy = f"Destroying {vendor}/{server}, last task start date: {last_start}"

    if destroy:
        resource_opts = copy.deepcopy(RESOURCE_OPTS.get(vendor), {})
        # use either regions or zones for cleaning up the stacks
        for opt_name, value in itertools.chain(zip(["region"] * len(regions), regions), zip(["zone"] * len(zones), zones)):
            resource_opts[opt_name] = value
            # In order not to cause unnecessary locks in Pulumi, we first get the stack's resources to see if
            # it's already empty, and in that case, we don't destroy it.
            try:
                stack = runner.get_stack(vendor, {}, resource_opts | dict(instance=server))
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
            runner.destroy(vendor, {}, resource_opts | dict(instance=server))


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=32, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
def cleanup(ctx, threads):
    from sc_runner import runner
    from sc_runner.resources import supported_vendors
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for (vendor, server), (_, regions, zones) in available_servers().items():
            if vendor not in supported_vendors:
                # sc-runner can't yet handle this vendor
                continue
            data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
            # process the cleanup in a thread as getting Pulumi state is very slow
            if vendor in {"aws"}:
                # with these vendors we use region to create resources, so clean those up
                executor.submit(cleanup_task, vendor, server, data_dir, regions=regions)
            else:
                # the others use zones
                executor.submit(cleanup_task, vendor, server, data_dir, zones=zones)


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
    lib.run_tasks(vendor, data_dir, gpu_count=gpu_count, nthreads=threads)


if __name__ == "__main__":
    cli()

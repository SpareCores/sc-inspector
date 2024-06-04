"""
Import only the necessary modules here, because the `inspect` will run on small memory machines, so we want to
preserve memory for running the benchmarks.
All other modules should be imported lazily, where needed.
"""
from concurrent.futures import ThreadPoolExecutor
import click
import lib
import logging
import os
import repo
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


EXCLUDE_INSTANCES: list[list[str]] = [
    # ["aws", "m5.xlarge"]
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

    for srv in servers():
        vendor = srv.vendor_id
        if vendor not in supported_vendors:
            # sc-runner can't yet handle this vendor
            continue
        server = srv.api_reference
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
        # start instance
        b64_user_data = base64.b64encode(
            USER_DATA.format(
                GITHUB_TOKEN=os.environ.get("GITHUB_TOKEN"),
                GITHUB_SERVER_URL=os.environ.get("GITHUB_SERVER_URL"),
                GITHUB_REPOSITORY=os.environ.get("GITHUB_REPOSITORY"),
                GITHUB_RUN_ID=os.environ.get("GITHUB_RUN_ID"),
                BENCHMARK_SECRETS_PASSPHRASE=os.environ.get("BENCHMARK_SECRETS_PASSPHRASE"),
                VENDOR=vendor,
                INSTANCE=server,
                GPU_COUNT=gpu_count,
            ).encode("utf-8")
        ).decode("ascii")
        # get default instance opts for the vendor and add ours
        instance_opts = default(getattr(sc_runner.resources, vendor).DEFAULTS, "instance_opts")
        instance_opts |= dict(user_data_base64=b64_user_data, key_name="spare-cores")
        runner.create(vendor, {}, RESOURCE_OPTS.get(vendor) | dict(instance=server, instance_opts=instance_opts))
        # temporary
        break


def cleanup_task(vendor, server, data_dir):
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
        # In order not to cause unnecessary locks in Pulumi, we first get the stack's resources to see if
        # it's already empty, and in that case, we don't destroy it.
        try:
            stack = runner.get_stack(vendor, {}, RESOURCE_OPTS.get(vendor, {}) | dict(instance=server))
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
        runner.destroy(vendor, {}, RESOURCE_OPTS.get(vendor) | dict(instance=server))


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=32, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
def cleanup(ctx, threads):
    from sc_runner import runner
    from sc_runner.resources import supported_vendors
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for srv in servers():
            vendor = srv.vendor_id
            if vendor not in supported_vendors:
                # sc-runner can't yet handle this vendor
                continue
            server = srv.api_reference
            data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
            # process this in a thread as getting Pulumi state is very slow
            executor.submit(cleanup_task, vendor, server, data_dir)


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

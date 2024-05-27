from datetime import datetime
from sc_crawler.tables import Server
from sc_runner import runner
from sqlmodel import create_engine, Session, select
from sc_runner.resources import default
import base64
import click
import lib
import logging
import os
import repo
import sc_data
import sc_runner.resources
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
apt-get update -y
# https://ubuntu.com/server/docs/nvidia-drivers-installation
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin ubuntu-drivers-common
ubuntu-drivers install
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
    for srv in servers():
        vendor = srv.vendor_id
        server = srv.api_reference
        gpu_count = srv.gpu_count
        if (vendor, server) in exclude:
            continue
        if start_only and (vendor, server) not in start_only:
            continue
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(filter(lambda task: lib.should_start(task, data_dir, gpu_count), lib.get_tasks(vendor)))
        if not tasks:
            continue
        print("start", vendor, server)
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


@cli.command()
@click.pass_context
def cleanup(ctx):
    for srv in servers():
        vendor = srv.vendor_id
        server = srv.api_reference
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(lib.get_tasks(vendor))
        if not tasks:
            continue
        for task in tasks:
            meta = lib.load_task_meta(task, data_dir=data_dir)
            if not meta.start:
                continue
            # give a little time and then destroy everything in the Pulumi stack for that server
            if datetime.now() - lib.WAIT_BETWEEN_TASKS * 1.25 >= meta.start:
                print(f"Destroying {vendor}/{server}")
                runner.destroy(vendor, {}, RESOURCE_OPTS.get(vendor) | dict(instance=server))


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
        # we must clone the repo before writing anything to it
        repo.get_repo()
    data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, instance)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lib.run_tasks(vendor, data_dir, gpu_count=gpu_count, nthreads=threads)


if __name__ == "__main__":
    cli()

from sc_crawler.tables import Server
from sqlmodel import create_engine, Session, select
from datetime import datetime
import click
import lib
import logging
import os
import sc_data
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


EXCLUDE_INSTANCES: list[list[str]] = [
    # ["aws", "m5.xlarge"]
]


def servers():
    path = sc_data.db.path
    engine = create_engine(f"sqlite:///{path}")

    session = Session(engine)

    for server in session.exec(select(Server)).all():
        yield server.vendor_id, server.name


@click.group()
@click.option("--repo-path", default=os.environ.get("REPO_PATH", os.getcwd()), help="Directory which contains the repository")
def cli(repo_path):
    pass


@cli.command()
@click.pass_context
@click.option("--exclude", type=(str, str), default=EXCLUDE_INSTANCES, multiple=True, help="Exclude $vendor $instance")
@click.option("--start-only", type=(str, str), multiple=True, help="Start only $vendor $instance")
def start(ctx, exclude, start_only):
    for vendor, server in servers():
        if (vendor, server) in exclude:
            continue
        if start_only and (vendor, server) not in start_only:
            continue
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(filter(lambda task: lib.should_start(task, data_dir), lib.get_tasks(vendor)))
        if not tasks:
            continue
        print("start", vendor, server)
        for task in tasks:
            meta = lib.Meta(start=datetime.now(), task_hash=lib.task_hash(task))
            lib.write_meta(meta, os.path.join(data_dir, task.name, lib.META_NAME))
        # start instance
        # make meta modifications commit/push it to the repo


@cli.command()
def cleanup():
    click.echo("cleanup")


@cli.command()
@click.pass_context
@click.option("--vendor", required=True, help="Vendor ID for this machine")
@click.option("--instance", required=True, help="Instance ID for this machine")
@click.option("--threads", default=8, show_default=True, help="Parallelism in a given task group")
def inspect(ctx, vendor, instance, threads):
    """Run inspection on this machine."""
    data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, instance)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lib.run_tasks(vendor, data_dir, nthreads=threads)


if __name__ == "__main__":
    cli()

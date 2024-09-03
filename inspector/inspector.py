"""
Import only the necessary modules here, because the `inspect` will run on small memory machines, so we want to
preserve memory for running the benchmarks.
All other modules should be imported lazily, where needed.
"""
from functools import cache
import click
import itertools
import lib
import logging
import os
import repo
import sys
import tempfile

RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"


# Create a formatter string with color codes
formatter_str = (
    f"{YELLOW}%(asctime)s{RESET}/"
    f"{GREEN}%(levelname)s{RESET}/{RED}%(threadName)s{RESET}: "
    "%(message)s"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=formatter_str,
    datefmt="%Y-%m-%d %H:%M:%S"
)
lib.logging = logging
# We can't (yet) start these
EXCLUDE_INSTANCES: list[tuple[str, str]] = []


@cache
def get_regions(vendor: str):
    """Return all regions for a vendor."""
    from sc_crawler.tables import ServerPrice, Server, Region, Zone
    from sqlmodel import create_engine, Session, select
    import sc_data
    path = sc_data.db.path
    engine = create_engine(f"sqlite:///{path}")
    session = Session(engine)
    stmt = select(Region).where(Region.vendor_id==vendor)
    return [region.region_id for region in session.exec(stmt.distinct()).all()]


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


@cli.command()
@click.pass_context
@click.option("--exclude", type=(str, str), default=EXCLUDE_INSTANCES, multiple=True, help="Exclude $vendor $instance")
@click.option("--start-only", type=(str, str), multiple=True, help="Start only $vendor $instance")
def start(ctx, exclude, start_only):
    from sc_runner.resources import supported_vendors
    import concurrent.futures
    import threading

    threading.current_thread().name = "main"

    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    count = 0
    lock = threading.Lock()
    for (vendor, server), (srv_data, regions, zones) in available_servers().items():
        if vendor not in supported_vendors:
            # sc-runner can't yet handle this vendor
            continue
        if vendor != "azure":
            continue
        gpu_count = srv_data.gpu_count
        logging.info(f"Evaluating {vendor}/{server} with {gpu_count} GPUs")
        if (vendor, server) in exclude:
            logging.info(f"Excluding {vendor}/{server}")
            continue
        if start_only and (vendor, server) not in start_only:
            logging.info(f"Excluding {vendor}/{server} as --start-only {start_only} is given")
            continue
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
        tasks = list(filter(lambda task: lib.should_start(task, data_dir, srv_data), lib.get_tasks(vendor)))
        if not tasks:
            logging.info(f"No tasks for {vendor}/{server}")
            continue
        f = executor.submit(lib.start_inspect, executor, lock, data_dir, vendor, server, tasks, srv_data, regions, zones)
        futures[f] = (vendor, server)
        count += 1
        if count == 8:
            break
    for f in concurrent.futures.as_completed(futures):
        vendor, server = futures[f]
        try:
            result = f.result()
        except Exception:
            logging.exception(f"Inspection for {vendor}/{server} raised an exception")
    logging.info("Start completed")
    executor.shutdown(wait=True)
    logging.info("Waiting for executors to finish")
    if os.environ.get("GITHUB_TOKEN"):
        repo.push_path(os.path.join(ctx.parent.params["repo_path"], "data"), f"Start finished {repo.gha_url()}")
        logging.info("Git push successful")


def cleanup_task(vendor, server, data_dir, regions=[], zones=[], force=False):
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
        # safety net: after the max timeout has passed, the machine must be terminated
        destroy = f"Destroying {vendor}/{server}, last_start: {max(start_times)}, last timeout: {wait_time}"
    if start_times and datetime.now() >= (wait_time := max(start_times) + sum_timeout + lib.DESTROY_AFTER):
        # We can only estimate the time by which all tasks should have been completed, as the start date is added
        # to the git repository before the machine starts up, the machine startup can take a long time, and the
        # tasks do not necessarily run sequentially.
        # So here, we are using the sum_timeout, which is the sum of all timeouts for unfinished jobs.
        destroy = f"Destroying {vendor}/{server}, last_start: {max(start_times)}, wait time: {wait_time}"

    # if all tasks have already finished, we can destroy the stack
    if already_ended and all(already_ended):
        destroy = f"Destroying {vendor}/{server}, all tasks have finished"

    if not destroy and not start_times and force:
        # forced cleanup, even if there are metas for the server (might be due to a forced retry from git, by deleting
        # the files)
        destroy = f"Forced cleanup of {vendor}/{server}"

    if destroy:
        resource_opts = {}
        with tempfile.TemporaryDirectory() as tempdir:
            pulumi_opts = dict(work_dir=tempdir)
            # use either regions or zones for cleaning up the stacks
            for opt_name, value in itertools.chain(zip(["region"] * len(regions), regions), zip(["zone"] * len(zones), zones)):
                resource_opts[opt_name] = value
                # if forced, we do a full refresh and delete for all resources
                if not force:
                    # In order not to cause unnecessary locks in Pulumi, we first get the stack's resources to see if
                    # it's already empty, and in that case, we don't destroy it.
                    try:
                        stack = runner.get_stack(vendor, pulumi_opts, resource_opts | dict(instance=server))
                    except AttributeError:
                        logging.exception("Couldn't get stack")
                        # this vendor is not yet supported
                        return
                    resources = stack.export_stack().deployment.get("resources", [])
                    if len(resources) <= 1:
                        # a non-existent stack will have zero, a clean (already destroyed) stack should have exactly one
                        # resource (the Pulumi Stack itself). If we can see either of these, we have nothing to clean up.
                        logging.info(f"Pulumi stack for {vendor}/{value}/{server} has {len(resources)} resources, no cleanup needed")
                        continue
                logging.info(destroy)
                runner.destroy_stack(vendor, pulumi_opts, resource_opts | dict(instance=server))


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=32, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
@click.option("--force/--no-force", type=bool, default=False, help="Do a cleanup even if there's no meta for the server")
@click.option("--all-regions/--no-all-regions", type=bool, default=False, help="Clean up in all regions, not just in those which list the server as available")
@click.option("--lookback-mins", type=int, show_default=True, help="Only clean up those instances that started at most this many minutes ago")
def cleanup(ctx, threads, force, all_regions, lookback_mins):
    from sc_runner.resources import supported_vendors
    from datetime import datetime, timedelta
    import concurrent.futures

    max_start = None
    if lookback_mins:
        max_start = datetime.now() - timedelta(minutes=lookback_mins)
    futures = []
    servers = lib.sort_available_servers(
        available_servers(),
        data_dir=os.path.join(ctx.parent.params["repo_path"], "data"),
        max_start=max_start,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for (vendor, server), (_, regions, zones) in servers:
            if all_regions:
                regions = get_regions(vendor)
            if vendor not in supported_vendors:
                # sc-runner can't yet handle this vendor
                continue
            data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, server)
            # process the cleanup in a thread as getting Pulumi state is very slow
            if vendor in {"gcp"}:
                # we use zones with these vendors
                futures.append([vendor, server, executor.submit(cleanup_task, vendor, server, data_dir, zones=zones, force=force)])
            else:
                # others use regions
                futures.append([vendor, server, executor.submit(cleanup_task, vendor, server, data_dir, regions=regions, force=force)])

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
    # Disable OOM killer for this task as Linux tends to kill this instead of benchmarks, like bw_mem
    pid=os.getpid()
    with open(f"/proc/{pid}/oom_adj", mode="w+") as f:
        f.write("-17")

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

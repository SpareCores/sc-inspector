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
EXCLUDE_INSTANCES: list[tuple[str, str]] = [
    ("aws", "t3.nano"),  # the currently used software stack doesn't fit onto this instance
    ("aws", "f2.48xlarge"),
    ("aws", "p4de.24xlarge"),
    ("aws", "p5.4xlarge"),
    ("aws", "p5.48xlarge"),
    ("aws", "p5e.48xlarge"),
    ("aws", "p5en.48xlarge"),
    ("aws", "p6-b200.48xlarge"),
    ("aws", "p6-b300.48xlarge"),
    ("aws", "trn2.48xlarge"),
    ("aws", "u7i-12tb.224xlarge"),
    ("aws", "u7i-8tb.112xlarge"),
    ("aws", "u7in-16tb.224xlarge"),
    ("aws", "u7in-24tb.224xlarge"),
    ("aws", "u7in-32tb.224xlarge"),
    ("azure", "Standard_A0"),
    ("azure", "Standard_A1"),
    ("azure", "Standard_A2"),
    ("azure", "Standard_A3"),
    ("azure", "Standard_A4"),
    ("azure", "Standard_A5"),
    ("azure", "Standard_A6"),
    ("azure", "Standard_A7"),
    ("azure", "Standard_D96ads_v6"),
    ("azure", "Standard_D96alds_v6"),
    ("azure", "Standard_D96als_v6"),
    ("azure", "Standard_D96as_v6"),
    ("azure", "Standard_DC16ads_v5"),
    ("azure", "Standard_DC16as_v5"),
    ("azure", "Standard_DC16eds_v5"),
    ("azure", "Standard_DC16es_v5"),
    ("azure", "Standard_DC2ads_v5"),
    ("azure", "Standard_DC2as_v5"),
    ("azure", "Standard_DC2eds_v5"),
    ("azure", "Standard_DC2es_v5"),
    ("azure", "Standard_DC2s"),
    ("azure", "Standard_DC32ads_v5"),
    ("azure", "Standard_DC32as_v5"),
    ("azure", "Standard_DC32eds_v5"),
    ("azure", "Standard_DC32es_v5"),
    ("azure", "Standard_DC48ads_v5"),
    ("azure", "Standard_DC48as_v5"),
    ("azure", "Standard_DC48eds_v5"),
    ("azure", "Standard_DC48es_v5"),
    ("azure", "Standard_DC4ads_v5"),
    ("azure", "Standard_DC4as_v5"),
    ("azure", "Standard_DC4eds_v5"),
    ("azure", "Standard_DC4es_v5"),
    ("azure", "Standard_DC4s"),
    ("azure", "Standard_DC64ads_v5"),
    ("azure", "Standard_DC64as_v5"),
    ("azure", "Standard_DC64eds_v5"),
    ("azure", "Standard_DC64es_v5"),
    ("azure", "Standard_DC8ads_v5"),
    ("azure", "Standard_DC8as_v5"),
    ("azure", "Standard_DC8eds_v5"),
    ("azure", "Standard_DC8es_v5"),
    ("azure", "Standard_DC8_v2"),
    ("azure", "Standard_DC96ads_cc_v5"),
    ("azure", "Standard_DC96ads_v5"),
    ("azure", "Standard_DC96as_cc_v5"),
    ("azure", "Standard_DC96as_v5"),
    ("azure", "Standard_DC96eds_v5"),
    ("azure", "Standard_DC96es_v5"),
    ("azure", "Standard_E96ads_v6"),
    ("azure", "Standard_E96as_v6"),
    ("azure", "Standard_EC128eds_v5"),
    ("azure", "Standard_EC128es_v5"),
    ("azure", "Standard_EC16ads_v5"),
    ("azure", "Standard_EC16as_v5"),
    ("azure", "Standard_EC16eds_v5"),
    ("azure", "Standard_EC16es_v5"),
    ("azure", "Standard_EC20ads_v5"),
    ("azure", "Standard_EC20as_v5"),
    ("azure", "Standard_EC2ads_v5"),
    ("azure", "Standard_EC2as_v5"),
    ("azure", "Standard_EC2eds_v5"),
    ("azure", "Standard_EC2es_v5"),
    ("azure", "Standard_EC32ads_v5"),
    ("azure", "Standard_EC32as_v5"),
    ("azure", "Standard_EC32eds_v5"),
    ("azure", "Standard_EC32es_v5"),
    ("azure", "Standard_EC48ads_v5"),
    ("azure", "Standard_EC48as_v5"),
    ("azure", "Standard_EC48eds_v5"),
    ("azure", "Standard_EC48es_v5"),
    ("azure", "Standard_EC4ads_v5"),
    ("azure", "Standard_EC4as_v5"),
    ("azure", "Standard_EC4eds_v5"),
    ("azure", "Standard_EC4es_v5"),
    ("azure", "Standard_EC64ads_v5"),
    ("azure", "Standard_EC64as_v5"),
    ("azure", "Standard_EC64eds_v5"),
    ("azure", "Standard_EC64es_v5"),
    ("azure", "Standard_EC8ads_v5"),
    ("azure", "Standard_EC8as_v5"),
    ("azure", "Standard_EC8eds_v5"),
    ("azure", "Standard_EC8es_v5"),
    ("azure", "Standard_EC96ads_cc_v5"),
    ("azure", "Standard_EC96ads_v5"),
    ("azure", "Standard_EC96as_cc_v5"),
    ("azure", "Standard_EC96as_v5"),
    ("azure", "Standard_EC96iads_v5"),
    ("azure", "Standard_EC96ias_v5"),
    ("azure", "Standard_FX12mds"),
    ("azure", "Standard_FX24mds"),
    ("azure", "Standard_FX36mds"),
    ("azure", "Standard_FX48mds"),
    ("azure", "Standard_FX4mds"),
    ("azure", "Standard_H16"),
    ("azure", "Standard_H16m"),
    ("azure", "Standard_H16mr"),
    ("azure", "Standard_H16r"),
    ("azure", "Standard_H8"),
    ("azure", "Standard_H8m"),
    ("azure", "Standard_HB120-16rs_v2"),
    ("azure", "Standard_HB120-16rs_v3"),
    ("azure", "Standard_HB120-32rs_v2"),
    ("azure", "Standard_HB120-32rs_v3"),
    ("azure", "Standard_HB120-64rs_v2"),
    ("azure", "Standard_HB120-64rs_v3"),
    ("azure", "Standard_HB120-96rs_v2"),
    ("azure", "Standard_HB120-96rs_v3"),
    ("azure", "Standard_HB120rs_v2"),
    ("azure", "Standard_HB120rs_v3"),
    ("azure", "Standard_HB176-144rs_v4"),
    ("azure", "Standard_HB176-24rs_v4"),
    ("azure", "Standard_HB176-48rs_v4"),
    ("azure", "Standard_HB176-96rs_v4"),
    ("azure", "Standard_HB176rs_v4"),
    ("azure", "Standard_HB60-15rs"),
    ("azure", "Standard_HB60-30rs"),
    ("azure", "Standard_HB60-45rs"),
    ("azure", "Standard_HB60rs"),
    ("azure", "Standard_HX176-144rs"),
    ("azure", "Standard_HX176-24rs"),
    ("azure", "Standard_HX176-48rs"),
    ("azure", "Standard_HX176-96rs"),
    ("azure", "Standard_HX176rs"),
    ("azure", "Standard_L80as_v3"),
    ("azure", "Standard_L80s_v3"),
    ("azure", "Standard_M128dms_v2"),
    ("azure", "Standard_M128ds_v2"),
    ("azure", "Standard_M128ms_v2"),
    ("azure", "Standard_M128s_v2"),
    ("azure", "Standard_M12ds_v3"),
    ("azure", "Standard_M12s_v3"),
    ("azure", "Standard_M176ds_3_v3"),
    ("azure", "Standard_M176ds_4_v3"),
    ("azure", "Standard_M176s_3_v3"),
    ("azure", "Standard_M176s_4_v3"),
    ("azure", "Standard_M192idms_v2"),
    ("azure", "Standard_M192ids_v2"),
    ("azure", "Standard_M192ims_v2"),
    ("azure", "Standard_M192is_v2"),
    ("azure", "Standard_M24ds_v3"),
    ("azure", "Standard_M24s_v3"),
    ("azure", "Standard_M32dms_v2"),
    ("azure", "Standard_M32ms_v2"),
    ("azure", "Standard_M416ds_6_v3"),
    ("azure", "Standard_M416ds_8_v3"),
    ("azure", "Standard_M416s_6_v3"),
    ("azure", "Standard_M416s_8_v3"),
    ("azure", "Standard_M48ds_1_v3"),
    ("azure", "Standard_M48s_1_v3"),
    ("azure", "Standard_M624ds_12_v3"),
    ("azure", "Standard_M624s_12_v3"),
    ("azure", "Standard_M64dms_v2"),
    ("azure", "Standard_M64ds_v2"),
    ("azure", "Standard_M64ms_v2"),
    ("azure", "Standard_M64s_v2"),
    ("azure", "Standard_M832ds_12_v3"),
    ("azure", "Standard_M832ids_16_v3"),
    ("azure", "Standard_M832is_16_v3"),
    ("azure", "Standard_M832s_12_v3"),
    ("azure", "Standard_M896ixds_32_v3"),
    ("azure", "Standard_M96ds_1_v3"),
    ("azure", "Standard_M96ds_2_v3"),
    ("azure", "Standard_M96s_1_v3"),
    ("azure", "Standard_M96s_2_v3"),
    ("azure", "Standard_NC12"),
    ("azure", "Standard_NC12s_v2"),
    ("azure", "Standard_NC12s_v3"),
    ("azure", "Standard_NC16ads_A10_v4"),
    ("azure", "Standard_NC16as_T4_v3"),
    ("azure", "Standard_NC24"),
    ("azure", "Standard_NC24r"),
    ("azure", "Standard_NC24rs_v2"),
    ("azure", "Standard_NC24rs_v3"),
    ("azure", "Standard_NC24s_v2"),
    ("azure", "Standard_NC24s_v3"),
    ("azure", "Standard_NC32ads_A10_v4"),
    ("azure", "Standard_NC4as_T4_v3"),
    ("azure", "Standard_NC6"),
    ("azure", "Standard_NC64as_T4_v3"),
    ("azure", "Standard_NC6s_v2"),
    ("azure", "Standard_NC6s_v3"),
    ("azure", "Standard_NC80adis_H100_v5"),
    ("azure", "Standard_NC8ads_A10_v4"),
    ("azure", "Standard_NC8as_T4_v3"),
    ("azure", "Standard_NCC40ads_H100_v5"),
    ("azure", "Standard_ND12s"),
    ("azure", "Standard_ND24rs"),
    ("azure", "Standard_ND24s"),
    ("azure", "Standard_ND40rs_v2"),
    ("azure", "Standard_ND6s"),
    ("azure", "Standard_ND96asr_v4"),
    ("azure", "Standard_ND96is_H100_v5"),
    ("azure", "Standard_ND96is_MI300X_v5"),
    ("azure", "Standard_ND96isr_H100_v5"),
    ("azure", "Standard_ND96isr_H200_v5"),
    ("azure", "Standard_NG16ads_V620_v1"),
    ("azure", "Standard_NG32ads_V620_v1"),
    ("azure", "Standard_NG8ads_V620_v1"),
    ("azure", "Standard_NP10s"),
    ("azure", "Standard_NP20s"),
    ("azure", "Standard_NP40s"),
    ("azure", "Standard_NV12"),
    ("azure", "Standard_NV12ads_A10_v5"),
    ("azure", "Standard_NV12ads_V710_v5"),
    ("azure", "Standard_NV12s_v2"),
    ("azure", "Standard_NV12s_v3"),
    ("azure", "Standard_NV16as_v4"),
    ("azure", "Standard_NV18ads_A10_v5"),
    ("azure", "Standard_NV24"),
    ("azure", "Standard_NV24ads_V710_v5"),
    ("azure", "Standard_NV24s_v2"),
    ("azure", "Standard_NV24s_v3"),
    ("azure", "Standard_NV32as_v4"),
    ("azure", "Standard_NV36adms_A10_v5"),
    ("azure", "Standard_NV36ads_A10_v5"),
    ("azure", "Standard_NV48s_v3"),
    ("azure", "Standard_NV4ads_V710_v5"),
    ("azure", "Standard_NV4as_v4"),
    ("azure", "Standard_NV6"),
    ("azure", "Standard_NV6ads_A10_v5"),
    ("azure", "Standard_NV6s_v2"),
    ("azure", "Standard_NV72ads_A10_v5"),
    ("azure", "Standard_NV8ads_V710_v5"),
    ("azure", "Standard_NV8as_v4"),
    ("azure", "Standard_PB6s"),
]


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
    import traceback

    threading.current_thread().name = "main"

    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1024)
    count = 0
    lock = threading.Lock()
    exception = None
    for (vendor, server), (srv_data, regions, zones) in available_servers().items():
        alicloud_servers = {
            "ecs.t5-lc1m1.small",
            "ecs.c6a.2xlarge",
            "ecs.c6r.large",
            "ecs.c6r.xlarge",
            "ecs.c7a.large",
        }
        if vendor == "alicloud" and server not in alicloud_servers:
            logging.info(f"Excluding {vendor}/{server}")
            continue
        if vendor not in supported_vendors:
            # sc-runner can't yet handle this vendor
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
        try:
            tasks = list(filter(lambda task: lib.should_start(task, data_dir, srv_data), lib.get_tasks(vendor)))
        except Exception as e:
            # stop if an exception occurred
            exception = e
            logging.exception(f"{vendor}/{server} failed")
            break
        if not tasks:
            logging.info(f"No tasks for {vendor}/{server}")
            continue
        f = executor.submit(lib.start_inspect, executor, lock, data_dir, vendor, server, tasks, srv_data, regions, zones)
        futures[f] = (vendor, server)
        count += 1
        # number of servers to start at a time: best to leave this at 1 to avoid quota issues,
        # but can be increased temporarily if needed to run a new benchmark on all servers faster
        # (although you will have to delete the failed tasks' meta.json and retry with count=1)
        if count == 1:
            break
    for f in concurrent.futures.as_completed(futures):
        vendor, server = futures[f]
        try:
            result = f.result()
        except Exception:
            logging.exception(f"Inspection for {vendor}/{server} raised an exception")
    logging.info("Start completed")
    # python will wait for all executors to finish
    executor.shutdown(wait=False)
    logging.info("Waiting for executors to finish")
    lib.thread_monitor(executor)
    repo.push_path(os.path.join(ctx.parent.params["repo_path"], "data"), f"Start finished {repo.gha_url()}")
    logging.info("Git push successful")

    # Print all active non-daemon threads and their stack traces
    non_daemon_threads = []
    for thread in threading.enumerate():
        if thread == threading.current_thread() or thread.daemon:
            # skip main thread and daemon threads
            continue
        logging.info(f"Thread {thread.name} (daemon={thread.daemon})")
        stack = traceback.format_stack(sys._current_frames()[thread.ident])
        for line in stack:
            logging.info(f"  {line.strip()}")
        non_daemon_threads.append(thread)

    if non_daemon_threads:
        # possibly running into this: https://github.com/pulumi/pulumi/issues/16095, do a forceful exit
        # without waiting for non-daemon threads to finish
        logging.info(f"Force exiting due to {len(non_daemon_threads)} non-daemon threads still running")
        if exception:
            os._exit(1)
        else:
            os._exit(0)

    if exception:
        # fail if an exception was raised in should_start
        raise exception


def cleanup_task(vendor, server, data_dir, regions=[], zones=[], force=False):
    """
    Some vendors support creating resources in regions, without explicitly specifying the zone, some don't,
    so we support both of them. We'll go through all regions or zones whatever is specified.
    """
    from datetime import datetime, timedelta
    from sc_runner import runner
    import threading

    tasks = list(lib.get_tasks(vendor))
    if not tasks:
        return

    # set thread name for logging
    threading.current_thread().name = f"{vendor}/{server}"

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
        # only consider start times of tasks that are not finished yet
        if not meta.end:
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
                runner.destroy_stack(vendor, pulumi_opts, resource_opts | dict(instance=server), stack_opts=dict(on_output=logging.info))


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=64, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
@click.option("--force/--no-force", type=bool, default=False, help="Do a cleanup even if there's no meta for the server")
@click.option("--all-regions/--no-all-regions", type=bool, default=False, help="Clean up in all regions, not just in those which list the server as available")
@click.option("--lookback-mins", type=int, show_default=True, help="Only clean up those instances that started at most this many minutes ago")
@click.option("--vendor", type=str, default=None, help="Only clean up resources for the specified vendor")
def cleanup(ctx, threads, force, all_regions, lookback_mins, vendor):
    from sc_runner.resources import supported_vendors
    from datetime import datetime, timedelta
    import concurrent.futures

    max_start = None
    if lookback_mins:
        max_start = datetime.now() - timedelta(minutes=lookback_mins)
    futures = []
    servers = lib.sort_available_servers(
        available_servers(vendor=vendor),
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

        error_occurred = False
        for vendor, server, f in futures:
            try:
                f.result()
            except Exception:
                logging.exception(f"Error in processing {vendor}/{server}")
                error_occurred = True

        if error_occurred:
            raise Exception("Errors occurred during cleanup")


@cli.command()
@click.pass_context
def parse(ctx):
    """Parse already written outputs from the repo and write them back."""
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
        if os.path.exists(data_dir):
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

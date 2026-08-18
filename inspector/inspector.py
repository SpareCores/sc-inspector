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


class ContextAwareFormatter(logging.Formatter):
    """Prefer logger name; fallback to thread name."""

    def format(self, record):
        source = record.name if record.name != "root" else record.threadName
        record.log_source = source
        return super().format(record)


# Create a formatter string with color codes
formatter_str = (
    f"{YELLOW}%(asctime)s{RESET}/"
    f"{GREEN}%(levelname)s{RESET}/{RED}%(log_source)s{RESET}: "
    "%(message)s"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=formatter_str,
    datefmt="%Y-%m-%d %H:%M:%S"
)
for handler in logging.getLogger().handlers:
    handler.setFormatter(ContextAwareFormatter(formatter_str, "%Y-%m-%d %H:%M:%S"))
lib.logging = logging
EXCLUDE_INSTANCES: list[tuple[str, str]] = [
    # too small memory instances
    ("aws", "t3.nano"),
    ("gcp", "f1-micro"),
    # we can't (yet) start these
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
    return [region.api_reference for region in session.exec(stmt.distinct()).all()]


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
            .join(Server)
            .where(Server.status == "ACTIVE")
            .order_by(ServerPrice.price)
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
            # Store zone-to-region mapping
            servers[(vendor, server.api_reference)][3][zone] = region
        else:
            servers[(vendor, server.api_reference)] = [server, [region], [zone], {zone: region}]
    return servers


@click.group()
@click.option("--repo-path", default=os.environ.get("REPO_PATH", os.getcwd()), help="Directory which contains the repository")
def cli(repo_path):
    pass


@cli.command()
@click.pass_context
@click.option("--exclude", type=(str, str), default=EXCLUDE_INSTANCES, multiple=True, help="Exclude $vendor $instance")
@click.option("--start-only", type=(str, str), multiple=True, help="Start only $vendor $instance")
@click.option("--vendor", type=str, default=None, help="Only start instances for the specified vendor")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max hosts to start this round (default: START_LIMIT env or 8)",
)
def start(ctx, exclude, start_only, vendor, limit):
    from sc_runner.resources import supported_vendors
    import concurrent.futures
    import threading
    import traceback

    threading.current_thread().name = "main"

    if limit is None:
        limit = int(os.environ.get("START_LIMIT", "8"))
    if limit < 1:
        raise click.ClickException("--limit must be >= 1")

    futures = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1024)
    count = 0
    lock = threading.Lock()
    exception = None
    # Hash order mixes instance sizes; price order would exhaust quota on small SKUs first.
    for (vnd, server), (srv_data, regions, zones, zone_to_region) in lib.sort_servers_for_start(
        available_servers(vendor=vendor)
    ):
        if vnd == "aws" and not (server.startswith("m9g") or server == "c4.xlarge"):
            logging.info(f"Excluding {vnd}/{server}")
            continue
        if vnd not in supported_vendors:
            # sc-runner can't yet handle this vendor
            continue
        gpu_count = srv_data.gpu_count
        logging.info(f"Evaluating {vnd}/{server} with {gpu_count} GPUs")
        if (vnd, server) in exclude:
            logging.info(f"Excluding {vnd}/{server}")
            continue
        if start_only and (vnd, server) not in start_only:
            logging.info(f"Excluding {vnd}/{server} as --start-only {start_only} is given")
            continue
        data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vnd, server)
        try:
            tasks = lib.tasks_to_start(vnd, data_dir, srv_data)
        except Exception as e:
            # record failure but keep evaluating other servers; raised at end of run
            if exception is None:
                exception = e
            logging.exception(f"{vnd}/{server} failed")
            continue
        if not tasks:
            logging.info(f"No tasks for {vnd}/{server}")
            continue
        logging.info(
            "Starting host %s/%s (%d/%d this round)",
            vnd,
            server,
            count + 1,
            limit,
        )
        f = executor.submit(lib.start_inspect, executor, lock, data_dir, vnd, server, tasks, srv_data, regions, zones, zone_to_region)
        futures[f] = (vnd, server)
        count += 1
        if start_only or count >= limit:
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


@cli.command("start-dbaas")
@click.pass_context
@click.option("--vendor", type=str, default="gcp", help="Only start DBaaS targets for the specified vendor")
@click.option(
    "--instance-key",
    type=str,
    default=None,
    help="Only start this managed DB instance key (e.g. db-perf-optimized-N-8/postgres/18/standalone)",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max DBaaS stacks to start this round (default: DBAAS_START_LIMIT or 4)",
)
def start_dbaas(ctx, vendor, instance_key, limit):
    """Start managed Postgres benchmark stacks, advancing through the catalog over rounds."""
    from dbaas_catalog import available_managed_dbs
    from dbaas_selector import dbaas_data_dir
    from dbaas_start import start_dbaas_inspect
    import concurrent.futures
    import threading

    if limit is None:
        limit = int(os.environ.get("DBAAS_START_LIMIT", "4"))
    if limit < 1:
        raise click.ClickException("--limit must be >= 1")

    threading.current_thread().name = "main"
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(8, limit))
    lock = threading.Lock()
    futures = {}
    count = 0
    for (vnd, key), (target, regions, zones, zone_to_region) in lib.sort_managed_dbs_for_start(
        available_managed_dbs(vendor=vendor)
    ):
        if instance_key and key != instance_key:
            continue
        data_dir = dbaas_data_dir(ctx.parent.params["repo_path"], vnd, key)
        tasks = lib.tasks_to_start_dbaas(vnd, data_dir, target)
        if not tasks:
            logging.info("No DBaaS tasks for %s/%s", vnd, key)
            continue
        logging.info("Starting DBaaS stack %s/%s (%d/%d this round)", vnd, key, count + 1, limit)
        futures[
            executor.submit(
                start_dbaas_inspect,
                executor,
                lock,
                data_dir,
                vnd,
                target,
                tasks,
                regions,
                zones,
                zone_to_region,
            )
        ] = (vnd, key)
        count += 1
        if instance_key or count >= limit:
            break
    if instance_key and count == 0:
        logging.warning("No startable DBaaS tasks for --instance-key %s", instance_key)
    error_occurred = False
    for f in concurrent.futures.as_completed(futures):
        vnd, key = futures[f]
        try:
            f.result()
        except Exception:
            logging.exception("DBaaS start failed for %s/%s", vnd, key)
            error_occurred = True
    executor.shutdown(wait=False)
    lib.thread_monitor(executor)
    repo.push_path(
        os.path.join(ctx.parent.params["repo_path"], "dbaas"),
        f"DBaaS start finished {repo.gha_url()}",
    )
    if error_occurred:
        raise RuntimeError("DBaaS start failed for one or more targets")


@cli.command("cleanup-dbaas")
@click.pass_context
@click.option("--threads", type=int, default=32, show_default=True)
@click.option("--vendor", type=str, default=None, help="Vendor to clean up DBaaS stacks for")
def cleanup_dbaas(ctx, threads, vendor):
    """S3 run-record driven cleanup for DBaaS stacks."""
    from sc_runner.resources import supported_vendors

    repo_path = ctx.parent.params["repo_path"]
    catalog = _dbaas_catalog_for_cleanup(vendor, os.path.join(repo_path, "dbaas"))
    catalog = {k: v for k, v in catalog.items() if k[0] in supported_vendors}
    records = _dbaas_run_records_for_catalog(catalog)
    if not records:
        logging.info("No completed DBaaS run records for catalog%s", f" vendor={vendor}" if vendor else "")
        return
    try:
        _run_dbaas_s3_cleanup(repo_path, records, threads=threads)
    except Exception:
        logging.exception("DBaaS cleanup failed")
        raise


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
    max_unfinished_timeout = timedelta()
    now = datetime.now()
    applicable_tasks = [
        task for task in tasks
        if not (task.servers_only and (vendor, server) not in task.servers_only)
        and not (task.servers_exclude and (vendor, server) in task.servers_exclude)
    ]
    task_metas = [(task, lib.load_task_meta(task, data_dir=data_dir)) for task in applicable_tasks]
    last_activity = max((meta.end for _, meta in task_metas if meta.end), default=None)
    for task, meta in task_metas:
        if not meta.start:
            continue
        if meta.end:
            end_times.append(meta.end)
            already_ended.append(True)
            continue
        stale = now > meta.start + task.timeout + lib.DESTROY_AFTER
        abandoned = lib.is_abandoned_boot_meta(meta, last_activity)
        if stale or abandoned:
            already_ended.append(True)
            logging.info(
                f"{vendor}/{server} Treating {task.name} as finished (abandoned={abandoned}, stale={stale})"
            )
            continue
        start_times.append(meta.start)
        already_ended.append(False)
        logging.info(f"{vendor}/{server} Adding task {task.name} timeout: {task.timeout}")
        sum_timeout += task.timeout
        max_unfinished_timeout = max(max_unfinished_timeout, task.timeout)

    if start_times and now >= (wait_time := max(start_times) + max_unfinished_timeout + lib.DESTROY_AFTER):
        # safety net: after the longest unfinished task timeout has passed, terminate the machine
        destroy = f"Destroying {vendor}/{server}, last_start: {max(start_times)}, last timeout: {wait_time}"
    if start_times and now >= (wait_time := max(start_times) + sum_timeout + lib.DESTROY_AFTER):
        # We can only estimate the time by which all tasks should have been completed, as the start date is added
        # to the git repository before the machine starts up, the machine startup can take a long time, and the
        # tasks do not necessarily run sequentially.
        # So here, we are using the sum_timeout, which is the sum of all timeouts for unfinished jobs.
        destroy = f"Destroying {vendor}/{server}, last_start: {max(start_times)}, wait time: {wait_time}"

    # if all tasks have already finished, we can destroy the stack
    if already_ended and all(already_ended):
        destroy = f"Destroying {vendor}/{server}, all tasks have finished"

    if not destroy and not start_times and end_times and now >= max(end_times) + lib.DESTROY_AFTER:
        destroy = f"Destroying {vendor}/{server}, last end: {max(end_times)}"

    if not destroy and not start_times and force:
        # forced cleanup, even if there are metas for the server (might be due to a forced retry from git, by deleting
        # the files)
        destroy = f"Forced cleanup of {vendor}/{server}"

    if destroy:
        # Catalog/plan regions can lag behind where start actually created stacks
        # (failed retries, ACTIVE-price churn). Always union locations discovered
        # from the Pulumi S3 backend so orphans in other regions/zones are hit.
        import pulumi_backend

        backend_regions, backend_zones = pulumi_backend.locations_for_instance(vendor, server)
        regions = list(dict.fromkeys([*(regions or []), *backend_regions]))
        zones = list(dict.fromkeys([*(zones or []), *backend_zones]))
        if not regions and not zones:
            logging.warning(
                "%s/%s due for destroy but no catalog or Pulumi backend locations to target",
                vendor,
                server,
            )
        resource_opts = {}
        destroy_errors = []
        destroy_attempted = []
        with tempfile.TemporaryDirectory() as tempdir:
            pulumi_opts = dict(work_dir=tempdir)
            # use either regions or zones for cleaning up the stacks
            for opt_name, value in itertools.chain(
                zip(["region"] * len(regions), regions),
                zip(["zone"] * len(zones), zones),
            ):
                resource_opts[opt_name] = value
                try:
                    # if forced, we do a full refresh and delete for all resources
                    if not force:
                        # In order not to cause unnecessary locks in Pulumi, we first get the stack's resources to see if
                        # it's already empty, and in that case, we don't destroy it.
                        try:
                            stack = runner.get_stack(
                                vendor, pulumi_opts, resource_opts | dict(instance=server)
                            )
                        except AttributeError:
                            logging.exception("Couldn't get stack")
                            # this vendor is not yet supported
                            return
                        resources = stack.export_stack().deployment.get("resources", [])
                        if len(resources) <= 1:
                            # a non-existent stack will have zero, a clean (already destroyed) stack should have exactly one
                            # resource (the Pulumi Stack itself). If we can see either of these, we have nothing to clean up.
                            logging.info(
                                f"Pulumi stack for {vendor}/{value}/{server} has {len(resources)} resources, no cleanup needed"
                            )
                            continue
                    logging.info(destroy)
                    destroy_attempted.append(value)
                    runner.destroy_stack(
                        vendor,
                        pulumi_opts,
                        resource_opts | dict(instance=server),
                        stack_opts=dict(on_output=lib.pulumi_log_output),
                    )
                except Exception:
                    logging.exception(f"Failed to destroy {vendor}/{value}/{server}")
                    destroy_errors.append(value)
        if destroy_attempted and len(destroy_errors) == len(destroy_attempted):
            raise Exception(
                f"Failed to destroy {vendor}/{server} in all {len(destroy_errors)} attempted location(s): "
                + ", ".join(destroy_errors)
            )
        if destroy_errors:
            logging.warning(
                f"{vendor}/{server}: destroy failed in {destroy_errors} but succeeded in other location(s)"
            )


def stack_key_for_record(record) -> tuple:
    if record.vendor == "gcp":
        return (record.vendor, record.zone, record.instance)
    return (record.vendor, record.region, record.instance)


def _resource_opts_for_record(vendor: str, record) -> dict:
    resource_opts = {"instance": record.instance}
    if getattr(record, "topology", "") == "dbaas" and getattr(record, "dbaas_slug", ""):
        resource_opts["dbaas_slug"] = record.dbaas_slug
    if vendor == "gcp":
        if not record.zone:
            raise RuntimeError(f"missing zone for GCP cleanup ({record.key})")
        resource_opts["zone"] = record.zone
    else:
        if not record.region:
            raise RuntimeError(f"missing region for cleanup ({record.key})")
        resource_opts["region"] = record.region
    return resource_opts


def _dbaas_catalog_for_cleanup(
    vendor: str | None,
    repo_dbaas: str,
    *,
    lookback_mins: int | None = None,
    data_only: bool = False,
    require_scan: bool = False,
) -> dict:
    from dbaas_catalog import available_managed_dbs

    catalog = available_managed_dbs(vendor=vendor)
    if not require_scan:
        return catalog
    return {
        k: v
        for k, v in catalog.items()
        if lib.should_scan_dbaas_for_cleanup(
            repo_dbaas, k[0], k[1], lookback_mins=lookback_mins, data_only=data_only
        )
    }


def _dbaas_run_records_for_catalog(catalog: dict) -> list:
    import s3_runs

    if not catalog:
        return []
    catalog_keys = set(catalog)
    records = []
    for vnd in {k[0] for k in catalog_keys}:
        for record in s3_runs.list_completed_runs(vendor=vnd):
            if record.topology != "dbaas":
                continue
            if (record.vendor, record.instance_key) not in catalog_keys:
                continue
            records.append(record)
    return records


def _run_dbaas_s3_cleanup(repo_path: str, records: list, *, threads: int) -> None:
    import concurrent.futures

    if not records:
        return
    groups: dict[tuple, list] = {}
    for record in records:
        groups.setdefault(stack_key_for_record(record), []).append(record)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for _key, group_records in groups.items():
            record = group_records[0]
            data_dir = None
            if record.instance_key:
                data_dir = os.path.join(
                    repo_path, "dbaas", record.vendor, record.instance_key
                )
            futures.append(
                executor.submit(
                    cleanup_s3_stack,
                    record.vendor,
                    group_records,
                    data_dir=data_dir,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()


def cleanup_s3_stack(vendor: str, records: list, *, data_dir: str | None = None) -> None:
    """Clean up one Pulumi stack; delete S3 markers only after the instance is gone."""
    from sc_runner import runner
    import s3_runs
    import tempfile

    record = records[0]
    keys = [r.key for r in records]
    location = record.region or record.zone
    resource_opts = _resource_opts_for_record(vendor, record)

    if data_dir:
        dbaas_key = (
            record.instance_key
            if getattr(record, "topology", "") == "dbaas" and record.instance_key
            else None
        )
        block_reason = lib.active_run_blocks_s3_cleanup(
            vendor,
            record.instance,
            records,
            data_dir,
            instance_key=dbaas_key,
        )
        if block_reason:
            logging.warning(
                "Skipping destroy for %s/%s in %s; %s; retaining %d run record(s)",
                vendor,
                record.instance,
                location,
                block_reason,
                len(keys),
            )
            return

    with tempfile.TemporaryDirectory() as tempdir:
        pulumi_opts = dict(work_dir=tempdir)

        def cancel_lock() -> None:
            runner.cancel(vendor, pulumi_opts, resource_opts)

        try:
            stack = lib.retry_locked_cleanup(
                lambda: runner.get_stack(vendor, pulumi_opts, resource_opts),
                cancel_func=cancel_lock,
            )
        except lib.StackLockedError:
            logging.warning(
                "Stack %s/%s in %s is locked; retaining %d run record(s) until cleanup succeeds",
                vendor,
                record.instance,
                location,
                len(keys),
            )
            return
        except AttributeError:
            raise RuntimeError(f"vendor {vendor} not supported for {record.key}") from None

        resources = stack.export_stack().deployment.get("resources", [])
        if len(resources) <= 1:
            logging.info(
                "Pulumi stack for %s/%s in %s already clean (%d resources); removing %d run record(s)",
                vendor,
                record.instance,
                location,
                len(resources),
                len(keys),
            )
            for key in keys:
                s3_runs.delete_run_record(key)
            return

        logging.info(
            "Destroying %s/%s in %s for %d run record(s) (e.g. %s)",
            vendor,
            record.instance,
            location,
            len(keys),
            record.key,
        )
        try:
            lib.retry_locked_cleanup(
                lambda: runner.destroy_stack(
                    vendor,
                    pulumi_opts,
                    resource_opts,
                    stack_opts=dict(on_output=lib.pulumi_log_output),
                ),
                cancel_func=cancel_lock,
            )
        except lib.StackLockedError:
            logging.warning(
                "Destroy for %s/%s in %s is locked; retaining %d run record(s) until cleanup succeeds",
                vendor,
                record.instance,
                location,
                len(keys),
            )
            return

    for key in keys:
        s3_runs.delete_run_record(key)


@cli.command("cleanup-sweep")
@click.pass_context
@click.option("--threads", type=int, default=64, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
@click.option("--force/--no-force", type=bool, default=False, help="Do a cleanup even if there's no meta for the server")
@click.option("--all-regions/--no-all-regions", type=bool, default=False,
              help="Clean up in all regions, not just in those which list the server as available")
@click.option(
    "--lookback-mins",
    type=int,
    default=None,
    help="Fixed lookback in minutes (daily safety net). Default: task-timeout-based window.",
)
@click.option(
    "--data-only/--no-data-only",
    default=False,
    help="Only consider servers with a data directory in the repo (skips the full catalog).",
)
@click.option("--vendor", type=str, default=None, help="Only clean up resources for the specified vendor")
def _run_meta_cleanup_for_vendor(
    repo_path: str,
    vendor: str | None,
    *,
    threads: int,
    lookback_mins: int | None = None,
    data_only: bool = True,
    force: bool = False,
    all_regions: bool = False,
) -> None:
    """Meta-driven destroy using catalog + Pulumi-backend locations (shared by cleanup jobs)."""
    from sc_runner.resources import supported_vendors
    import concurrent.futures
    import pulumi_backend

    repo_data = os.path.join(repo_path, "data")
    candidates = available_servers(vendor=vendor)
    # Include instances that still have Pulumi stacks even if they dropped out of
    # the ACTIVE price catalog (otherwise orphans in retired regions are ignored).
    for vnd in ([vendor] if vendor else sorted(supported_vendors)):
        if vnd not in supported_vendors:
            continue
        for instance in pulumi_backend.instances_with_stacks(vnd):
            key = (vnd, instance)
            if key in candidates:
                continue
            if force or lib.should_scan_for_cleanup(
                repo_data, vnd, instance, lookback_mins=lookback_mins, data_only=data_only
            ):
                candidates[key] = [None, [], [], {}]
                logging.info(
                    "Adding %s/%s to meta cleanup from Pulumi backend stacks",
                    vnd,
                    instance,
                )
    if not force:
        candidates = {
            k: v
            for k, v in candidates.items()
            if k[0] in supported_vendors
            and lib.should_scan_for_cleanup(
                repo_data, k[0], k[1], lookback_mins=lookback_mins, data_only=data_only
            )
        }
    else:
        candidates = {k: v for k, v in candidates.items() if k[0] in supported_vendors}
    if not candidates:
        logging.info(
            "Meta cleanup: no servers to scan%s",
            f" for {vendor}" if vendor else "",
        )
        return

    futures = []
    servers = lib.sort_available_servers(candidates, data_dir=repo_data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for (vnd, server), (_, regions, zones, _zone_to_region) in servers:
            if all_regions:
                regions = get_regions(vnd)
            elif vnd == "vultr":
                regions = lib.vultr_cleanup_regions(server, list(regions))
            # Always merge locations from Pulumi checkpoints so stacks left in
            # regions no longer in the catalog are still destroyed.
            backend_regions, backend_zones = pulumi_backend.locations_for_instance(vnd, server)
            regions = list(dict.fromkeys([*list(regions or []), *backend_regions]))
            zones = list(dict.fromkeys([*list(zones or []), *backend_zones]))
            data_dir = os.path.join(repo_path, "data", vnd, server)
            if vnd in {"gcp"}:
                futures.append(
                    (vnd, server, executor.submit(
                        cleanup_task, vnd, server, data_dir, zones=zones, force=force
                    ))
                )
            else:
                futures.append(
                    (vnd, server, executor.submit(
                        cleanup_task, vnd, server, data_dir, regions=regions, force=force
                    ))
                )
        failures = False
        for vnd, server, future in futures:
            try:
                future.result()
            except Exception:
                logging.exception("Meta cleanup failed for %s/%s", vnd, server)
                failures = True
        if failures:
            raise Exception("Errors occurred during meta cleanup")


@cli.command("cleanup-sweep")
@click.pass_context
@click.option("--threads", type=int, default=64, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
@click.option("--force/--no-force", type=bool, default=False, help="Do a cleanup even if there's no meta for the server")
@click.option("--all-regions/--no-all-regions", type=bool, default=False,
              help="Clean up in all regions, not just in those which list the server as available")
@click.option(
    "--lookback-mins",
    type=int,
    default=None,
    help="Fixed lookback in minutes (daily safety net). Default: task-timeout-based window.",
)
@click.option(
    "--data-only/--no-data-only",
    default=False,
    help="Only consider servers with a data directory in the repo (skips the full catalog).",
)
@click.option("--vendor", type=str, default=None, help="Only clean up resources for the specified vendor")
def cleanup_sweep(ctx, threads, force, all_regions, lookback_mins, data_only, vendor):
    """Catalog/meta-driven cleanup sweep (daily safety net)."""
    from sc_runner.resources import supported_vendors

    repo_path = ctx.parent.params["repo_path"]
    _run_meta_cleanup_for_vendor(
        repo_path,
        vendor,
        threads=threads,
        lookback_mins=lookback_mins,
        data_only=data_only,
        force=force,
        all_regions=all_regions,
    )

    repo_dbaas = os.path.join(repo_path, "dbaas")
    dbaas_catalog = _dbaas_catalog_for_cleanup(
        vendor,
        repo_dbaas,
        lookback_mins=lookback_mins,
        data_only=data_only,
        require_scan=True,
    )
    dbaas_catalog = {k: v for k, v in dbaas_catalog.items() if k[0] in supported_vendors}
    try:
        _run_dbaas_s3_cleanup(
            repo_path,
            _dbaas_run_records_for_catalog(dbaas_catalog),
            threads=threads,
        )
    except Exception:
        logging.exception("DBaaS cleanup sweep failed")
        raise


@cli.command()
@click.pass_context
@click.option("--threads", type=int, default=64, show_default=True,
              help="Number of threads to run Pulumi concurrently. Each thread consumes around 60 MiB of RAM.")
@click.option("--vendor", type=str, default=None, help="Only clean up resources for the specified vendor")
def cleanup(ctx, threads, vendor):
    """S3 run-record driven cleanup (frequent scheduled job)."""
    from collections import defaultdict
    from sc_runner.resources import supported_vendors
    import concurrent.futures
    import s3_runs

    repo.get_repo()
    repo.pull()
    data_dir = os.path.join(ctx.parent.params["repo_path"], "data")
    repo_path = ctx.parent.params["repo_path"]

    records = s3_runs.list_completed_runs(vendor=vendor)
    # DBaaS finished markers are owned by cleanup-dbaas / cleanup-sweep; this
    # job only sees multi-VM records (avoids destroying with data/ metas).
    records = [r for r in records if r.topology != "dbaas"]
    logging.info(
        "Found %d completed S3 run record(s) to clean up%s",
        len(records),
        f" for {vendor}" if vendor else "",
    )

    stacks: dict[tuple, list] = defaultdict(list)
    for record in records:
        if record.vendor not in supported_vendors:
            logging.warning("Skipping unsupported vendor %s in %s", record.vendor, record.key)
            continue
        stacks[stack_key_for_record(record)].append(record)

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for group_records in stacks.values():
            stack_vendor = group_records[0].vendor
            futures[executor.submit(
                cleanup_s3_stack, stack_vendor, group_records, data_dir=data_dir
            )] = group_records

        failures: list[str] = []
        for future, group_records in futures.items():
            try:
                future.result()
            except Exception:
                keys = ", ".join(r.key for r in group_records)
                logging.exception("S3-driven cleanup failed for stack record(s): %s", keys)
                failures.extend(r.key for r in group_records)

        if failures:
            raise Exception(
                f"S3 cleanup failed for {len(failures)} run(s): {', '.join(failures)}"
            )
    logging.info(
        "S3 cleanup finished for %d stack(s) (%d run record(s))%s",
        len(futures),
        len(records),
        f" ({vendor})" if vendor else "",
    )

    # Safety net for hosts that never wrote an S3 terminated marker (hung boot,
    # inspect crash, etc.): use git metas + Pulumi backend stack locations.
    logging.info(
        "Running meta/Pulumi-backend cleanup pass%s",
        f" for {vendor}" if vendor else "",
    )
    _run_meta_cleanup_for_vendor(
        repo_path,
        vendor,
        threads=threads,
        lookback_mins=None,
        data_only=True,
    )


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
@click.option("--gpu-count", default=0.0, type=float, help="Number of GPUs")
@click.option("--threads", default=8, show_default=True, help="Parallelism in a given task group")
@click.option(
    "--dbaas-instance-key",
    default=None,
    help="Managed DB instance key for DBaaS sparse checkout (under dbaas/<vendor>/)",
)
def inspect(ctx, vendor, instance, gpu_count, threads, dbaas_instance_key):
    """Run inspection on this machine."""
    # Disable OOM killer for this task as Linux tends to kill this instead of benchmarks, like bw_mem
    pid = os.getpid()
    try:
        with open(f"/proc/{pid}/oom_score_adj", mode="w") as f:
            f.write("-1000")
    except Exception:
        # If it fails, log but continue (not critical)
        logging.warning("Could not disable OOM killer for inspector process")

    logging.info("Updating the git repo")
    # we must clone the repo before writing anything to it
    repo_path = ctx.parent.params["repo_path"]
    if dbaas_instance_key:
        sparse = (f"dbaas/{vendor}/{dbaas_instance_key}",)
        data_dir = os.path.join(repo_path, "dbaas", vendor, dbaas_instance_key)
    else:
        sparse = (f"data/{vendor}/{instance}",)
        data_dir = os.path.join(repo_path, "data", vendor, instance)
    repo.get_repo(sparse_paths=sparse)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lib.record_timing_inspector_start(data_dir)
    try:
        repo.push_path(lib.timing_dir(data_dir), f"Inspector started from {repo.gha_url()}")
    except Exception:
        logging.exception("Failed to push inspector timing")
    if dbaas_instance_key:
        from postgres_dbaas import wait_db_ready

        wait_db_ready()
    try:
        lib.run_tasks(
            vendor,
            data_dir,
            instance=instance,
            gpu_count=gpu_count,
            nthreads=threads,
        )
    finally:
        lib.finalize_task_metas(
            vendor,
            data_dir,
            instance,
            gpu_count=gpu_count,
        )
        lib.record_timing_inspector_end(data_dir)
        try:
            import s3_runs

            s3_runs.upload_task_logs_to_s3(data_dir)
        except Exception:
            logging.exception("Failed to upload task logs to S3")
        try:
            repo.push_path(lib.timing_dir(data_dir), f"Inspector finished from {repo.gha_url()}")
        except Exception:
            logging.exception("Failed to push inspector timing")


@cli.command()
@click.option("--vendor", required=True, help="Vendor ID for this machine")
@click.option("--instance", required=True, help="Instance ID for this machine")
@click.option("--listen-port", default=18765, show_default=True, type=int, help="MP listen port")
def companion(vendor, instance, listen_port):
    """Run benchmark client daemon for multi-VM Postgres benchmarks."""
    from companion import run_companion

    run_companion(vendor, instance, listen_port)


if __name__ == "__main__":
    cli()

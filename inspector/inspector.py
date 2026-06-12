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

# AliCloud `start` allowlist: derived from sc-crawler schema via ``sc_data.db.path`` (see ``alicloud_inspector_allowlist``).
ALICLOUD_INSPECTOR_EXCLUDED_COUNTRIES = frozenset({"CN", "HK"})
ALICLOUD_INSPECTOR_BUDGET_USD = 1000.0
ALICLOUD_INSPECTOR_HOURS = 3.0  # fallback when repo has no GPU timing samples
# ``sc_crawler`` stores RAM as MiB; require strictly more than 2 GiB.
ALICLOUD_INSPECTOR_MIN_RAM_MIB = 2048
# Prefer this many largest (by total VRAM) unmeasured GPU shapes before filling with cheaper ones.
ALICLOUD_INSPECTOR_BIG_GPU_SLOTS = 4


def _alicloud_gpu_benchmark_measured(repo_path: str, api_reference: str) -> bool:
    """True when nvidia_smi completed successfully in the inspector-data repo."""
    import json

    meta_path = os.path.join(repo_path, "data", "alicloud", api_reference, "nvidia_smi", "meta.json")
    if not os.path.isfile(meta_path):
        return False
    try:
        meta = json.loads(open(meta_path).read())
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("exit_code") == 0


def _alicloud_inspector_run_hours(repo_path: str) -> float:
    """Median wall-clock GPU benchmark duration from inspector-data, with setup overhead."""
    import glob
    import json
    from datetime import datetime

    base = os.path.join(repo_path, "data", "alicloud")
    if not os.path.isdir(base):
        return ALICLOUD_INSPECTOR_HOURS

    hours: list[float] = []
    for inst in os.listdir(base):
        nsmi = os.path.join(base, inst, "nvidia_smi", "meta.json")
        if not os.path.isfile(nsmi):
            continue
        try:
            if json.loads(open(nsmi).read()).get("exit_code") != 0:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        starts: list[datetime] = []
        ends: list[datetime] = []
        for meta_path in glob.glob(os.path.join(base, inst, "*", "meta.json")):
            if "/timing/" in meta_path:
                continue
            try:
                meta = json.loads(open(meta_path).read())
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("start"):
                starts.append(datetime.fromisoformat(meta["start"]))
            if meta.get("end"):
                ends.append(datetime.fromisoformat(meta["end"]))
        if not starts or not ends:
            continue
        elapsed = (max(ends) - min(starts)).total_seconds() / 3600
        if 0.5 < elapsed < 12:
            hours.append(elapsed)

    if not hours:
        return ALICLOUD_INSPECTOR_HOURS
    hours.sort()
    median = hours[len(hours) // 2]
    return max(median * 1.15, 2.0)


@cache
def alicloud_inspector_allowlist(repo_path: str) -> frozenset[str]:
    """Return AliCloud GPU instances we can afford to benchmark within budget.

    Only unmeasured GPU shapes (no successful ``nvidia_smi`` in *repo_path*) with
    ONDEMAND pricing outside ``ALICLOUD_INSPECTOR_EXCLUDED_COUNTRIES`` are candidates.
    Reserves budget for the largest unmeasured shapes (by total VRAM), then fills with
    cheaper unmeasured instances.
    """
    from sqlalchemy import and_, func
    from sc_crawler.tables import Region, Server, ServerPrice
    from sqlmodel import Session, create_engine, select

    import sc_data

    path = sc_data.db.path
    if path is None or not os.path.isfile(path):
        logging.warning(
            "AliCloud inspector allowlist: sc_data DB not available (path=%r) — skipping all AliCloud instances",
            path,
        )
        return frozenset()

    excluded = tuple(ALICLOUD_INSPECTOR_EXCLUDED_COUNTRIES)
    min_price = func.min(ServerPrice.price).label("min_hr")
    stmt = (
        select(
            Server.api_reference,
            Server.gpu_count,
            Server.gpu_memory_total,
            min_price,
        )
        .join(
            ServerPrice,
            and_(
                ServerPrice.vendor_id == Server.vendor_id,
                ServerPrice.server_id == Server.server_id,
            ),
        )
        .join(
            Region,
            and_(
                Region.vendor_id == ServerPrice.vendor_id,
                Region.region_id == ServerPrice.region_id,
            ),
        )
        .where(ServerPrice.vendor_id == "alicloud")
        .where(Server.gpu_count > 0)
        .where(Server.memory_amount > ALICLOUD_INSPECTOR_MIN_RAM_MIB)
        .where(Server.status == "ACTIVE")
        .where(ServerPrice.allocation == "ONDEMAND")
        .where(ServerPrice.status == "ACTIVE")
        .where(Region.country_id.not_in(excluded))
        .group_by(Server.api_reference, Server.gpu_count, Server.gpu_memory_total)
    )
    engine = create_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        rows = session.exec(stmt).all()

    run_hours = _alicloud_inspector_run_hours(repo_path)
    candidates = []
    for api_reference, gpu_count, gpu_memory_total, min_hr in rows:
        if _alicloud_gpu_benchmark_measured(repo_path, api_reference):
            continue
        candidates.append(
            {
                "api_reference": api_reference,
                "gpu_count": gpu_count or 0,
                "gpu_memory_total": gpu_memory_total or 0,
                "min_hr": float(min_hr),
                "cost": float(min_hr) * run_hours,
            }
        )

    budget = ALICLOUD_INSPECTOR_BUDGET_USD
    chosen: list[str] = []
    spent = 0.0

    big = sorted(
        candidates,
        key=lambda r: (-r["gpu_memory_total"], -r["gpu_count"], r["cost"]),
    )[:ALICLOUD_INSPECTOR_BIG_GPU_SLOTS]
    for row in big:
        if spent + row["cost"] <= budget:
            chosen.append(row["api_reference"])
            spent += row["cost"]

    for row in sorted(candidates, key=lambda r: r["cost"]):
        if row["api_reference"] in chosen:
            continue
        if spent + row["cost"] <= budget:
            chosen.append(row["api_reference"])
            spent += row["cost"]

    logging.info(
        "AliCloud GPU allowlist: %d unmeasured candidates, chose %d (~$%.0f of $%.0f budget, %.1fh/run est)",
        len(candidates),
        len(chosen),
        spent,
        budget,
        run_hours,
    )
    for api_reference in sorted(chosen):
        row = next(r for r in candidates if r["api_reference"] == api_reference)
        logging.info(
            "  allow %s (%.1fx VRAM %s MiB, $%.3f/hr, ~$%.1f/run)",
            api_reference,
            row["gpu_count"],
            row["gpu_memory_total"],
            row["min_hr"],
            row["cost"],
        )
    return frozenset(chosen)


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
def start(ctx, exclude, start_only, vendor):
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
    for (vnd, server), (srv_data, regions, zones, zone_to_region) in available_servers(vendor=vendor).items():
        if vnd == "aws" and not (server.startswith("m9g") or server == "c4.xlarge"):
            logging.info(f"Excluding {vnd}/{server}")
            continue
        if vnd == "alicloud" and server not in alicloud_inspector_allowlist(ctx.parent.params["repo_path"]):
            logging.info(f"Excluding {vnd}/{server} (not in AliCloud GPU budget allowlist)")
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
        f = executor.submit(lib.start_inspect, executor, lock, data_dir, vnd, server, tasks, srv_data, regions, zones, zone_to_region)
        futures[f] = (vnd, server)
        count += 1
        limit = 8 if vnd in {"alicloud", "aws", "upcloud", "vultr"} else 1
        if count == limit:
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
def cleanup(ctx, threads, force, all_regions, lookback_mins, data_only, vendor):
    from sc_runner.resources import supported_vendors
    import concurrent.futures

    repo_data = os.path.join(ctx.parent.params["repo_path"], "data")
    candidates = available_servers(vendor=vendor)
    candidates = {
        k: v
        for k, v in candidates.items()
        if lib.should_scan_for_cleanup(
            repo_data, k[0], k[1], lookback_mins=lookback_mins, data_only=data_only
        )
    }
    futures = []
    servers = lib.sort_available_servers(candidates, data_dir=repo_data)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for (vendor, server), (_, regions, zones, _zone_to_region) in servers:
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
@click.option("--gpu-count", default=0.0, type=float, help="Number of GPUs")
@click.option("--threads", default=8, show_default=True, help="Parallelism in a given task group")
def inspect(ctx, vendor, instance, gpu_count, threads):
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
    sparse = (f"data/{vendor}/{instance}",)
    repo.get_repo(sparse_paths=sparse)
    data_dir = os.path.join(ctx.parent.params["repo_path"], "data", vendor, instance)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    lib.record_timing_inspector_start(data_dir)
    try:
        repo.push_path(lib.timing_dir(data_dir), f"Inspector started from {repo.gha_url()}")
    except Exception:
        logging.exception("Failed to push inspector timing")
    try:
        lib.run_tasks(
            vendor,
            data_dir,
            instance=instance,
            gpu_count=gpu_count,
            nthreads=threads,
        )
    finally:
        lib.finalize_task_metas(vendor, data_dir, instance, gpu_count=gpu_count)
        lib.record_timing_inspector_end(data_dir)
        try:
            repo.push_path(lib.timing_dir(data_dir), f"Inspector finished from {repo.gha_url()}")
        except Exception:
            logging.exception("Failed to push inspector timing")


if __name__ == "__main__":
    cli()

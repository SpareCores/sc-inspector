"""Orchestrate DBaaS benchmark stack provisioning with regional fallback."""

from __future__ import annotations

import base64
import logging
import os
import threading
from datetime import timedelta

from azure_dbaas_quota import check_dbaas_postgres_quota, filter_clients_by_vm_quota
from benchmark_tiers import merge_client_requirements
from companion_picker import rank_client_instances
from dbaas_catalog import ManagedDbTarget
from dbaas_selector import stack_slug, target_sizing_stub
from dbaas_tiers import provision_spec
from lib import (
    USER_DATA,
    InstanceCreationTiming,
    boot_meta_for_task,
    candidate_regions,
    inspector_user_data_replacements,
    pulumi_stack_opts,
    record_instance_start_failure,
    record_timing_api,
    retry_locked,
    write_meta,
    META_NAME,
)
from sc_runner import runner
from sc_runner.resources.managed_db import DbaasStackSpec, ManagedDbSpec

import repo


def _dbaas_user_data_replacements(
    vendor: str,
    target: ManagedDbTarget,
    client,
    region: str,
    zone: str | None,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    provision: dict,
) -> dict[str, str]:
    stub = target_sizing_stub(target)
    repl = inspector_user_data_replacements(
        vendor,
        client.api_reference,
        stub,
        region,
        zone,
        timeout_mins,
        ssh_deploy_key_b64,
        repo_url_ssh,
        role="dbaas_client",
        include_run_upload=True,
    )
    repl.update(
        {
            "TOPOLOGY": "dbaas",
            "MANAGED_DB_INSTANCE_KEY": target.instance_key,
            "SC_DB_HOST": "{SC_DB_HOST}",
            "SC_DB_PORT": "5432",
            "SC_DB_USER": provision.get("admin_login", "scadmin"),
            "SC_DB_PASSWORD": "{SC_DB_PASSWORD}",
            "SC_DB_NAME": provision.get("database_name", "bench"),
            "SC_DB_BOOTSTRAP_USER": "postgres" if vendor == "gcp" else provision.get("admin_login", "scadmin"),
            "SC_DB_BOOTSTRAP_DATABASE": "postgres",
            "SC_DB_BOOTSTRAP_PASSWORD": "{SC_DB_PASSWORD}",
            "DB_WAIT_TIMEOUT_SEC": os.environ.get("DB_WAIT_TIMEOUT_SEC", "1200"),
            "MEM_GIB": str(target.memory_gib),
            "SC_PROVISION_VENDOR_ID": target.vendor_id,
            "SC_PROVISION_NATIVE_ID": target.native_id,
            "SC_PROVISION_ENGINE_VERSION": target.engine_version,
            "SC_PROVISION_HA_MODE": target.ha_mode,
            "SC_PROVISION_SKU_ID": target.sku_id,
            "SC_PROVISION_CPU_COUNT": str(target.cpu_count),
            "SC_PROVISION_MEMORY_GIB": str(target.memory_gib),
            "SC_PROVISION_STORAGE_GIB": str(provision["storage_gib"]),
            "SC_PROVISION_STORAGE_EDITION": provision["storage_edition"],
            "SC_PROVISION_IOPS_TIER": provision["iops_tier"],
            "SC_PROVISION_CLIENT_INSTANCE": client.api_reference,
            "SC_PROVISION_REGION": region,
            "SC_PROVISION_ZONE": zone or "",
            "SC_PROVISION_NETWORK_MODE": "private_vpc" if vendor == "gcp" else "private_vnet",
            "SC_PROVISION_CACHE_TIER": provision["cache_tier"],
            "SC_PROVISION_STACK_SLUG": stack_slug(target, provision["cache_tier"]),
            "SC_PROVISION_SYNC_COMMIT_SETTABLE": "",
            "USER_DATA_TEMPLATE": USER_DATA,
        }
    )
    return repl


def _group_tasks_by_cache_tier(tasks) -> dict[str, list]:
    groups: dict[str, list] = {}
    for task in tasks:
        groups.setdefault(task.cache_tier, []).append(task)
    return groups


def _build_dbaas_resource_opts(
    *,
    vendor: str,
    client,
    region: str,
    zone: str | None,
    slug: str,
    stack_spec: DbaasStackSpec,
) -> dict:
    opts = dict(
        public_key=os.environ.get("SSH_PUBLIC_KEY", ""),
        instance=client.api_reference,
        dbaas_slug=slug,
        dbaas=stack_spec,
    )
    if vendor == "gcp":
        if not zone:
            raise ValueError("GCP DBaaS requires a zone")
        opts["zone"] = zone
    else:
        opts["region"] = region
    return opts


def _dbaas_location_candidates(
    vendor: str,
    target: ManagedDbTarget,
    regions: list[str],
    zones: list[str],
    zone_to_region: dict[str, str | None] | None,
):
    from lib import candidate_regions, candidate_zones

    if vendor == "gcp" and zones:
        return [
            ("zone", z)
            for z in candidate_zones(vendor, target.native_id, zones, zone_to_region)
        ]
    return [("region", r) for r in candidate_regions(vendor, target.native_id, regions)]


def _destroy_dbaas_stack(
    vendor: str,
    resource_opts: dict,
    instance_logger,
    error_msgs: list,
) -> None:
    """Tear down a DBaaS Pulumi stack so slug-shared Azure resources can be reprovisioned."""
    try:
        runner.destroy_stack(
            vendor,
            {},
            resource_opts,
            stack_opts=pulumi_stack_opts(error_msgs, [], instance_logger),
        )
    except Exception:
        logging.exception(
            "DBaaS stack destroy failed for %s/%s",
            resource_opts.get("region"),
            resource_opts.get("instance"),
        )


def _try_provision_dbaas_stack(
    vendor: str,
    target: ManagedDbTarget,
    client,
    region: str,
    zone: str | None,
    cache_tier: str,
    provision: dict,
    slug: str,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    instance_logger,
    instance_timing,
    error_msgs,
) -> bool:
    logging.info(
        "DBaaS: %s/%s tier=%s + client %s in %s",
        vendor,
        target.instance_key,
        cache_tier,
        client.api_reference,
        region,
    )

    repl = _dbaas_user_data_replacements(
        vendor,
        target,
        client,
        region,
        zone,
        timeout_mins,
        ssh_deploy_key_b64,
        repo_url_ssh,
        provision,
    )

    md_spec = ManagedDbSpec(
        engine=target.engine,
        engine_version=target.engine_version,
        native_id=target.native_id,
        sku_name=provision["sku_name"],
        sku_tier=provision["sku_tier"],
        ha_mode=target.ha_mode,
        storage_gib=provision["storage_gib"],
        storage_type=provision["storage_type"],
        storage_edition=provision["storage_edition"],
        storage_iops_tier=provision["iops_tier"],
    )
    stack_spec = DbaasStackSpec(
        managed_db=md_spec,
        client_instance=client.api_reference,
        client_disk_gib=30,
        client_user_data_template=repl["USER_DATA_TEMPLATE"],
        client_user_data_static={
            k: v for k, v in repl.items() if k != "USER_DATA_TEMPLATE"
        },
        instance_key_slug=slug,
        extra_exports={
            "instance_key": target.instance_key,
            "cache_tier": cache_tier,
            "topology": "dbaas",
        },
    )

    resource_opts = _build_dbaas_resource_opts(
        vendor=vendor,
        client=client,
        region=region,
        zone=zone,
        slug=slug,
        stack_spec=stack_spec,
    )
    _destroy_dbaas_stack(vendor, resource_opts, instance_logger, error_msgs)
    pulumi_output = []
    stack_opts = pulumi_stack_opts(
        error_msgs, pulumi_output, instance_logger, instance_timing, client.api_reference
    )
    arch = client.cpu_architecture or "x86_64"
    extra: dict[str, str] = {}
    if vendor == "azure":
        extra["image_sku"] = "server-arm64" if "arm" in arch.lower() else "server"
    try:
        retry_locked(
            runner.create,
            vendor,
            {},
            resource_opts | extra,
            stack_opts=stack_opts,
            instance_timing=instance_timing,
        )
        return True
    except Exception as exc:
        logging.exception(
            "DBaaS create failed for %s with client %s",
            region,
            client.api_reference,
        )
        if not error_msgs:
            error_msgs.append(str(exc))
        _destroy_dbaas_stack(vendor, resource_opts, instance_logger, error_msgs)
        return False


def try_start_dbaas_inspect(
    executor,
    lock,
    data_dir: str,
    vendor: str,
    target: ManagedDbTarget,
    tasks,
    regions,
    zones,
    zone_to_region,
    timeout_mins: int,
    ssh_deploy_key_b64: str,
    repo_url_ssh: str,
    instance_logger,
    instance_timing,
    error_msgs,
) -> bool:
    """Provision managed DB + client VM; one stack per cache-tier group."""
    from lib import DbaasDbTask

    dbaas_tasks = [t for t in tasks if isinstance(t, DbaasDbTask)]
    if not dbaas_tasks:
        return False

    tier_groups = _group_tasks_by_cache_tier(dbaas_tasks)
    # One managed-DB stack per cache tier (C100 vs C30 storage). Provision a single
    # tier per start-dbaas invocation; re-run after cleanup for the next tier.
    cache_tier, tier_tasks = sorted(tier_groups.items())[0]
    stub = target_sizing_stub(target)
    provision = provision_spec(target, cache_tier)
    client_req = merge_client_requirements(
        [t.client_requirements(stub) for t in tier_tasks]
    )
    slug = stack_slug(target, cache_tier)

    for kind, location in _dbaas_location_candidates(
        vendor, target, regions, zones, zone_to_region
    ):
        if kind == "zone":
            zone = location
            region = (zone_to_region or {}).get(
                location, "-".join(location.split("-")[:-1])
            )
        else:
            zone = None
            region = location

        pg_ok, pg_reason = check_dbaas_postgres_quota(
            vendor,
            region,
            provision["sku_name"],
            int(target.cpu_count),
        )
        if not pg_ok:
            logging.info(
                "Skipping DBaaS %s/%s: %s",
                vendor,
                location,
                pg_reason,
            )
            continue

        clients = rank_client_instances(vendor, region, client_req)
        if not clients:
            logging.info("No DBaaS client for %s/%s", vendor, location)
            continue
        clients = filter_clients_by_vm_quota(vendor, region, clients)
        if not clients:
            continue

        for client in clients:
            if _try_provision_dbaas_stack(
                vendor,
                target,
                client,
                region,
                zone,
                cache_tier,
                provision,
                slug,
                timeout_mins,
                ssh_deploy_key_b64,
                repo_url_ssh,
                instance_logger,
                instance_timing,
                error_msgs,
            ):
                return True
    return False


def start_dbaas_inspect(
    executor,
    lock,
    data_dir: str,
    vendor: str,
    target: ManagedDbTarget,
    tasks,
    regions,
    zones,
    zone_to_region=None,
):
    """Entry point from inspector start-dbaas CLI."""
    from lib import DbaasDbTask

    current_thread = threading.current_thread()
    current_thread.name = f"{vendor}/{target.instance_key}"
    instance_logger = logging.getLogger(f"{vendor}/{target.instance_key}")

    dbaas_tasks = [t for t in tasks if isinstance(t, DbaasDbTask)]
    if not dbaas_tasks:
        raise RuntimeError("no DBaaS tasks to start")
    tier_groups = _group_tasks_by_cache_tier(dbaas_tasks)
    cache_tier, tier_tasks = sorted(tier_groups.items())[0]
    logging.info(
        "DBaaS start %s/%s cache_tier=%s (%d tasks; %d tier(s) pending overall)",
        vendor,
        target.instance_key,
        cache_tier,
        len(tier_tasks),
        len(tier_groups),
    )

    error_msgs = []
    instance_timing = InstanceCreationTiming()
    sum_timeout = timedelta()
    for task in tier_tasks:
        sum_timeout += task.timeout
    with lock:
        repo.pull()
        for task in tier_tasks:
            meta = boot_meta_for_task(task, data_dir)
            write_meta(meta, os.path.join(data_dir, task.name, META_NAME))
        repo.push_path(data_dir, f"Starting DBaaS from {repo.gha_url()}")

    timeout_mins = int(sum_timeout.total_seconds() / 60)
    github_repo = os.environ.get("GITHUB_REPOSITORY", "SpareCores/sc-inspector-data")
    repo_url_ssh = f"git@github.com:{github_repo}.git"
    ssh_deploy_key = os.environ.get("SSH_DEPLOY_KEY", "")
    ssh_deploy_key_b64 = (
        base64.b64encode(ssh_deploy_key.encode("utf-8")).decode("ascii") if ssh_deploy_key else ""
    )

    started = try_start_dbaas_inspect(
        executor,
        lock,
        data_dir,
        vendor,
        target,
        tier_tasks,
        regions,
        zones,
        zone_to_region,
        timeout_mins,
        ssh_deploy_key_b64,
        repo_url_ssh,
        instance_logger,
        instance_timing,
        error_msgs,
    )
    if started and instance_timing.complete():
        with lock:
            repo.pull()
            record_timing_api(data_dir, instance_timing.start, instance_timing.end)
            repo.push_path(data_dir, f"DBaaS creation timing from {repo.gha_url()}")
    if not started:
        record_instance_start_failure(lock, data_dir, tier_tasks, error_msgs)
        raise RuntimeError(
            error_msgs[0] if error_msgs else "DBaaS stack was not provisioned in any region"
        )

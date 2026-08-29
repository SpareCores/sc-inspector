"""List Pulumi stacks from the S3 backend used by sc-runner / inspector cleanup."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import logging
import os

# Vendors whose StackName is zone + instance (no region field).
_ZONE_VENDORS = frozenset({"gcp"})
# Vendors whose StackName is region + zone + instance.
_REGION_ZONE_VENDORS = frozenset({"aws", "azure"})
# Vendors whose StackName is region + instance.
_REGION_VENDORS = frozenset({"vultr", "hcloud", "upcloud", "ovh", "alicloud"})


@dataclass(frozen=True)
class PulumiStackRef:
    name: str
    vendor: str
    instance: str
    region: str | None = None
    zone: str | None = None
    size: int = 0


def _backend_bucket_and_prefix() -> tuple[str, str]:
    url = os.environ.get("PULUMI_BACKEND_URL", "").strip()
    if not url.startswith("s3://"):
        raise RuntimeError(
            f"PULUMI_BACKEND_URL must be an s3:// URL to list stacks (got {url!r})"
        )
    rest = url[len("s3://") :]
    bucket, _, _base = rest.partition("/")
    project = os.environ.get("PULUMI_PROJECT_NAME", os.environ.get("PULUMI_PROJECT", "runner"))
    # Pulumi DIY backend stores checkpoints under .pulumi/stacks/<project>/
    return bucket, f".pulumi/stacks/{project}"


def _s3_client():
    import boto3
    from botocore.config import Config

    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return boto3.client("s3", region_name=region, config=Config(signature_version="s3v4"))


def _normalize_loc(value: str | None) -> str | None:
    if value is None or value in {"None", "none", ""}:
        return None
    return value


def parse_stack_for_instance(stack_name: str, vendor: str, instance: str) -> PulumiStackRef | None:
    """Parse a stack name if it belongs to vendor/instance (instance may contain dots)."""
    prefix = f"{vendor}."
    if not stack_name.startswith(prefix):
        return None
    rest = stack_name[len(prefix) :]
    # Match ...<location>.<instance> or ...<location>.<instance>.<slug>
    token = f".{instance}"
    if rest == instance:
        # vendor.instance (no location) — unexpected for runner stacks
        return None
    if rest.endswith(token):
        middle = rest[: -len(token)]
    else:
        marker = f".{instance}."
        idx = rest.find(marker)
        if idx < 0:
            return None
        middle = rest[:idx]
    if not middle:
        return None

    region = None
    zone = None
    if vendor in _ZONE_VENDORS:
        zone = _normalize_loc(middle)
    elif vendor in _REGION_ZONE_VENDORS:
        region_part, sep, zone_part = middle.partition(".")
        if not sep:
            return None
        region = _normalize_loc(region_part)
        zone = _normalize_loc(zone_part)
    elif vendor in _REGION_VENDORS:
        region = _normalize_loc(middle)
    else:
        return None

    return PulumiStackRef(
        name=stack_name,
        vendor=vendor,
        instance=instance,
        region=region,
        zone=zone,
    )


def parse_stack_name(stack_name: str) -> PulumiStackRef | None:
    """Best-effort parse when the instance id is not known in advance.

    Prefer :func:`parse_stack_for_instance` when matching a concrete instance —
    AWS api_references contain dots (e.g. ``t3.micro``).
    """
    parts = stack_name.split(".")
    if len(parts) < 3:
        return None
    vendor = parts[0]
    if vendor in _ZONE_VENDORS:
        # gcp.zone.instance[.slug] — instance may contain dots: join remainder
        zone = parts[1]
        # Heuristic: last optional slug is a known dbaas token or leave as instance tail.
        instance = ".".join(parts[2:])
        # Strip trailing .None dbaas placeholder.
        if instance.endswith(".None"):
            instance = instance[: -len(".None")]
        return PulumiStackRef(
            name=stack_name,
            vendor=vendor,
            instance=instance,
            zone=_normalize_loc(zone),
        )
    if vendor in _REGION_VENDORS:
        region = parts[1]
        instance = ".".join(parts[2:])
        if instance.endswith(".None"):
            instance = instance[: -len(".None")]
        return PulumiStackRef(
            name=stack_name,
            vendor=vendor,
            instance=instance,
            region=_normalize_loc(region),
        )
    if vendor in _REGION_ZONE_VENDORS:
        region = parts[1]
        zone = parts[2]
        instance = ".".join(parts[3:])
        if instance.endswith(".None"):
            instance = instance[: -len(".None")]
        return PulumiStackRef(
            name=stack_name,
            vendor=vendor,
            instance=instance,
            region=_normalize_loc(region),
            zone=_normalize_loc(zone),
        )
    return None


@cache
def list_vendor_stack_names(vendor: str) -> tuple[tuple[str, int], ...]:
    """Return (stack_name, size) checkpoints for a vendor from the Pulumi S3 backend."""
    bucket, prefix = _backend_bucket_and_prefix()
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    stack_prefix = f"{prefix}/{vendor}."
    found: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=stack_prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if not key.endswith(".json") or key.endswith(".bak"):
                continue
            name = key.rsplit("/", 1)[-1][: -len(".json")]
            if not name.startswith(f"{vendor}."):
                continue
            found.append((name, int(obj.get("Size") or 0)))
    logging.info(
        "Pulumi backend listed %d %s stack checkpoint(s) under s3://%s/%s/",
        len(found),
        vendor,
        bucket,
        stack_prefix.rstrip("/"),
    )
    return tuple(found)


def locations_for_instance(vendor: str, instance: str) -> tuple[list[str], list[str]]:
    """Return (regions, zones) for stacks of this vendor/instance in the backend."""
    regions: list[str] = []
    zones: list[str] = []
    try:
        stacks = list_vendor_stack_names(vendor)
    except Exception:
        logging.exception(
            "Failed to list Pulumi backend stacks for %s/%s; falling back to catalog locations",
            vendor,
            instance,
        )
        return [], []
    for name, _size in stacks:
        parsed = parse_stack_for_instance(name, vendor, instance)
        if not parsed:
            continue
        if parsed.region:
            regions.append(parsed.region)
        if parsed.zone:
            zones.append(parsed.zone)
    regions = list(dict.fromkeys(regions))
    zones = list(dict.fromkeys(zones))
    if regions or zones:
        logging.info(
            "Pulumi backend locations for %s/%s: regions=%s zones=%s",
            vendor,
            instance,
            regions,
            zones,
        )
    return regions, zones


def stack_names_ending_with(vendor: str, suffixes: set[str]) -> list[str]:
    """Return raw backend stack names for `vendor` whose trailing dot-segment is in `suffixes`.

    Used to find DBaaS-shaped stacks (see ``dbaas_selector.stack_slug``) directly by
    their slug, without needing to know the client instance type in advance (unlike
    :func:`parse_stack_for_instance`, which requires the instance to locate the split
    point between location and instance).
    """
    if not suffixes:
        return []
    try:
        stacks = list_vendor_stack_names(vendor)
    except Exception:
        logging.exception("Failed to list Pulumi backend stacks for %s", vendor)
        return []
    return [name for name, _size in stacks if name.rsplit(".", 1)[-1] in suffixes]


def parse_dbaas_stack_location(stack_name: str, vendor: str) -> tuple[str | None, str | None]:
    """Return (region, zone) from a DBaaS-shaped stack name's leading location segments.

    Assumes ``<vendor>.<region>.<zone>.<...instance/slug>`` or ``<vendor>.<zone>.<...>``
    per vendor shape (see ``_REGION_ZONE_VENDORS`` / ``_ZONE_VENDORS`` / ``_REGION_VENDORS``
    above); the client-instance and dbaas-slug segments after the location are ignored.
    """
    parts = stack_name.split(".")
    if vendor in _ZONE_VENDORS:
        return None, _normalize_loc(parts[1]) if len(parts) > 1 else None
    if vendor in _REGION_ZONE_VENDORS:
        region = _normalize_loc(parts[1]) if len(parts) > 1 else None
        zone = _normalize_loc(parts[2]) if len(parts) > 2 else None
        return region, zone
    if vendor in _REGION_VENDORS:
        return (_normalize_loc(parts[1]) if len(parts) > 1 else None), None
    return None, None


def instances_with_stacks(vendor: str) -> set[str]:
    """Instance api_reference values that have at least one backend stack."""
    try:
        stacks = list_vendor_stack_names(vendor)
    except Exception:
        logging.exception("Failed to list Pulumi backend instances for %s", vendor)
        return set()
    out: set[str] = set()
    for name, _size in stacks:
        parsed = parse_stack_name(name)
        if parsed and parsed.instance:
            out.add(parsed.instance)
    return out


def clear_stack_list_cache() -> None:
    """Drop cached backend listings (tests / long-lived processes)."""
    list_vendor_stack_names.cache_clear()

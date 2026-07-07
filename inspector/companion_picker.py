"""Pick a companion client instance from the catalog for multi-VM Postgres benchmarks."""

from __future__ import annotations

import logging
from functools import cache, lru_cache
from types import SimpleNamespace
from typing import Any

from benchmark_tiers import ClientRequirements
from sc_crawler.table_fields import CpuArchitecture

STRESSNG_BEST1_BENCHMARK_ID = "stressngfull:best1"

# Benchmark client images are amd64-only; companions must run x86_64 VMs.
_X86_CPU_ARCHITECTURES = (CpuArchitecture.X86_64,)


@cache
def _catalog_engine():
    from sqlmodel import create_engine

    import sc_data

    return create_engine(f"sqlite:///{sc_data.db.path}")


@lru_cache(maxsize=256)
def _location_ids(vendor: str, location: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Resolve region and zone ids for a deployment location string."""
    from sc_crawler.tables import Region, Zone
    from sqlmodel import Session, select

    engine = _catalog_engine()
    with Session(engine) as session:
        region_ids = tuple(
            session.exec(
                select(Region.region_id)
                .where(Region.vendor_id == vendor)
                .where(Region.api_reference == location)
            ).all()
        )
        zone_ids = tuple(
            session.exec(
                select(Zone.zone_id)
                .where(Zone.vendor_id == vendor)
                .where(Zone.api_reference == location)
            ).all()
        )
    return region_ids, zone_ids


def _location_price_filter(region_ids: tuple[int, ...], zone_ids: tuple[int, ...]):
    from sc_crawler.tables import ServerPrice
    from sqlmodel import col, or_

    parts = []
    if region_ids:
        parts.append(col(ServerPrice.region_id).in_(region_ids))
    if zone_ids:
        parts.append(col(ServerPrice.zone_id).in_(zone_ids))
    if not parts:
        return None
    return or_(*parts) if len(parts) > 1 else parts[0]


def _server_api_refs_in_location(vendor: str, location: str) -> set[str]:
    """ACTIVE ONDEMAND server api_references available in a region or zone."""
    from sc_crawler.tables import Server, ServerPrice
    from sqlmodel import Session, select

    region_ids, zone_ids = _location_ids(vendor, location)
    loc_filter = _location_price_filter(region_ids, zone_ids)
    if loc_filter is None:
        return set()

    engine = _catalog_engine()
    with Session(engine) as session:
        stmt = (
            select(Server.api_reference)
            .join(ServerPrice, ServerPrice.server_id == Server.server_id)
            .where(ServerPrice.vendor_id == vendor)
            .where(ServerPrice.status == "ACTIVE")
            .where(ServerPrice.allocation == "ONDEMAND")
            .where(Server.status == "ACTIVE")
            .where(Server.gpu_count == 0)
            .where(loc_filter)
            .distinct()
        )
        return set(session.exec(stmt).all())


def _stressng_best1_scores(vendor: str, server_ids: list[int]) -> dict[int, float]:
    from sc_crawler.tables import BenchmarkScore
    from sqlmodel import Session, select

    if not server_ids:
        return {}
    engine = _catalog_engine()
    with Session(engine) as session:
        rows = session.exec(
            select(BenchmarkScore.server_id, BenchmarkScore.score)
            .where(BenchmarkScore.vendor_id == vendor)
            .where(BenchmarkScore.benchmark_id == STRESSNG_BEST1_BENCHMARK_ID)
            .where(BenchmarkScore.server_id.in_(server_ids))
        ).all()
    return {int(server_id): float(score) for server_id, score in rows}


def _eligible_servers_with_prices(
    vendor: str,
    location: str,
    req: ClientRequirements,
) -> list[tuple[Any, float]]:
    """Return (server stub, min_ondemand_price) rows meeting specs in location."""
    from sc_crawler.tables import Server, ServerPrice
    from sqlmodel import Session, select

    region_ids, zone_ids = _location_ids(vendor, location)
    loc_filter = _location_price_filter(region_ids, zone_ids)
    if loc_filter is None:
        return []

    mem_mib = req.min_memory_gib * 1024
    engine = _catalog_engine()
    with Session(engine) as session:
        rows = session.exec(
            select(
                Server.server_id,
                Server.api_reference,
                Server.vcpus,
                Server.memory_amount,
                Server.gpu_count,
                Server.cpu_architecture,
                ServerPrice.price,
            )
            .join(ServerPrice, ServerPrice.server_id == Server.server_id)
            .where(ServerPrice.vendor_id == vendor)
            .where(ServerPrice.status == "ACTIVE")
            .where(ServerPrice.allocation == "ONDEMAND")
            .where(Server.status == "ACTIVE")
            .where(Server.gpu_count == 0)
            .where(Server.cpu_architecture.in_(_X86_CPU_ARCHITECTURES))
            .where(Server.vcpus >= req.min_vcpus)
            .where(Server.memory_amount >= mem_mib)
            .where(loc_filter)
        ).all()

    best: dict[int, tuple[Any, float]] = {}
    for server_id, api_ref, vcpus, memory_amount, gpu_count, cpu_arch, price in rows:
        price_f = float(price)
        if server_id not in best or price_f < best[server_id][1]:
            stub = SimpleNamespace(
                server_id=server_id,
                api_reference=api_ref,
                vcpus=vcpus,
                memory_amount=memory_amount,
                gpu_count=gpu_count or 0,
                cpu_architecture=cpu_arch,
            )
            best[server_id] = (stub, price_f)
    return list(best.values())


def _rank_client_candidates(
    candidates: list[tuple[Any, float]],
    scores: dict[int, float],
) -> list[Any]:
    return [
        server
        for server, _price in sorted(
            candidates,
            key=lambda item: (
                -scores.get(item[0].server_id, 0.0),
                item[1],
                item[0].api_reference,
            ),
        )
    ]


def rank_client_instances(vendor: str, location: str, req: ClientRequirements) -> list[Any]:
    """Return catalog client servers in preference order (stress-ng score, then price)."""
    candidates = _eligible_servers_with_prices(vendor, location, req)
    if not candidates:
        return []
    server_ids = [s.server_id for s, _ in candidates]
    scores = _stressng_best1_scores(vendor, server_ids)
    return _rank_client_candidates(candidates, scores)


def pick_client_instance(vendor: str, location: str, req: ClientRequirements):
    """Return best catalog Server row or None."""
    ranked = rank_client_instances(vendor, location, req)
    if not ranked:
        logging.info(f"No client candidates in {vendor}/{location} for {req}")
        return None
    server = ranked[0]
    logging.info(
        f"Picked client {vendor}/{server.api_reference} in {location} "
        f"({len(ranked)} candidate(s))"
    )
    return server

"""Tests for Pulumi backend stack name parsing / location discovery."""
from __future__ import annotations

import pulumi_backend as pb


def test_parse_vultr_stack():
    ref = pb.parse_stack_for_instance(
        "vultr.blr.vcg-a40-24c-120g-48vram", "vultr", "vcg-a40-24c-120g-48vram"
    )
    assert ref is not None
    assert ref.region == "blr"
    assert ref.zone is None
    assert ref.instance == "vcg-a40-24c-120g-48vram"


def test_parse_vultr_ignores_other_instance():
    assert (
        pb.parse_stack_for_instance(
            "vultr.blr.vcg-a40-24c-120g-48vram", "vultr", "vc2-1c-0.5gb-free"
        )
        is None
    )


def test_parse_aws_dotted_instance():
    ref = pb.parse_stack_for_instance(
        "aws.us-west-2.us-west-2a.t3.micro", "aws", "t3.micro"
    )
    assert ref is not None
    assert ref.region == "us-west-2"
    assert ref.zone == "us-west-2a"
    assert ref.instance == "t3.micro"


def test_parse_aws_none_zone():
    ref = pb.parse_stack_for_instance("aws.us-east-1.None.t3.micro", "aws", "t3.micro")
    assert ref is not None
    assert ref.region == "us-east-1"
    assert ref.zone is None


def test_parse_gcp_zone_stack():
    ref = pb.parse_stack_for_instance(
        "gcp.europe-west4-a.g2-standard-24", "gcp", "g2-standard-24"
    )
    assert ref is not None
    assert ref.zone == "europe-west4-a"
    assert ref.region is None


def test_parse_gcp_dbaas_slug():
    ref = pb.parse_stack_for_instance(
        "gcp.us-central1-a.c4a-standard-4.cache-basic", "gcp", "c4a-standard-4"
    )
    assert ref is not None
    assert ref.zone == "us-central1-a"
    assert ref.instance == "c4a-standard-4"


def test_parse_stack_name_vultr():
    ref = pb.parse_stack_name("vultr.lhr.vcg-a40-24c-120g-48vram")
    assert ref is not None
    assert ref.region == "lhr"
    assert ref.instance == "vcg-a40-24c-120g-48vram"


def test_stack_names_ending_with_matches_dbaas_slug(monkeypatch):
    pb.clear_stack_list_cache()

    def fake_list(vendor):
        assert vendor == "aws"
        return (
            ("aws.us-east-1.us-east-1a.t3.micro", 15000),
            ("aws.us-east-1.us-east-1a.m6g.large.perfopt8-pg18", 20000),
            ("aws.us-west-2.None.m6g.large.memopt16-pg16", 20000),
        )

    monkeypatch.setattr(pb, "list_vendor_stack_names", fake_list)
    matches = pb.stack_names_ending_with("aws", {"perfopt8-pg18", "dbc4-pg17"})
    assert matches == ["aws.us-east-1.us-east-1a.m6g.large.perfopt8-pg18"]


def test_stack_names_ending_with_empty_suffixes():
    assert pb.stack_names_ending_with("aws", set()) == []


def test_parse_dbaas_stack_location_region_zone_vendor():
    region, zone = pb.parse_dbaas_stack_location(
        "aws.us-east-1.us-east-1a.m6g.large.perfopt8-pg18", "aws"
    )
    assert region == "us-east-1"
    assert zone == "us-east-1a"


def test_parse_dbaas_stack_location_region_zone_vendor_none_zone():
    region, zone = pb.parse_dbaas_stack_location(
        "aws.us-west-2.None.m6g.large.memopt16-pg16", "aws"
    )
    assert region == "us-west-2"
    assert zone is None


def test_parse_dbaas_stack_location_zone_vendor():
    region, zone = pb.parse_dbaas_stack_location(
        "gcp.us-central1-a.c4a-standard-4.cache-basic", "gcp"
    )
    assert region is None
    assert zone == "us-central1-a"


def test_locations_for_instance_uses_listed_stacks(monkeypatch):
    pb.clear_stack_list_cache()

    def fake_list(vendor):
        assert vendor == "vultr"
        return (
            ("vultr.blr.vcg-a40-24c-120g-48vram", 15000),
            ("vultr.lhr.vcg-a40-24c-120g-48vram", 15000),
            ("vultr.ewr.vc2-1c-0.5gb-free", 8000),
            ("vultr.sgp.vcg-a40-24c-120g-48vram", 400),
        )

    monkeypatch.setattr(pb, "list_vendor_stack_names", fake_list)
    regions, zones = pb.locations_for_instance("vultr", "vcg-a40-24c-120g-48vram")
    assert regions == ["blr", "lhr", "sgp"]
    assert zones == []

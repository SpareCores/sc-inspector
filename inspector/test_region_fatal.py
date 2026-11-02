"""Region/SKU-fatal classification drives DBaaS fallback."""

from lib import is_region_fatal_error, is_sku_fatal_error

AZURE_PG_INTERNAL_ERROR = (
    "1 error occurred:\n"
    "\t* Status=200 Code=\"InternalServerError\" Message=\"An unexpected error "
    "occured while processing the request. Tracking ID: "
    "'bb876d95-7fea-4bc2-b820-f1f3808d1c7e'\""
)


def test_azure_internal_server_error_is_region_fatal():
    assert is_region_fatal_error(RuntimeError("update failed"), [AZURE_PG_INTERNAL_ERROR])


def test_aws_network_limit_is_region_fatal():
    assert is_region_fatal_error(RuntimeError("VpcLimitExceeded: too many VPCs"))


def test_orphan_rg_is_region_fatal():
    assert is_region_fatal_error(
        RuntimeError("update failed"),
        ["cannot create already existing resource '/subscriptions/x/resourcegroups/eastasiad16adsv6-pg18'"],
    )


def test_regional_cores_exhausted_is_region_fatal():
    assert is_region_fatal_error(
        RuntimeError("update failed"),
        ["exceeding approved Total Regional Cores quota. Location: eastus2"],
    )


def test_instance_capacity_is_not_region_fatal():
    assert not is_region_fatal_error(
        RuntimeError("update failed"),
        ["error creating: InsufficientInstanceCapacity for t3.micro"],
    )


def test_tier_validation_error_is_not_region_fatal():
    assert not is_region_fatal_error(
        RuntimeError("update failed"),
        ['Status=400 Code="ParameterOutOfRange" Message="The value of the \'Sku.Tier\'"'],
    )


def test_compute_tier_mismatch_is_sku_fatal():
    assert is_sku_fatal_error(
        RuntimeError("update failed"),
        [
            'Status=200 Code="CannotProvisionComputeNameOnComputeTier" '
            "Message=\"Cannot provision a server with 'sku.name' set to "
            "'Standard_D16ads_v6' and 'sku.tier' set to 'GeneralPurpose'.\""
        ],
    )

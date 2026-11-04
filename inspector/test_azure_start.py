"""Azure VM start: retry/give-up decisions in start_inspect."""

import lib

GEN2_ERROR = (
    "The selected VM size 'Standard_A0' cannot boot Hypervisor Generation '2'. "
    "If this was a Create operation please check that the Hypervisor Generation "
    "of the Image matches the Hypervisor Generation of the selected VM Size."
)


def test_gen1_retry_on_gen2_boot_error():
    assert lib.azure_gen1_retry_image_sku("server", False, [GEN2_ERROR]) == "server-gen1"


def test_no_retry_for_other_errors():
    quota = "Operation could not be completed as it results in exceeding approved quota"
    assert lib.azure_gen1_retry_image_sku("server", False, [quota]) is None
    assert lib.azure_gen1_retry_image_sku("server", False, []) is None


def test_no_second_gen1_retry():
    # gen1 already tried: give up, so start_inspect schedules delayed_destroy
    assert lib.azure_gen1_retry_image_sku("server-gen1", False, [GEN2_ERROR]) is None


def test_confidential_vm_never_retries_on_gen1():
    # CVM sizes must stay on the Gen2 CVM image; a failure is final for the region
    # (and must still clean up the stack, see start_inspect)
    assert lib.azure_gen1_retry_image_sku("server", True, [GEN2_ERROR]) is None
    assert lib.azure_gen1_retry_image_sku("server", True, ["any other error"]) is None

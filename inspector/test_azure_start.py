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


def _locked():
    import pulumi

    return pulumi.automation.errors.ConcurrentUpdateError("stack is currently locked by 1 lock(s)")


def test_retry_locked_raises_when_every_attempt_is_locked(monkeypatch):
    import pytest

    monkeypatch.setattr(lib.time, "sleep", lambda s: None)
    calls = []

    def create():
        calls.append(1)
        raise _locked()

    errors = []
    with pytest.raises(lib.StackLockedError):
        lib.retry_locked(create, error_msgs=errors)
    assert len(calls) == 3
    # recorded as a retryable start error, so the SKU is retried later, not stranded
    assert errors and lib.is_retryable_start_error(errors[-1])


def test_retry_locked_returns_after_lock_clears(monkeypatch):
    monkeypatch.setattr(lib.time, "sleep", lambda s: None)
    attempts = iter([_locked(), _locked(), "created"])

    def create():
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    assert lib.retry_locked(create) == "created"

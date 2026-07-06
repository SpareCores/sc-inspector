import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from benchmark_tiers import (  # noqa: E402
    async_peak_vu_cap,
    client_max_vcpus,
    companion_client_vcpus,
    concurrency_ladder,
    max_profile_vus_by_warehouses,
    max_profile_vus_by_scalefactor,
    multi_vm_profile_ladder,
    multi_vm_workload_params,
    profile_vu_upper_bound,
    workload_for_cache_tier,
)


def test_async_f16_c30_ladder_includes_oversubscribe_rungs():
    w = workload_for_cache_tier(0.3, 16, 125.0, durability="async")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "async")
    assert wh_cap >= 32
    assert ladder == [1, 2, 4, 8, 16, 24, 32]


def test_async_f16_c100_ladder_capped_by_warehouses():
    w = workload_for_cache_tier(1.0, 16, 125.0, durability="async")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "async")
    assert wh_cap == 16
    assert ladder == [1, 2, 4, 8, 16]
    assert profile_vu_upper_bound(16, "async", wh_cap) == 16


def test_durable_f16_ladder_stays_conservative():
    w = workload_for_cache_tier(0.3, 16, 125.0, durability="durable")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "durable")
    assert 24 not in ladder
    assert 32 not in ladder
    assert ladder == [1, 2, 4, 8, 16]
    assert w.run_vus == 16


def test_small_host_ladder_scales_down():
    ladder = concurrency_ladder(4, 100, "async")
    assert ladder == [1, 2, 4, 6, 8]


def test_durable_small_host_no_oversubscribe():
    ladder = concurrency_ladder(4, 100, "durable")
    assert ladder == [1, 2, 4]


def test_large_host_async_oversubscribe():
    ladder = concurrency_ladder(64, 500, "async")
    assert 96 in ladder
    assert 128 in ladder


def test_durable_large_host_capped_at_sixteen():
    ladder = concurrency_ladder(64, 500, "durable")
    assert max(ladder) == 16
    assert 32 not in ladder


def test_huge_host_ladder_is_geometric_and_reaches_2x_vcpus():
    ladder = concurrency_ladder(800, 10_000, "async")
    assert max(ladder) == 1600
    assert 1200 in ladder
    for prev, cur in zip(ladder, ladder[1:]):
        assert cur <= prev * 2


def test_async_peak_vu_cap_scales_with_host():
    assert async_peak_vu_cap(16) == 32
    assert async_peak_vu_cap(800) == 1600


def test_companion_client_scales_for_huge_db():
    assert client_max_vcpus(800) == 800
    assert companion_client_vcpus(64, 800) == 400


def test_huge_host_durable_still_capped_low():
    ladder = concurrency_ladder(800, 10_000, "durable")
    assert max(ladder) == 16
    assert ladder == [1, 2, 4, 8, 16]


def test_companion_client_sizing_matches_multi_vm_measurements():
    # F16 async/durable OLTP: ~6.5 client cores observed at 16 VUs → 8 vCPU floor.
    assert client_max_vcpus(16) == 16
    assert companion_client_vcpus(16, 16) == 8


def test_companion_client_scales_down_on_small_db_host():
    assert client_max_vcpus(1) == 1
    assert companion_client_vcpus(1, 1) == 1


def test_benchbase_read_heavy_profiles_from_scalefactor_not_warehouses():
    params = multi_vm_workload_params("read_heavy", "benchbase", 1.0, 16, 125.0, "durable")
    assert params.scale_units == 50
    assert max_profile_vus_by_scalefactor(50) == 10
    ladder = multi_vm_profile_ladder(16, params.scale_units, "benchbase", "read_heavy", "durable")
    assert ladder == [1, 2, 4, 8, 10]


def test_benchbase_crud_simple_f16_async_reaches_2x_vcpus():
    params = multi_vm_workload_params("crud_simple", "benchbase", 1.0, 16, 125.0, "async")
    ladder = multi_vm_profile_ladder(16, params.scale_units, "benchbase", "crud_simple", "async")
    assert ladder == [1, 2, 4, 8, 16, 24, 32]


def test_benchbase_small_host_scales_down():
    params = multi_vm_workload_params("read_heavy", "benchbase", 1.0, 1, 4.0, "durable")
    ladder = multi_vm_profile_ladder(1, params.scale_units, "benchbase", "read_heavy", "durable")
    assert ladder == [1]


def test_huge_host_benchbase_async_scales_to_2x_vcpus():
    params = multi_vm_workload_params("crud_simple", "benchbase", 1.0, 1920, 32_000.0, "async")
    ladder = multi_vm_profile_ladder(1920, params.scale_units, "benchbase", "crud_simple", "async")
    assert max(ladder) == async_peak_vu_cap(1920)
    assert 1920 in ladder


def test_multi_vm_params_thread_durability():
    async_params = multi_vm_workload_params("oltp_mixed", "hammerdb", 0.3, 16, 125.0, "async")
    durable_params = multi_vm_workload_params("oltp_mixed", "hammerdb", 1.0, 16, 125.0, "durable")
    assert async_params.run_vus >= durable_params.run_vus

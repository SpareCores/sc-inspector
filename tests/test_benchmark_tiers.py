import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from benchmark_tiers import (  # noqa: E402
    concurrency_ladder,
    hammerdb_client_max_vcpus,
    hammerdb_client_vcpus,
    max_profile_vus_by_warehouses,
    multi_vm_workload_params,
    profile_vu_upper_bound,
    workload_for_cache_tier,
)


def test_async_f16_c30_ladder_includes_oversubscribe_rungs():
    w = workload_for_cache_tier(0.3, 16, 125.0, durability="async")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "async")
    assert wh_cap >= 32
    assert ladder == [1, 4, 8, 16, 24, 32]


def test_async_f16_c100_ladder_capped_by_warehouses():
    w = workload_for_cache_tier(1.0, 16, 125.0, durability="async")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "async")
    assert wh_cap == 16
    assert ladder == [1, 4, 8, 16]
    assert profile_vu_upper_bound(16, "async", wh_cap) == 16


def test_durable_f16_ladder_stays_conservative():
    w = workload_for_cache_tier(0.3, 16, 125.0, durability="durable")
    wh_cap = max_profile_vus_by_warehouses(w.warehouses, 16)
    ladder = concurrency_ladder(16, wh_cap, "durable")
    assert 24 not in ladder
    assert 32 not in ladder
    assert ladder == [1, 4, 8, 16]
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


def test_hammerdb_client_sizing_async_vs_durable():
    assert hammerdb_client_max_vcpus(16, "durable") == 16
    assert hammerdb_client_max_vcpus(16, "async") == 24
    async_vcpus = hammerdb_client_vcpus(32, 16, 16, "async")
    durable_vcpus = hammerdb_client_vcpus(16, 16, 16, "durable")
    assert async_vcpus > durable_vcpus


def test_multi_vm_params_thread_durability():
    async_params = multi_vm_workload_params("oltp_mixed", "hammerdb", 0.3, 16, 125.0, "async")
    durable_params = multi_vm_workload_params("oltp_mixed", "hammerdb", 1.0, 16, 125.0, "durable")
    assert async_params.run_vus >= durable_params.run_vus

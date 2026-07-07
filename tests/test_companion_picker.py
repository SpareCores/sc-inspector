import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

# companion_picker imports sc_crawler at module load time.
_sc = ModuleType("sc_crawler")
_sc_tf = ModuleType("sc_crawler.table_fields")
_sc_tf.CpuArchitecture = SimpleNamespace(X86_64="x86_64")
sys.modules["sc_crawler"] = _sc
sys.modules["sc_crawler.table_fields"] = _sc_tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from companion_picker import _rank_client_candidates, rank_client_instances  # noqa: E402


def test_rank_client_candidates_orders_by_score_then_price():
    servers = [
        SimpleNamespace(server_id=1, api_reference="Standard_B8als_v2"),
        SimpleNamespace(server_id=2, api_reference="Standard_D4s_v3"),
        SimpleNamespace(server_id=3, api_reference="Standard_E4s_v3"),
    ]
    candidates = [(servers[0], 0.5), (servers[1], 0.2), (servers[2], 0.3)]
    scores = {1: 10.0, 2: 20.0, 3: 20.0}

    ranked = _rank_client_candidates(candidates, scores)

    assert [s.api_reference for s in ranked] == [
        "Standard_D4s_v3",
        "Standard_E4s_v3",
        "Standard_B8als_v2",
    ]


@patch("companion_picker._stressng_best1_scores", return_value={1: 5.0, 2: 10.0})
@patch("companion_picker._eligible_servers_with_prices")
def test_rank_client_instances_returns_sorted_list(mock_eligible, _mock_scores):
    servers = [
        SimpleNamespace(server_id=1, api_reference="Standard_B8als_v2"),
        SimpleNamespace(server_id=2, api_reference="Standard_D4s_v3"),
    ]
    mock_eligible.return_value = [(servers[0], 0.1), (servers[1], 0.2)]

    ranked = rank_client_instances("azure", "northeurope", object())

    assert [s.api_reference for s in ranked] == ["Standard_D4s_v3", "Standard_B8als_v2"]


if __name__ == "__main__":
    tests = [
        test_rank_client_candidates_orders_by_score_then_price,
        test_rank_client_instances_returns_sorted_list,
    ]
    for fn in tests:
        fn()
        print(f"ok {fn.__name__}")
    print("ALL PASSED")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from lib import instance_start_order_key, sort_servers_for_start  # noqa: E402


def _servers(*names: str) -> dict:
    return {( "azure", name): [None, [], [], {}] for name in names}


def test_instance_start_order_key_is_stable():
    assert instance_start_order_key("azure", "Standard_D2s_v5") == instance_start_order_key(
        "azure", "Standard_D2s_v5"
    )


def test_sort_servers_for_start_is_deterministic_and_mixed():
    names = [
        "Standard_A1",
        "Standard_D96ads_v6",
        "Standard_F2s_v2",
        "Standard_NC24ads_A100_v4",
        "Standard_B2s",
    ]
    servers = _servers(*names)
    ordered = [key[1] for key, _ in sort_servers_for_start(servers)]
    assert ordered == [key[1] for key, _ in sort_servers_for_start(servers)]
    # Price order would put A1/B2s first; hash order should differ.
    assert ordered[:2] != ["Standard_A1", "Standard_B2s"]

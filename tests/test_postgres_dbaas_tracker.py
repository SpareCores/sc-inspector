import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inspector"))

from postgres_dbaas import _tracker_env  # noqa: E402


def test_tracker_env_includes_sentinel_and_job_name():
    task = SimpleNamespace(name="hammerdb_postgres_dbaas_oltp_mixed_durable_c100")
    with patch.dict(
        os.environ,
        {
            "GITHUB_RUN_ID": "28933304109",
            "SENTINEL_API_TOKEN": "secret",
            "SENTINEL_API_URL": "https://sentinel.example/api",
        },
        clear=False,
    ):
        env = _tracker_env(task)
    assert env["TRACKER_PROJECT_NAME"] == "inspector"
    assert env["TRACKER_JOB_NAME"] == "hammerdb_postgres_dbaas_oltp_mixed_durable_c100"
    assert env["TRACKER_EXTERNAL_RUN_ID"] == "28933304109"
    assert env["SENTINEL_API_TOKEN"] == "secret"
    assert env["SENTINEL_API_URL"] == "https://sentinel.example/api"
    assert env["TRACKER_QUIET"] == "true"


if __name__ == "__main__":
    test_tracker_env_includes_sentinel_and_job_name()
    print("ok")

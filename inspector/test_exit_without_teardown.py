"""CLI must exit once the command returns, even if interpreter teardown would hang.

Regression for start-instances (azure) in run 41660184518: "Git push successful"
was logged with no non-daemon threads left, then the process sat in teardown
for ~4h until the job was cancelled.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = textwrap.dedent(
    """
    import atexit
    import logging
    import sys
    import threading

    sys.path.insert(0, {here!r})
    import click
    import inspector

    @inspector.cli.command("teardown-hang")
    @click.option("--fail", is_flag=True)
    def teardown_hang(fail):
        threading.Thread(target=threading.Event().wait, name="stuck").start()
        atexit.register(threading.Event().wait)
        logging.info("command done")
        if fail:
            raise RuntimeError("boom")

    inspector.main()
    """
)


def _run(tmp_path, *args):
    script = tmp_path / "run_cli.py"
    script.write_text(SCRIPT.format(here=HERE))
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )


def test_success_exits_despite_blocked_teardown(tmp_path):
    res = _run(tmp_path, "teardown-hang")
    assert res.returncode == 0, res.stderr
    assert "command done" in res.stdout


def test_exception_exits_nonzero_with_traceback(tmp_path):
    res = _run(tmp_path, "teardown-hang", "--fail")
    assert res.returncode == 1
    assert "command done" in res.stdout
    assert "RuntimeError: boom" in res.stderr


def test_usage_error_keeps_click_exit_code(tmp_path):
    res = _run(tmp_path, "no-such-command")
    assert res.returncode == 2
    assert "No such command" in res.stderr


@pytest.mark.parametrize("code, expected", [(None, 0), (3, 3), ("fatal", 1)])
def test_system_exit_codes(monkeypatch, code, expected):
    import inspector as insp

    def fake_cli_main(args=None):
        raise SystemExit(code)

    exits = []
    monkeypatch.setattr(insp.cli, "main", fake_cli_main)
    monkeypatch.setattr(insp, "_exit_now", exits.append)
    insp.main()
    assert exits == [expected]

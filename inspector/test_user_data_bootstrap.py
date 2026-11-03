"""Bootstrap S3 fetch uses curl's own --retry, and does not retry presign errors.

Regression for an Azure DBaaS client whose single curl sat on the default 300s
connect timeout (exit 28) and never started the benchmark.
"""
from __future__ import annotations

import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BOOTSTRAP = Path(__file__).with_name("user_data_bootstrap.sh")


class _ScriptServer(ThreadingHTTPServer):
    hits: int
    fail_first: int
    fail_code: int
    body: bytes


def _serve(fail_first: int, fail_code: int, body: bytes) -> tuple[_ScriptServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            server: _ScriptServer = self.server  # type: ignore[assignment]
            server.hits += 1
            if server.hits <= server.fail_first:
                self.send_response(server.fail_code)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(server.body)))
            self.end_headers()
            self.wfile.write(server.body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    httpd = _ScriptServer(("127.0.0.1", 0), Handler)
    httpd.hits = 0
    httpd.fail_first = fail_first
    httpd.fail_code = fail_code
    httpd.body = body
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}/user_data.sh"


def _run_bootstrap(tmp_path: Path, url: str) -> subprocess.CompletedProcess[str]:
    script = BOOTSTRAP.read_text().replace("{USER_DATA_SCRIPT_URL}", url)
    path = tmp_path / "bootstrap.sh"
    path.write_text(script)
    path.chmod(0o755)
    env = os.environ.copy()
    env["INSPECTOR_BOOTSTRAP_LIB_ONLY"] = "1"
    return subprocess.run(
        ["sh", str(path)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_curl_retries_transient_http_then_succeeds(tmp_path: Path):
    httpd, url = _serve(2, 503, b"#!/bin/sh\necho ok\n")
    try:
        result = _run_bootstrap(tmp_path, url)
    finally:
        httpd.shutdown()

    assert result.returncode == 0, result.stderr
    assert httpd.hits == 3


def test_presign_http_error_is_not_retried(tmp_path: Path):
    httpd, url = _serve(5, 403, b"#!/bin/sh\n")
    try:
        result = _run_bootstrap(tmp_path, url)
    finally:
        httpd.shutdown()

    assert result.returncode == 22
    assert httpd.hits == 1


def test_empty_download_fails(tmp_path: Path):
    httpd, url = _serve(0, 200, b"")
    try:
        result = _run_bootstrap(tmp_path, url)
    finally:
        httpd.shutdown()

    assert result.returncode == 1
    assert httpd.hits == 1
    assert "empty" in result.stderr

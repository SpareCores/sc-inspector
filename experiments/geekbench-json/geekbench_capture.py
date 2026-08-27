#!/usr/bin/env python3
"""Capture Geekbench Browser upload POSTs on localhost."""

from __future__ import annotations

import json
import re
import ssl
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

CAPTURE_DIR = Path("/tmp/geekbench-capture")
DOCUMENT_PATH = CAPTURE_DIR / "upload-document.json"
CERT_DIR = Path("/usr/local/share/geekbench-capture")
HOSTNAME = "browser.geekbench.com"
FAKE_RESULT_ID = 99999999
FAKE_CLAIM_KEY = 0


def upload_response_body() -> bytes:
    return json.dumps({"id": FAKE_RESULT_ID, "key": FAKE_CLAIM_KEY}).encode()


def _extract_document(body: bytes) -> dict | None:
    content_type = ""
    for line in body.split(b"\r\n", 1)[0:1]:
        pass
    # Body is raw HTTP body; Content-Type comes from handler headers instead.
    return None


def extract_document_from_multipart(body: bytes, content_type: str) -> dict | None:
    match = re.search(r'boundary=(.+)', content_type)
    if not match:
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None
    boundary = match.group(1).strip().strip('"')
    marker = b"--" + boundary.encode()
    for part in body.split(marker):
        if b'name="document"' not in part:
            continue
        payload = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
        return json.loads(payload)
    return None


class CaptureHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[geekbench-capture] {fmt % args}\n")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        document = extract_document_from_multipart(body, content_type)
        if document is not None:
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            DOCUMENT_PATH.write_text(json.dumps(document))
            self.log_message("saved upload document (%d bytes)", len(body))
        else:
            self.log_message("upload without document field (%d bytes)", len(body))

        response = upload_response_body()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def ensure_cert() -> tuple[Path, Path]:
    key = CERT_DIR / "server.key"
    crt = CERT_DIR / "server.crt"
    ca_link = Path(f"/usr/local/share/ca-certificates/{HOSTNAME}.crt")
    if not key.exists() or not crt.exists():
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key),
                "-out",
                str(crt),
                "-days",
                "3650",
                "-nodes",
                "-subj",
                f"/CN={HOSTNAME}",
            ],
            check=True,
        )
    if not ca_link.exists() or ca_link.read_bytes() != crt.read_bytes():
        subprocess.run(["cp", str(crt), str(ca_link)], check=True)
        subprocess.run(["update-ca-certificates"], check=True)
    return key, crt


def main() -> None:
    key, crt = ensure_cert()
    server = HTTPServer(("127.0.0.1", 443), CaptureHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    sys.stderr.write(f"[geekbench-capture] listening on 127.0.0.1:443 as {HOSTNAME}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()

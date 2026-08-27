#!/usr/bin/env python3
"""Validate upload-document -> results.json conversion against captured data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "inspector"))
from geekbench import geekbench_upload_document_to_json  # noqa: E402


def main() -> None:
    capture = Path(__file__).parent / "capture-out" / "upload_POST__v6_cpu_upload.json.bin"
    raw = capture.read_bytes()
    boundary = raw.split(b"\r\n", 1)[0].decode().strip("-")
    document = None
    for part in raw.split(b"--" + boundary.encode()):
        if b'name="document"' in part:
            document = json.loads(part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0])
            break
    assert document is not None
    results = geekbench_upload_document_to_json(document)
    print(json.dumps(results, indent=2))
    assert "Single-Core Performance" in results
    assert "Multi-Core Performance" in results
    assert results["Single-Core Performance"]["Score"]["score"] == document["sections"][0]["score"]
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()

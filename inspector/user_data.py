"""Compress templated user_data.sh into a self-extracting shell archive."""

from __future__ import annotations

import base64
import lzma
import subprocess
import textwrap
from pathlib import Path

AWS_USER_DATA_LIMIT = 16 * 1024
VENDOR_USER_DATA_LIMITS: dict[str, int] = {"aws": AWS_USER_DATA_LIMIT}

# ZeroMQ z85 (GNU basenc --z85); ported from CPython 3.14 base64.z85encode.
_b85alphabet = (
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    b"abcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~"
)
_z85alphabet = (
    b"0123456789abcdefghijklmnopqrstuvwxyz"
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
)
_z85_encode_translation = bytes.maketrans(_b85alphabet, _z85alphabet)

_ARCHIVE_BODY_HEADER = (
    "basenc -d --z85 -i <<'INSPECTOR_USER_DATA' | head -c \"$XZ_BYTES\" | xz -dc | bash\n"
)
_ARCHIVE_FOOTER = "\nINSPECTOR_USER_DATA\n"


def _z85encode(data: bytes) -> bytes:
    pad = (4 - len(data) % 4) % 4
    padded = data + b"\0" * pad
    return base64.b85encode(padded, pad=False).translate(_z85_encode_translation)


def pack_user_data_script(script: str) -> str:
    """Wrap *script* in an xz+z85 self-extracting shell archive."""
    compressed = lzma.compress(script.encode("utf-8"), preset=9)
    payload = "\n".join(textwrap.wrap(_z85encode(compressed).decode("ascii"), 76))
    return (
        f"#!/bin/bash\nXZ_BYTES={len(compressed)}\n"
        + _ARCHIVE_BODY_HEADER
        + payload
        + _ARCHIVE_FOOTER
    )


def apply_replacements(template: str, replacements: dict[str, str]) -> str:
    script = template
    for key, value in replacements.items():
        script = script.replace("{" + key + "}", str(value))
    return script


def render_packed_user_data(
    template: str,
    replacements: dict[str, str],
    *,
    vendor: str | None = None,
) -> str:
    """Substitute placeholders and pack for any cloud vendor."""
    packed = pack_user_data_script(apply_replacements(template, replacements))
    if vendor and (limit := VENDOR_USER_DATA_LIMITS.get(vendor)):
        nbytes = len(packed.encode("utf-8"))
        if nbytes > limit:
            raise RuntimeError(
                f"{vendor} packed user_data is {nbytes} bytes (limit {limit})"
            )
    return packed


def _worst_case_replacements() -> dict[str, str]:
    return {
        "SSH_DEPLOY_KEY_B64": base64.b64encode(b"x" * 4096).decode(),
        "REPO_URL": "git@github.com:SpareCores/sc-inspector-data.git",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "SpareCores/sc-inspector-data",
        "GITHUB_RUN_ID": "99999999999-9",
        "BENCHMARK_SECRETS_PASSPHRASE": "x" * 128,
        "SENTINEL_API_TOKEN": "",
        "HF_TOKEN": "hf_" + "x" * 64,
        "VENDOR": "aws",
        "INSTANCE": "m9gd.metal-48xl",
        "GPU_COUNT": "8",
        "SHUTDOWN_MINS": "999",
        "HOST_TIMING_DIR": "/var/lib/inspector-timing/aws/m9gd.metal-48xl/99999999999-9",
    }


def _simulate_substitutions(template: str) -> str:
    """Worst-case placeholder sizes for local size checks."""
    return apply_replacements(template, _worst_case_replacements())


def _decode_archive_payload(archive: str) -> bytes:
    xz_bytes = int(archive.split("\n", 2)[1].split("=", 1)[1])
    body = archive.split(_ARCHIVE_BODY_HEADER, 1)[1].rsplit(_ARCHIVE_FOOTER, 1)[0]
    z85 = body.replace("\n", "").encode("ascii")
    decoded = subprocess.check_output(["basenc", "-d", "--z85", "-i"], input=z85)
    return decoded[:xz_bytes]


def _self_test() -> None:
    template_path = Path(__file__).with_name("user_data.sh")
    template = template_path.read_text()
    replacements = _worst_case_replacements()
    script = apply_replacements(template, replacements)
    archive = render_packed_user_data(template, replacements, vendor="aws")
    compressed = lzma.compress(script.encode("utf-8"), preset=9)
    raw_len = len(script.encode())
    archive_len = len(archive.encode())
    b64_len = len(base64.b64encode(compressed))
    z85_len = len(_z85encode(compressed))
    print(f"substituted script: {raw_len} bytes")
    print(f"xz payload:         {len(compressed)} bytes")
    print(f"base64 payload:     {b64_len} bytes")
    print(f"z85 payload:        {z85_len} bytes (save {b64_len - z85_len} vs b64)")
    print(f"packed archive:     {archive_len} bytes (limit {AWS_USER_DATA_LIMIT})")
    if archive_len > AWS_USER_DATA_LIMIT:
        raise SystemExit(f"packed archive exceeds AWS limit by {archive_len - AWS_USER_DATA_LIMIT} bytes")

    decoded = _decode_archive_payload(archive)
    if lzma.decompress(decoded) != script.encode():
        raise SystemExit("roundtrip mismatch")

    smoke_archive = pack_user_data_script("#!/bin/bash\necho ARCHIVE_SMOKE_OK\n")
    proc = subprocess.run(["bash", "-s"], input=smoke_archive, text=True, capture_output=True)
    if proc.returncode != 0 or "ARCHIVE_SMOKE_OK" not in proc.stdout:
        raise SystemExit(
            f"smoke run failed: rc={proc.returncode} stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )

    print("self-test OK")


if __name__ == "__main__":
    _self_test()

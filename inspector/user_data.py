"""Publish full user_data scripts to S3 and render a tiny bootstrap stub."""

from __future__ import annotations

from pathlib import Path

# Placeholders resolved after the VM exists (Pulumi Output.apply / DBaaS create).
LATE_BINDING_KEYS = frozenset(
    {
        "CLIENT_PRIVATE_IP",
        "SC_DB_HOST",
        "SC_DB_PASSWORD",
    }
)

_BOOTSTRAP_PATH = Path(__file__).with_name("user_data_bootstrap.sh")
BOOTSTRAP_TEMPLATE = _BOOTSTRAP_PATH.read_text()


def apply_replacements(template: str, replacements: dict[str, str]) -> str:
    script = template
    for key, value in replacements.items():
        script = script.replace("{" + key + "}", str(value))
    return script


def render_bootstrap(script_url: str) -> str:
    return apply_replacements(BOOTSTRAP_TEMPLATE, {"USER_DATA_SCRIPT_URL": script_url})


def split_replacements(replacements: dict[str, str]) -> tuple[dict[str, str], frozenset[str]]:
    late = frozenset(k for k in LATE_BINDING_KEYS if k in replacements)
    early = {k: v for k, v in replacements.items() if k not in late}
    return early, late


def publish_user_data_script(
    vendor: str,
    instance: str,
    template: str,
    replacements: dict[str, str],
) -> tuple[str, str]:
    """Upload the full script to S3 and return (bootstrap_script, bootstrap_b64)."""
    import base64

    import s3_runs

    early, late = split_replacements(replacements)
    script = apply_replacements(template, early)
    key = s3_runs.upload_user_data_script(vendor, instance, script)
    url = s3_runs.presigned_user_data_script_get_url(key)
    bootstrap = render_bootstrap(url)
    b64 = base64.b64encode(bootstrap.encode("utf-8")).decode("ascii")
    return bootstrap, b64

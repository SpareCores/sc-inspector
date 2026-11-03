#!/bin/sh
set -eu

# Tests set this so the stub can be run without cloud-init's log file.
if [ "${INSPECTOR_BOOTSTRAP_LIB_ONLY:-}" != 1 ]; then
    # Minimal EC2/cloud-init stub: fetch the full user_data.sh from S3 and execute it.
    set -x
    exec >> /var/log/user_data.log 2>&1
fi

script=$(mktemp)
trap 'rm -f "$script"' EXIT

# cloud-init scripts_user can run before egress or DNS works. One bare curl then
# sits on the 300s default connect timeout (exit 28) and the boot never starts.
# --retry covers that timeout, resolve failures, and the transient HTTP statuses
# (408/429/5xx). --retry-connrefused adds ECONNREFUSED. --retry-all-errors is
# intentionally off: with -f it would also retry an expired or missing presign.
# --connect-timeout bounds each attempt; --retry-max-time bounds the whole series.
curl -sfS \
    --connect-timeout 15 \
    --max-time 120 \
    --retry 7 \
    --retry-connrefused \
    --retry-max-time 300 \
    -o "$script" \
    '{USER_DATA_SCRIPT_URL}'
# Guard against a truncated/empty download that somehow returned 200.
if [ ! -s "$script" ]; then
    echo "user_data script download is empty" >&2
    exit 1
fi
chmod +x "$script"

if [ "${INSPECTOR_BOOTSTRAP_LIB_ONLY:-}" = 1 ]; then
    exit 0
fi

# Late-bound placeholders (filled by Pulumi for multi-VM / DBaaS stacks).
apply_late_bindings() {
    python3 - <<'PY'
import os
import pathlib

path = pathlib.Path(os.environ["INSPECTOR_USER_DATA_SCRIPT"])
text = path.read_text()
for key, value in os.environ.items():
    if not key.startswith("INSPECTOR_LATE_"):
        continue
    name = key.removeprefix("INSPECTOR_LATE_")
    placeholder = "{" + name + "}"
    if not value or (value.startswith("{") and value.endswith("}")):
        continue
    text = text.replace(placeholder, value)
path.write_text(text)
PY
}

export INSPECTOR_USER_DATA_SCRIPT="$script"
export INSPECTOR_LATE_CLIENT_PRIVATE_IP="{CLIENT_PRIVATE_IP}"
export INSPECTOR_LATE_SC_DB_HOST="{SC_DB_HOST}"
export INSPECTOR_LATE_SC_DB_PORT="{SC_DB_PORT}"
export INSPECTOR_LATE_SC_DB_PASSWORD="{SC_DB_PASSWORD}"
apply_late_bindings

exec "$script"

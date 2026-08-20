#!/bin/sh -x
set -eu

# Minimal EC2/cloud-init stub: fetch the full user_data.sh from S3 and execute it.
exec >> /var/log/user_data.log 2>&1

script=$(mktemp)
trap 'rm -f "$script"' EXIT

# Fail hard on HTTP errors (expired/missing presign) — without -e we used to chmod
# an empty tempfile and "succeed" with a no-op boot.
curl -sfS -o "$script" '{USER_DATA_SCRIPT_URL}'
# Guard against a truncated/empty download that somehow returned 200.
if [ ! -s "$script" ]; then
    echo "user_data script download is empty" >&2
    exit 1
fi
chmod +x "$script"

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
export INSPECTOR_LATE_SC_DB_PASSWORD="{SC_DB_PASSWORD}"
apply_late_bindings

exec "$script"

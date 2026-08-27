#!/bin/sh
set -eu

arch="$(uname -m)"
capture_dir="/tmp/geekbench-capture"
upload_document="${capture_dir}/upload-document.json"

cleanup() {
  if [ -n "${capture_pid:-}" ]; then
    kill "${capture_pid}" 2>/dev/null || true
    wait "${capture_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

grep -q '[[:space:]]browser\.geekbench\.com$' /etc/hosts || \
  echo '127.0.0.1 browser.geekbench.com' >> /etc/hosts

python3 /opt/geekbench_capture.py 2>/tmp/geekbench-capture.log &
capture_pid=$!

ready=0
for _ in $(seq 1 100); do
  if python3 - <<'PY' 2>/dev/null
import socket
s = socket.create_connection(("127.0.0.1", 443), timeout=0.2)
s.close()
PY
  then
    ready=1
    break
  fi
  sleep 0.1
done
if [ "${ready}" -ne 1 ]; then
  echo "Geekbench upload capture server failed to start" >&2
  cat /tmp/geekbench-capture.log >&2
  exit 1
fi

geekbench_stderr="/tmp/geekbench.stderr"
/opt/geekbench/geekbench6 --cpu --upload 2>"${geekbench_stderr}" | egrep -v 'geekbench\.com.*claim'
geekbench_status=$?
cat "${geekbench_stderr}" >&2

if [ -f "${upload_document}" ]; then
  {
    printf 'GEEKBENCH_UPLOAD_DOCUMENT\n'
    cat "${upload_document}"
    printf '\nGEEKBENCH_UPLOAD_DOCUMENT_END\n'
  } >&2
fi

exit "${geekbench_status}"

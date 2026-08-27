#!/bin/bash
# Test redirecting browser.geekbench.com to local capture server
set -euo pipefail

echo "127.0.0.1 browser.geekbench.com" >> /etc/hosts

# Start capture server in background
python3 /opt/capture-server.py &
SERVER_PID=$!
sleep 2

echo "Running geekbench (1 CPU)..."
taskset -c 0 /opt/geekbench/geekbench6 --cpu > /tmp/geekbench-capture/stdout 2> /tmp/geekbench-capture/stderr || true

kill $SERVER_PID 2>/dev/null || true

echo "=== stdout ==="
cat /tmp/geekbench-capture/stdout
echo "=== captured files ==="
ls -la /tmp/geekbench-capture/

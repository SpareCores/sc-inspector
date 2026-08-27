# Geekbench upload JSON capture experiments

## Problem

Geekbench free edition uploads structured benchmark data to
`https://browser.geekbench.com/v6/cpu/upload.json`, but only the Pro
edition can export JSON locally. Fetching results back from the browser
URL no longer works because the site is behind Cloudflare.

## Solution (approach 1)

Redirect `browser.geekbench.com` to localhost, terminate TLS with a
self-signed certificate trusted via the system CA store (Geekbench uses
`/etc/ssl/certs/ca-certificates.crt`), and capture the multipart POST
body. The upload payload contains a `document` field with full benchmark
JSON including per-workload scores and rates.

This works without a Pro license and does not depend on eBPF, LD_PRELOAD,
or SSL interception.

## Verified (x86, Geekbench 6.7.1 free)

- POST path: `/v6/cpu/upload.json`
- Body: `multipart/form-data` with JSON `document` field (~32 KB)
- Capture server saves `/tmp/geekbench-capture/upload-document.json`
- Geekbench expects upload response JSON: `{"id": <int>, "key": <int>}`
- `geekbench.sh` re-emits the document on stderr between markers for the
  inspector transform step

## Running the experiment

```bash
docker build -t geekbench-e2e experiments/geekbench-json/
docker run --rm --cpus=1 \
  -v "$PWD/experiments/geekbench-json/e2e-out:/out" \
  geekbench-e2e bash -c 'bash /opt/geekbench.sh > /out/stdout 2> /out/stderr'
```

Expect ~15–20 minutes on a single CPU.

## Production integration

See:

- `sc-images/images/benchmark/geekbench_capture.py` – local HTTPS capture server
- `sc-images/images/benchmark/geekbench.sh` – starts capture, runs benchmark
- `sc-inspector/inspector/geekbench.py` – `geekbench_upload_document_to_json()`
- `sc-inspector/inspector/transform.py` – reads stderr markers instead of HTTP

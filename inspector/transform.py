import json
import os
import re
import requests


def raw(meta, task, task_dir, stdout, stderr) -> list[str]:
    outputs: list[str] = []
    for name in ("stdout", "stderr"):
        path = os.path.join(task_dir, name)
        data = locals()[name]
        if len(data):
            with open(path, "wb") as f:
                f.write(data)
            outputs.append(name)
        elif os.path.exists(path):
            # Drop stale stream from a prior run so meta.outputs matches the tree.
            os.remove(path)

    return outputs


def fetch_geekbench_results(meta, task, task_dir, stdout, stderr) -> list[str]:
    from geekbench import (
        geekbench_html_to_json,
        geekbench_upload_document_from_stderr,
        geekbench_upload_document_to_json,
    )

    outputs: list[str] = []
    document = geekbench_upload_document_from_stderr(stderr)
    if document:
        results = geekbench_upload_document_to_json(document)
    else:
        urls = re.findall(
            re.compile(r'https://[^\s"]*/v6/cpu[^\s"]*'), stdout.decode("utf-8")
        )
        if not urls:
            return outputs
        res = requests.get(urls[0])
        assert 200 <= res.status_code < 300, f"Status code is {res.status_code}"
        results = geekbench_html_to_json(res.text)
    with open(os.path.join(task_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    outputs.append("results.json")

    return outputs

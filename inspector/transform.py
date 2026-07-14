import json
import os
import re
import requests


def raw(meta, task, task_dir, stdout, stderr) -> list[str]:
    outputs: list[str] = []
    for name in ("stdout", "stderr"):
        if len(locals()[name]):
            with open(os.path.join(task_dir, name), "wb") as f:
                f.write(locals()[name])
            outputs.append(name)

    return outputs


def fetch_geekbench_results(meta, task, task_dir, stdout, stderr) -> list[str]:
    from geekbench import geekbench_html_to_json

    outputs: list[str] = []
    urls = re.findall(
        re.compile(r'https://[^\s"]*/v6/cpu[^\s"]*'), stdout.decode("utf-8")
    )
    if urls:
        res = requests.get(urls[0])
        assert 200 <= res.status_code < 300, f"Status code is {res.status_code}"
        results = geekbench_html_to_json(res.text)
        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        outputs.append("results.json")

    return outputs

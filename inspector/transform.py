import json
import os
import re
import requests

from lxml import html
from sc_crawler.lookup import benchmarks

GEEKBENCH_TABLES = ["Single-Core Performance", "Multi-Core Performance"]
GEEKBENCH_BENCHMARKS = [
    b.name.split(": ")[1]
    for b in benchmarks
    if b.framework == "geekbench" and b.name.split(": ")[1] != "Score"
]


def raw(meta, task, task_dir, stdout, stderr) -> list[str]:
    outputs: list[str] = []
    for name in ("stdout", "stderr"):
        if len(locals()[name]):
            with open(os.path.join(task_dir, name), "wb") as f:
                f.write(locals()[name])
            outputs.append(name)

    return outputs


def _geekbench_table(doc: html.Element, name: str) -> html.Element:
    return doc.xpath(
        f'.//div[@class="heading"]/h3[text()="{name}"]/../following-sibling::div[@class="table-wrapper"][1]/table'
    )[0]


def _geekbench_th_score(table: html.Element) -> int:
    return int(table.xpath('.//th[@class="score"]/text()')[0].strip())


def _geekbench_td_scores(table: html.Element, name: str) -> dict:
    cell = table.xpath(
        f'.//td[@class="name" and normalize-space(string())="{name}"]/../td[@class="score"]'
    )[0]
    return {
        "score": int(cell.xpath("text()")[0].strip()),
        "description": cell.xpath('span[@class="description"]/text()')[0],
    }


def fetch_geekbench_results(meta, task, task_dir, stdout, stderr) -> list[str]:
    outputs: list[str] = []
    urls = re.findall(
        re.compile(r'https://[^\s"]*/v6/cpu[^\s"]*'), stdout.decode("utf-8")
    )
    if urls:
        res = requests.get(urls[0])
        assert 200 <= res.status_code < 300, f"Status code is {res.status_code}"
        doc = html.fromstring(res.content)
        results = {}
        for tstring in GEEKBENCH_TABLES:
            results[tstring] = {}
            table = _geekbench_table(doc, tstring)
            results[tstring]["Score"] = _geekbench_th_score(table)
            for bstring in GEEKBENCH_BENCHMARKS:
                results[tstring][bstring] = _geekbench_td_scores(table, bstring)
        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        outputs.append("results.json")

    return outputs

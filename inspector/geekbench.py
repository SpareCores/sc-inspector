from lxml import html
from sc_crawler.lookup import benchmarks


GEEKBENCH_TABLES = ["Single-Core Performance", "Multi-Core Performance"]
GEEKBENCH_BENCHMARKS = [
    b.name.split(": ")[1]
    for b in benchmarks
    if b.framework == "geekbench" and b.name.split(": ")[1] != "Score"
]


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


def geekbench_html_to_json(html_content: str) -> dict:
    doc = html.fromstring(html_content)
    results = {}
    for tstring in GEEKBENCH_TABLES:
        results[tstring] = {}
        table = _geekbench_table(doc, tstring)
        results[tstring]["Score"] = _geekbench_th_score(table)
        for bstring in GEEKBENCH_BENCHMARKS:
            results[tstring][bstring] = _geekbench_td_scores(table, bstring)
    return results

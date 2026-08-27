import json

from lxml import html
from sc_crawler.lookup import benchmarks


GEEKBENCH_TABLES = ["Single-Core Performance", "Multi-Core Performance"]
GEEKBENCH_SECTION_NAMES = {
    "Single-Core": "Single-Core Performance",
    "Multi-Core": "Multi-Core Performance",
}
GEEKBENCH_UPLOAD_MARKERS = (
    "GEEKBENCH_UPLOAD_DOCUMENT\n",
    "\nGEEKBENCH_UPLOAD_DOCUMENT_END",
)
GEEKBENCH_BENCHMARKS = [
    b.name.split(": ")[1]
    for b in benchmarks
    if b.framework == "geekbench" and b.name.split(": ")[1] != "Score"
]

_UNIT_SUFFIX = {
    11: ("pairs/sec", 1e6, 1),
    12: ("FPS", 1, 1),
    13: ("TE/sec", 1e6, 2),
    15: ("rows/sec", 1e3, 1),
    16: ("Elements/sec", 1e6, 2),
    17: ("Words/sec", 1, 1),
    20: ("images/sec", 1, 1),
    23: ("lines/sec", 1e3, 2),
    25: ("pages/sec", 1, 1),
    26: ("routes/sec", 1, 1),
}


def _format_workload_rate(rate: float, units: int) -> str:
    if units == 2:
        if rate >= 1024**3:
            return f"{rate / 1024**3:.2f} GB/sec"
        if rate >= 1024**2:
            return f"{rate / 1024**2:.1f} MB/sec"
        if rate >= 1024:
            return f"{rate / 1024:.1f} KB/sec"
        return f"{rate:.0f} B/sec"
    if units == 9:
        if rate >= 1e9:
            return f"{rate / 1e9:.2f} Gpixels/sec"
        if rate >= 1e6:
            return f"{rate / 1e6:.1f} Mpixels/sec"
        if rate >= 1e3:
            return f"{rate / 1e3:.1f} Kpixels/sec"
        return f"{rate:.0f} pixels/sec"
    suffix, scale, precision = _UNIT_SUFFIX.get(units, ("units/sec", 1, 1))
    scaled = rate / scale
    if scale > 1:
        prefix = {1e9: "G", 1e6: "M", 1e3: "K"}.get(scale, "")
        suffix = suffix.replace("pairs/sec", f"{prefix}pairs/sec")
        suffix = suffix.replace("rows/sec", f"{prefix}rows/sec")
        suffix = suffix.replace("Elements/sec", f"{prefix}Elements/sec")
        suffix = suffix.replace("lines/sec", f"{prefix}lines/sec")
        suffix = suffix.replace("TE/sec", f"{prefix}TE/sec")
    return f"{scaled:.{precision}f} {suffix}"


def geekbench_upload_document_to_json(document: dict) -> dict:
    results = {}
    for section in document["sections"]:
        section_name = GEEKBENCH_SECTION_NAMES.get(section["name"], section["name"])
        results[section_name] = {
            "Score": {"score": int(section["score"])},
        }
        for workload in section["workloads"]:
            entry = {"score": int(workload["score"])}
            if "rate" in workload and "units" in workload:
                entry["description"] = _format_workload_rate(
                    float(workload["rate"]), int(workload["units"])
                )
            results[section_name][workload["name"]] = entry
    return results


def geekbench_upload_document_from_stderr(stderr: bytes) -> dict | None:
    text = stderr.decode("utf-8")
    start_marker, end_marker = GEEKBENCH_UPLOAD_MARKERS
    if start_marker not in text:
        return None
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return json.loads(text[start:end])


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
        results[tstring]["Score"] = {"score": _geekbench_th_score(table)}
        for bstring in GEEKBENCH_BENCHMARKS:
            results[tstring][bstring] = _geekbench_td_scores(table, bstring)
    return results

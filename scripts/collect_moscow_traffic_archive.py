"""Collect dated citywide traffic facts from the public Moscow Agency archive."""

import argparse
import csv
import hashlib
import html
import json
import re
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

VERSION = "moscow-agency-traffic.v1"
START, END = "2025-01-01", "2025-10-31"
BASE = "https://www.mskagency.ru"
WORDS = {
    "ноль": 0,
    "ноля": 0,
    "один": 1,
    "одного": 1,
    "два": 2,
    "двух": 2,
    "три": 3,
    "трех": 3,
    "четыре": 4,
    "четырех": 4,
    "пять": 5,
    "пяти": 5,
    "шесть": 6,
    "шести": 6,
    "семь": 7,
    "семи": 7,
    "восемь": 8,
    "восьми": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
}
NUMBER = r"(?:10|[0-9]|" + "|".join(WORDS) + r")"
SCORE = re.compile(r"(?<![\w-])(" + NUMBER + r")\s+балл", re.I)


def plain(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


class ArticleBody(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.in_paragraph = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "div" and (
            self.depth or dict(attrs).get("itemprop") == "articleBody"
        ):
            self.depth += 1
        if self.depth and tag == "p":
            self.in_paragraph = True

    def handle_endtag(self, tag):
        if self.depth and tag == "div":
            self.depth -= 1
        if tag == "p":
            self.in_paragraph = False
            self.parts.append(" ")

    def handle_data(self, value):
        if self.depth and self.in_paragraph:
            self.parts.append(value)


def current_citywide(text):
    return not re.search(
        r"ожида|прогноз|могут|могли|будут|будет|завтра|вчера|недел|месяц|с начала|в среднем"
        r"|в центре|на мкад|на участке|подмосков|област|почти|более|менее|не достиг|не превыш"
        r"|от .* до |утром|ранее",
        text,
    )


def parse_article(raw, url):
    """Return only a current citywide measurement; publication time is a proxy."""
    title = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.S)
    body_parser = ArticleBody()
    body_parser.feed(raw)
    body = "".join(body_parser.parts)
    stamp = re.search(r'<p class="date">\s*(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2})', raw)
    if not (title and body and stamp):
        raise ValueError(f"Unrecognized article structure: {url}")
    title, body = (
        plain(title[1]).lower().replace("ё", "е"),
        plain(body).lower().replace("ё", "е"),
    )
    published = datetime.strptime(stamp[1], "%d.%m.%Y %H:%M").replace(
        tzinfo=ZoneInfo("Europe/Moscow")
    )
    if not START <= published.date().isoformat() <= END:
        return None
    if not re.search(r"москв|столич|столиц", title) or re.search(
        r"подмосков|област", title
    ):
        return None
    if not current_citywide(title):
        return None
    if not re.search(r"пробк|загруженность|скорость", title):
        return None
    if not re.search(
        r"\b(?:достигли|достигла|достигло|оценивается|составляет|составляют|выросла|выросли|снизилась|снизились|увеличилась)\b",
        title,
    ):
        return None
    scores = {int(m) if m.isdigit() else WORDS[m] for m in SCORE.findall(title)}
    if len(scores) != 1:
        return None
    score = scores.pop()
    lead = re.split(r"(?<=[.!?])\s+", body)[0]
    if not current_citywide(lead):
        return None
    lead_scores = {int(m) if m.isdigit() else WORDS[m] for m in SCORE.findall(lead)}
    if lead_scores != {score}:
        return None
    score_source = (
        "yandex" if "яндекс" in lead else "codd" if "цодд" in lead else "unspecified"
    )
    speeds = set()
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if not current_citywide(sentence) or "в москве" not in sentence:
            continue
        if re.search(r"увелич|сниз|выше|ниже", sentence):
            continue
        match = re.search(
            r"средняя скорость (?:транспортного потока|движения)(?:[^.!?]{0,100}?)составляет (\d+(?:[.,]\d+)?)\s*км/ч",
            sentence,
        )
        if match:
            speed = float(match[1].replace(",", "."))
            if 0 < speed <= 100:
                speeds.add(speed)
    return {
        "published_at": published.isoformat(),
        "date": published.date().isoformat(),
        "hour": published.hour,
        "congestion_score": score,
        "score_provider": score_source,
        "mean_speed_kmh": next(iter(speeds)) if len(speeds) == 1 else None,
        "speed_provider": "codd"
        if len(speeds) == 1 and "центра организации дорожного движения" in body
        else "unspecified"
        if len(speeds) == 1
        else None,
        "source_url": url,
        "timestamp_kind": "publication_proxy",
        "original_vintage_verified": False,
    }


def search_rows(raw):
    return re.findall(
        r'<li[^>]*data-material_id="(\d+)"[^>]*data-datei="(\d{8})".*?<a[^>]*title="([^"]*)"[^>]*class="js-title"',
        raw,
        re.S,
    )


def collect(cache, output, offline=False):
    cache.mkdir(parents=True, exist_ok=True)
    receipts = {}

    def fetch(url):
        path = cache / (hashlib.sha256(url.encode()).hexdigest() + ".html")
        if not path.exists():
            if offline:
                raise ValueError(f"Missing cached URL: {url}")
            result = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--max-time",
                    "30",
                    url,
                ],
                capture_output=True,
                check=True,
            )
            path.write_bytes(result.stdout)
            time.sleep(0.5)
        payload = path.read_bytes()
        receipts[url] = {
            "url": url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        return payload.decode("utf-8")

    candidates = set()
    for term in ("пробки", "загруженность", "средняя скорость"):
        params = {
            "criteria": term,
            "from": "01.01.2025",
            "to": "31.10.2025",
            "type": "text",
        }
        first = fetch(BASE + "/search/?" + urlencode(params))
        pages = re.search(r'data-pages="(\d+)"', first)
        pages = int(pages[1]) if pages else 1
        if pages > 100:
            raise ValueError("Unexpected search size; review bounds before downloading")
        for page in range(1, pages + 1):
            raw = (
                first
                if page == 1
                else fetch(BASE + "/search/?" + urlencode({**params, "page": page}))
            )
            entries = search_rows(raw)
            if not entries:
                raise ValueError(f"Empty or changed search page {term}/{page}")
            for article_id, day, title in entries:
                if "20250101" <= day <= "20251031" and re.search(
                    r"балл", html.unescape(title), re.I
                ):
                    candidates.add(BASE + "/materials/" + article_id)
        print(
            f"Search {term}: {pages} pages; {len(candidates)} candidate articles",
            flush=True,
        )
    records = []
    rejected = []
    for url in sorted(candidates):
        record = parse_article(fetch(url), url)
        if record is None:
            rejected.append(url)
        else:
            records.append(record)
    if not records:
        raise ValueError("No verified measurements; not writing an empty success")
    records.sort(key=lambda r: (r["published_at"], r["source_url"]))
    output.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    ).encode()
    (output / "observations.jsonl").write_bytes(content)
    with (output / "observations.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema": VERSION,
        "feature_version": None,
        "model_status": "raw_external_data_not_integrated",
        "source": BASE,
        "publisher": "Moscow City News Agency",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "requested_period": [START, END],
        "actual_period": [records[0]["date"], records[-1]["date"]],
        "timezone": "Europe/Moscow",
        "observations": len(records),
        "days": len({r["date"] for r in records}),
        "publication_hour_buckets": len({(r["date"], r["hour"]) for r in records}),
        "requested_hour_buckets": 304 * 24,
        "by_month": dict(sorted(Counter(r["date"][:7] for r in records).items())),
        "by_score_provider": dict(Counter(r["score_provider"] for r in records)),
        "speed_observations": sum(r["mean_speed_kmh"] is not None for r in records),
        "files": {
            name: {"sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
            for name in ("observations.jsonl", "observations.csv")
        },
        "pages": list(receipts.values()),
        "rejected_candidate_urls": rejected,
        "leakage_rule": "Only facts published before forecast origin may be used; November/December actuals excluded. Original historical vintage not proven.",
        "rights": "Numerical facts and source links only; no article text redistributed. No open-data licence established.",
        "limitations": [
            "Selective news reports, biased toward congestion peaks; not a continuous traffic archive.",
            "Citywide road congestion, not tram speeds or route-specific measurements.",
            "Publication time approximates measurement time; never fill absent hours with zero.",
            "Yandex and CODD scales are separate providers; do not pool as the same measurement.",
            "Different publishers may repeat one underlying observation; source URL uniqueness is not sensor independence.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in manifest.items()
                if k not in ("pages", "rejected_candidate_urls")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    collect(args.cache, args.output, args.offline)

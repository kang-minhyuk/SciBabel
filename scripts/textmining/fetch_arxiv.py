from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import requests

from common import append_jsonl, seen_ids

ARXIV_API = "http://export.arxiv.org/api/query"


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def matches_window(published: str, start_date: str, end_date: str) -> bool:
    pub = parse_dt(published)
    if pub is None:
        return True
    sd = parse_dt(start_date) if start_date else None
    ed = parse_dt(end_date) if end_date else None
    if sd and pub < sd:
        return False
    if ed and pub > ed:
        return False
    return True


def build_query(categories: list[str], search_query: str) -> str:
    if search_query.strip():
        return search_query.strip()
    if not categories:
        raise ValueError("Provide --category or --search-query")
    return " OR ".join([f"cat:{c.strip()}" for c in categories if c.strip()])


def fetch_page(query: str, start: int, max_results: int) -> list[dict[str, Any]]:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=30)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)
    rows: list[dict[str, Any]] = []
    for entry in feed.entries:
        rows.append(
            {
                "id": entry.get("id", ""),
                "title": (entry.get("title", "") or "").replace("\n", " ").strip(),
                "abstract": (entry.get("summary", "") or "").replace("\n", " ").strip(),
                "categories": [t.get("term", "") for t in entry.get("tags", []) if t.get("term")],
                "published": entry.get("published", ""),
                "updated": entry.get("updated", ""),
                "url": entry.get("id", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv metadata/abstracts to JSONL")
    parser.add_argument("--domain", default="", help="Domain label, e.g. CSM")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--search-query", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep-sec", type=float, default=1.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    query = build_query(args.category, args.search_query)
    existing_ids = seen_ids(out_path)

    kept = 0
    start = 0
    while kept < args.max_results:
        remaining = args.max_results - kept
        page_size = min(args.batch_size, remaining)
        page = fetch_page(query=query, start=start, max_results=page_size)
        if not page:
            break

        to_write: list[dict[str, Any]] = []
        for row in page:
            rid = str(row.get("id", "")).strip()
            if not rid or rid in existing_ids:
                continue
            if not matches_window(str(row.get("published", "")), args.start_date, args.end_date):
                continue
            row["source"] = "arxiv"
            if args.domain:
                row["domain"] = args.domain
            to_write.append(row)
            existing_ids.add(rid)

        if to_write:
            append_jsonl(out_path, to_write)
            kept += len(to_write)

        start += len(page)
        if len(page) < page_size:
            break
        time.sleep(max(0.0, args.sleep_sec))

    print(f"query={query}")
    print(f"wrote={kept} out={out_path}")


if __name__ == "__main__":
    main()

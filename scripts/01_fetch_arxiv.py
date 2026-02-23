from __future__ import annotations

import argparse
import json
from pathlib import Path

import feedparser

ARXIV_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv(query: str, max_results: int) -> list[dict[str, str]]:
    url = (
        f"{ARXIV_URL}?search_query={query}&start=0"
        f"&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    )
    feed = feedparser.parse(url)

    records: list[dict[str, str]] = []
    for entry in feed.entries:
        records.append(
            {
                "id": entry.get("id", ""),
                "title": (entry.get("title", "") or "").replace("\n", " ").strip(),
                "abstract": (entry.get("summary", "") or "").replace("\n", " ").strip(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv abstracts into JSONL")
    parser.add_argument("--query", required=True, help="arXiv API query, e.g. cat:cs.LG")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--domain", default="", help="Optional domain label (CSM/PM/CCE)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    rows = fetch_arxiv(args.query, args.max_results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            row["query"] = args.query
            if args.domain:
                row["domain"] = args.domain
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} records to {out_path}")


if __name__ == "__main__":
    main()

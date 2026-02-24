from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import requests

from common import append_jsonl, seen_ids
from diagnose_chemrxiv import run_diagnostics

API_URL = "http://chemrxiv.org/engage/chemrxiv/public-api/v1/items"


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("itemHits")
    if isinstance(hits, list):
        return hits
    return []


def request_json(
    session: requests.Session,
    params: dict[str, Any],
    timeout: int,
    max_retries: int,
    backoff_sec: float,
    verbose: bool,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(API_URL, params=params, timeout=timeout)
            if verbose:
                print(f"GET {resp.url} -> {resp.status_code}")
            if resp.status_code == 403:
                raise RuntimeError(
                    "ChemRxiv API returned 403 (likely blocked or deprecated). "
                    "Run diagnostics via: python scripts/textmining/diagnose_chemrxiv.py"
                )
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(f"retryable_status:{resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(backoff_sec * (2**attempt))
    assert last_exc is not None
    raise last_exc


def normalize_categories(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    categories = item.get("categories")
    if isinstance(categories, list):
        for c in categories:
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, dict):
                name = c.get("name") or c.get("label") or c.get("title")
                if name:
                    out.append(str(name))
    return out


def item_to_row(item: dict[str, Any], domain: str) -> dict[str, Any]:
    cats = normalize_categories(item)
    rid = str(item.get("id") or item.get("itemId") or item.get("slug") or item.get("doi") or "")
    title = str(item.get("title") or "").strip()
    abstract = str(item.get("abstract") or item.get("description") or "").strip()
    pub = str(item.get("publishedDate") or item.get("published") or item.get("createdDate") or "")
    doi = str(item.get("doi") or "")
    lic = str(item.get("license") or "")
    url = str(item.get("url") or item.get("landingPage") or "")

    return {
        "id": rid,
        "source": "chemrxiv",
        "domain": domain,
        "title": title,
        "abstract": abstract,
        "categories": cats,
        "published": pub,
        "doi": doi,
        "license": lic,
        "url": url,
    }


def category_match(item_categories: list[str], filters: list[str]) -> bool:
    if not filters:
        return True
    all_cat = " | ".join(item_categories).lower()
    return any(f.lower() in all_cat for f in filters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ChemRxiv metadata/abstracts to JSONL")
    parser.add_argument("--domain", default="CCE")
    parser.add_argument("--max-results", type=int, default=5000)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--sleep-sec", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--category-filter", action="append", default=[])
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-sec", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default="")
    parser.add_argument("--diagnose", action="store_true", help="Run ChemRxiv diagnostics and exit")
    parser.add_argument("--diagnose-out", default="", help="Optional path for diagnosis markdown output")
    args = parser.parse_args()

    if args.diagnose:
        jsonl_path, md_path, conclusion = run_diagnostics(out_md=args.diagnose_out or None, timeout=args.timeout)
        print(f"ChemRxiv diagnosis complete: {md_path}")
        print(f"Raw request metadata: {jsonl_path}")
        print(f"Conclusion: {conclusion}")
        return

    if not args.out:
        raise SystemExit("--out is required unless --diagnose is used")

    out_path = Path(args.out)
    existing = seen_ids(out_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "http://chemrxiv.org/engage/chemrxiv/public-dashboard",
            "Origin": "http://chemrxiv.org",
            "Connection": "keep-alive",
        }
    )

    written = 0
    offset = 0
    page_size = max(1, int(args.page_size))
    while written < args.max_results:
        params = {"limit": page_size, "skip": offset}

        payload = request_json(
            session=session,
            params=params,
            timeout=args.timeout,
            max_retries=args.max_retries,
            backoff_sec=args.backoff_sec,
            verbose=args.verbose,
        )
        items = extract_items(payload)
        if not items:
            break

        rows: list[dict[str, Any]] = []
        for item in items:
            row = item_to_row(item, domain=args.domain)
            rid = row["id"]
            if not rid or rid in existing:
                continue
            if not category_match(row.get("categories", []), args.category_filter):
                continue
            rows.append(row)
            existing.add(rid)

        if rows:
            append_jsonl(out_path, rows)
            written += len(rows)

        offset += page_size
        if len(items) < page_size:
            break
        time.sleep(max(0.0, args.sleep_sec))

    print(f"pagination=skip/limit page_size={page_size}")
    print(f"wrote={written} out={out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import requests

from common import append_jsonl, seen_ids

API_URL = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
PAGINATION_CANDIDATES = [
    ("offset", "limit"),
    ("skip", "limit"),
    ("page", "size"),
]


def extract_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits = payload.get("itemHits")
    if isinstance(hits, list):
        return hits
    return []


def first_item_key(items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    it = items[0]
    return str(it.get("id") or it.get("itemId") or it.get("doi") or it.get("slug") or "")


def probe_pagination(session: requests.Session, base_params: dict[str, Any], timeout: int) -> tuple[str, str, int]:
    resp = session.get(API_URL, params=base_params, timeout=timeout)
    resp.raise_for_status()
    base_payload = resp.json()
    base_items = extract_items(base_payload)
    if not base_items:
        raise RuntimeError("ChemRxiv returned no items on first request.")

    base_first = first_item_key(base_items)
    page_size = len(base_items)

    for a, b in PAGINATION_CANDIDATES:
        p2 = dict(base_params)
        if (a, b) == ("page", "size"):
            p2[a] = 2
            p2[b] = page_size
        else:
            p2[a] = page_size
            p2[b] = page_size

        r2 = session.get(API_URL, params=p2, timeout=timeout)
        if r2.status_code != 200:
            continue
        items2 = extract_items(r2.json())
        if not items2:
            continue
        second_first = first_item_key(items2)
        if second_first and second_first != base_first:
            return a, b, page_size

    raise RuntimeError("Unable to autodetect ChemRxiv pagination parameters.")


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
    parser.add_argument("--sleep-sec", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--category-filter", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    existing = seen_ids(out_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "SciBabel-TextMining/1.0 (+https://github.com/kang-minhyuk/SciBabel)",
            "Accept": "application/json",
        }
    )
    base_params: dict[str, Any] = {}

    try:
        p_a, p_b, page_size = probe_pagination(session, base_params, args.timeout)
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 403:
            print("WARN: ChemRxiv API returned 403 Forbidden. Skipping ChemRxiv fetch in this run.")
            print(f"wrote=0 out={out_path}")
            return
        raise
    except RuntimeError as exc:
        raise RuntimeError(f"ChemRxiv pagination detection failed: {exc}")

    written = 0
    page_idx = 0
    while written < args.max_results:
        params = dict(base_params)
        if (p_a, p_b) == ("page", "size"):
            params[p_a] = page_idx + 1
            params[p_b] = page_size
        else:
            params[p_a] = page_idx * page_size
            params[p_b] = page_size

        resp = session.get(API_URL, params=params, timeout=args.timeout)
        resp.raise_for_status()
        payload = resp.json()
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

        page_idx += 1
        if len(items) < page_size:
            break
        time.sleep(max(0.0, args.sleep_sec))

    print(f"pagination={p_a}/{p_b} page_size={page_size}")
    print(f"wrote={written} out={out_path}")


if __name__ == "__main__":
    main()

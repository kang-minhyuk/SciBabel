from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from common import append_jsonl, seen_ids

API_URL = "https://api.openalex.org/works"
CHEMISTRY_CONCEPT_ID = "C178790620"
CHEMICAL_ENGINEERING_CONCEPT_ID = "C185592680"

CHEME_KEYWORDS = [
    "reactor",
    "separation",
    "distillation",
    "membrane",
    "adsorption",
    "catalysis",
    "mass transfer",
    "transport",
    "process control",
    "reaction engineering",
    "heat transfer",
]


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _log(msg: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.utcnow().isoformat()}Z] {msg}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def rebuild_abstract(inv: dict[str, list[int]] | None) -> str:
    if not isinstance(inv, dict):
        return ""
    positions: dict[int, str] = {}
    for token, idxs in inv.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions[i] = token
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys())).strip()


def make_filter(concept_ids: list[str], from_year: int, display_name: str = "") -> str:
    filters = [f"from_publication_date:{from_year}-01-01", "has_abstract:true"]
    if display_name:
        filters.append(f"concepts.display_name:{display_name}")
    elif concept_ids:
        filters.append("concepts.id:" + "|".join(concept_ids))
    return ",".join(filters)


def _concept_pairs(item: dict[str, Any]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for c in (item.get("concepts") or []):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "")
        score = float(c.get("score") or 0.0)
        out.append((cid, score))
    return out


def concept_score(item: dict[str, Any], concept_id: str) -> float:
    scores = [score for cid, score in _concept_pairs(item) if cid.endswith(concept_id)]
    return max(scores) if scores else 0.0


def cheme_keyword_match(title: str, abstract: str, concepts: list[str]) -> bool:
    text = f"{title} {abstract} {' '.join(concepts)}".lower()
    return any(k in text for k in CHEME_KEYWORDS)


def classify_domain(item: dict[str, Any]) -> tuple[str, bool]:
    title = str(item.get("display_name") or "").strip()
    abstract = rebuild_abstract(item.get("abstract_inverted_index"))
    concepts = [
        str(c.get("display_name"))
        for c in (item.get("concepts") or [])
        if isinstance(c, dict) and c.get("display_name")
    ]
    kw = cheme_keyword_match(title, abstract, concepts)
    chem_score = concept_score(item, CHEMISTRY_CONCEPT_ID)
    cheme_score = concept_score(item, CHEMICAL_ENGINEERING_CONCEPT_ID)

    if kw:
        return "CHEME", True
    if cheme_score >= chem_score and cheme_score > 0:
        return "CHEME", False
    return "CHEM", False


def item_to_row(item: dict[str, Any], domain: str) -> dict[str, Any]:
    rid = str(item.get("id", "")).strip()
    title = str(item.get("display_name") or "").strip()
    abstract = rebuild_abstract(item.get("abstract_inverted_index"))
    concepts = item.get("concepts") or []
    cats = [str(c.get("display_name")) for c in concepts if isinstance(c, dict) and c.get("display_name")]
    year = item.get("publication_year")
    year_val = int(year) if isinstance(year, int) else 0
    return {
        "id": rid,
        "source": "openalex",
        "domain": domain,
        "title": title,
        "abstract": abstract,
        "year": year_val,
        "concepts": cats,
        "categories": cats,
        "published": str(item.get("publication_date") or ""),
    }


def keyword_match(row: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        return True
    concepts = row.get("concepts", row.get("categories", []))
    text = f"{row.get('title','')} {row.get('abstract','')} {' '.join(concepts)}".lower()
    return any(k.lower() in text for k in keywords)


def load_existing_map(out_path: Path, other_out: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for rid in seen_ids(out_path):
        mapping[rid] = "self"
    if other_out and other_out.exists():
        for rid in seen_ids(other_out):
            mapping[rid] = "other"
    return mapping


def remove_ids_from_jsonl(path: Path, ids_to_drop: set[str]) -> int:
    if not path.exists() or not ids_to_drop:
        return 0
    kept: list[dict[str, Any]] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("id") or "")
            if rid in ids_to_drop:
                dropped += 1
                continue
            kept.append(row)
    with path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenAlex works for CHEM/CHEME corpus expansion")
    parser.add_argument("--domain", default="CHEM")
    parser.add_argument("--concept-id", action="append", default=[])
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--display-name", default="")
    parser.add_argument("--from-year", type=int, default=2014)
    parser.add_argument("--max-results", type=int, default=10000)
    parser.add_argument("--target-doc-count", type=int, default=0)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-sec", type=float, default=0.3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--other-out", default="", help="Path to sibling domain JSONL for dedup coordination")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    other_out = Path(args.other_out) if args.other_out else None
    existing = load_existing_map(out_path, other_out)
    target = int(args.target_doc_count or args.max_results)
    log_path = (
        Path(args.log_file)
        if args.log_file
        else Path("logs") / f"textmining_fetch_{now_stamp()}.log"
    )
    moved_from_other: set[str] = set()

    domain = str(args.domain).upper()
    if domain not in {"CHEM", "CHEME"}:
        raise ValueError("--domain must be CHEM or CHEME")

    display_name = args.display_name.strip() or ("Chemistry" if domain == "CHEM" else "Chemical Engineering")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "SciBabel-TextMining/1.0 (mailto:maintainer@example.com)",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    params: dict[str, Any] = {
        "filter": make_filter(args.concept_id, args.from_year, display_name=display_name),
        "per-page": min(max(1, args.per_page), 200),
        "sort": "publication_date:desc",
    }

    total = len([k for k, v in existing.items() if v == "self"])
    written = 0
    cursor = "*"
    _log(
        f"openalex_fetch_start domain={domain} target={target} existing={total} out={out_path} display_name={display_name}",
        log_path,
    )

    while total < target:
        params["cursor"] = cursor
        resp = session.get(API_URL, params=params, timeout=args.timeout)
        if args.verbose:
            print(f"GET {resp.url} -> {resp.status_code}")
        if resp.status_code == 400 and "concepts.display_name" in params.get("filter", "") and args.concept_id:
            params["filter"] = make_filter(args.concept_id, args.from_year, display_name="")
            _log("openalex_filter_fallback=concepts.id", log_path)
            continue
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("results") or []
        if not items:
            break

        rows: list[dict[str, Any]] = []
        for item in items:
            assigned_domain, kw_match = classify_domain(item)
            if assigned_domain != domain:
                continue

            row = item_to_row(item, assigned_domain)
            rid = row["id"]
            if not rid:
                continue
            if rid in existing and existing[rid] == "self":
                continue
            if len(str(row.get("abstract", ""))) < 200:
                continue
            if domain == "CHEME" and not kw_match:
                if not keyword_match(row, CHEME_KEYWORDS):
                    continue
            if not keyword_match(row, args.keyword):
                continue

            if rid in existing and existing[rid] == "other" and domain == "CHEME" and other_out is not None:
                moved_from_other.add(rid)

            rows.append(row)
            existing[rid] = "self"
            written += 1
            total += 1
            if total >= target:
                break

        if rows:
            append_jsonl(out_path, rows)
            if total % 1000 == 0 or written % 1000 == 0:
                _log(
                    f"openalex_progress domain={domain} total={total} new={written} cursor={cursor}",
                    log_path,
                )

        cursor = str(payload.get("meta", {}).get("next_cursor") or "")
        if not cursor:
            break
        time.sleep(max(0.0, args.sleep_sec))

    moved = 0
    if domain == "CHEME" and other_out is not None and moved_from_other:
        moved = remove_ids_from_jsonl(other_out, moved_from_other)
        _log(f"openalex_dedup_moved_to_CHEME={moved}", log_path)

    _log(
        f"openalex_fetch_done domain={domain} total={total} new={written} moved_from_other={moved} out={out_path}",
        log_path,
    )
    print(f"total={total} newly_written={written} out={out_path}")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests

from common import append_jsonl, read_jsonl, seen_ids

ARXIV_API = "http://export.arxiv.org/api/query"
RETRYABLE_STATUS = {429, 500, 503}


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _log(msg: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.utcnow().isoformat()}Z] {msg}"
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    return low in {"1", "true", "yes", "y", "on"}


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def arxiv_end_dt_str(dt: datetime) -> str:
    dt_utc = _to_utc(dt)
    return dt_utc.strftime("%Y%m%d2359")


def next_slice_end_date(current_end: datetime, slice_days: int) -> datetime:
    return _to_utc(current_end) - timedelta(days=max(1, int(slice_days)))


def build_category_query(category: str, end_date: datetime) -> str:
    return f"cat:{category} AND submittedDate:[190001010000 TO {arxiv_end_dt_str(end_date)}]"


def build_query(search_query: str) -> str:
    if not search_query.strip():
        raise ValueError("Query mode requires --search-query")
    return search_query.strip()


def fetch_page(session: requests.Session, query: str, start: int, max_results: int, timeout: int = 30) -> tuple[list[dict[str, Any]], requests.Response]:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = session.get(ARXIV_API, params=params, timeout=timeout)
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
    return rows, resp


def fetch_page_with_retry(
    *,
    session: requests.Session,
    query: str,
    start: int,
    max_results: int,
    max_retries: int,
    backoff_base: float,
    log_path: Path,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            rows, resp = fetch_page(session=session, query=query, start=start, max_results=max_results, timeout=timeout)
            _log(
                f"arxiv_page_ok attempt={attempt} status={resp.status_code} start={start} returned={len(rows)} url={resp.url}",
                log_path,
            )
            return rows
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            url = exc.response.url if exc.response is not None else ARXIV_API
            last_exc = exc
            _log(f"arxiv_retry attempt={attempt} status={status} start={start} url={url}", log_path)
            if attempt >= max_retries or status not in RETRYABLE_STATUS:
                break
            sleep_for = min(60.0, float(backoff_base) ** max(1, attempt + 1))
            time.sleep(sleep_for)
        except Exception as exc:
            last_exc = exc
            _log(f"arxiv_retry attempt={attempt} status=exception start={start} err={exc}", log_path)
            if attempt >= max_retries:
                break
            sleep_for = min(60.0, float(backoff_base) ** max(1, attempt + 1))
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def _normalize_categories(args: argparse.Namespace) -> list[str]:
    cats: list[str] = []
    cats.extend([str(c).strip() for c in (args.categories or []) if str(c).strip()])
    cats.extend([str(c).strip() for c in (args.category or []) if str(c).strip()])
    seen: set[str] = set()
    out: list[str] = []
    for c in cats:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def category_file(out_dir: Path, domain: str, category: str) -> Path:
    safe = category.replace(".", "_")
    return out_dir / f"{domain.lower()}_{safe}.jsonl"


def merge_dedup_jsonl(inputs: list[Path], merge_out: Path, domain: str) -> tuple[int, int]:
    merge_out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    unique = 0
    dup = 0
    with merge_out.open("w", encoding="utf-8") as w:
        for path in inputs:
            if not path.exists():
                continue
            for row in read_jsonl(path):
                rid = str(row.get("id", "")).strip()
                if not rid:
                    continue
                if rid in seen:
                    dup += 1
                    continue
                seen.add(rid)
                out = {
                    "id": rid,
                    "title": str(row.get("title", "")),
                    "abstract": str(row.get("abstract", "")),
                    "published": str(row.get("published", "")),
                    "updated": str(row.get("updated", "")),
                    "categories": row.get("categories", []),
                    "url": str(row.get("url", "")),
                    "domain": domain,
                    "source": "arxiv",
                }
                w.write(json.dumps(out, ensure_ascii=False) + "\n")
                unique += 1
    return unique, dup


def harvest_category(
    *,
    session: requests.Session,
    domain: str,
    category: str,
    out_path: Path,
    per_target: int,
    max_results_per_request: int,
    max_retries: int,
    backoff_base: float,
    sleep_sec: float,
    time_slice_days: int,
    max_slices: int,
    resume: bool,
    initial_end_date: datetime,
    log_path: Path,
) -> tuple[int, int]:
    if not resume and out_path.exists():
        out_path.unlink()

    cat_seen = seen_ids(out_path) if resume else set()
    total = len(cat_seen)
    new_written = 0
    end_date = _to_utc(initial_end_date)
    slices_used = 0

    _log(
        f"arxiv_fetch_start domain={domain} category={category} target={per_target} existing={total} out={out_path}",
        log_path,
    )

    while total < per_target and slices_used < max_slices:
        query = build_category_query(category, end_date)
        _log(
            f"arxiv_slice_start domain={domain} category={category} slice={slices_used + 1} end_date={arxiv_end_dt_str(end_date)}",
            log_path,
        )

        start = 0
        pages_without_new = 0
        saw_any_rows = False

        while total < per_target:
            try:
                rows = fetch_page_with_retry(
                    session=session,
                    query=query,
                    start=start,
                    max_results=max_results_per_request,
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                    log_path=log_path,
                )
            except Exception as exc:
                _log(
                    f"arxiv_stagnation domain={domain} category={category} slice={slices_used + 1} reason=retry_exhausted err={exc}",
                    log_path,
                )
                break
            if not rows:
                break
            saw_any_rows = True

            to_write: list[dict[str, Any]] = []
            for row in rows:
                rid = str(row.get("id", "")).strip()
                if not rid or rid in cat_seen:
                    continue
                row["source"] = "arxiv"
                row["domain"] = domain
                to_write.append(row)
                cat_seen.add(rid)
                total += 1
                new_written += 1
                if total >= per_target:
                    break

            if to_write:
                append_jsonl(out_path, to_write)
                pages_without_new = 0
                if total % 1000 == 0 or new_written % 1000 == 0:
                    _log(
                        f"arxiv_progress domain={domain} category={category} total={total} new={new_written} slice={slices_used + 1}",
                        log_path,
                    )
            else:
                pages_without_new += 1
                if pages_without_new >= 10:
                    _log(
                        f"arxiv_stagnation domain={domain} category={category} slice={slices_used + 1} pages_without_new={pages_without_new}",
                        log_path,
                    )
                    break

            if len(rows) < max_results_per_request:
                break

            start += len(rows)
            time.sleep(max(0.0, sleep_sec))

        slices_used += 1
        if not saw_any_rows:
            break

        prev_end = end_date
        end_date = next_slice_end_date(end_date, time_slice_days)
        _log(
            f"arxiv_rollover domain={domain} category={category} from={arxiv_end_dt_str(prev_end)} to={arxiv_end_dt_str(end_date)}",
            log_path,
        )

    _log(
        f"arxiv_fetch_done domain={domain} category={category} total={total} new={new_written} slices_used={slices_used}",
        log_path,
    )
    return total, slices_used


def harvest_query_mode(
    *,
    session: requests.Session,
    domain: str,
    search_query: str,
    out_path: Path,
    target_doc_count: int,
    max_results_per_request: int,
    max_retries: int,
    backoff_base: float,
    sleep_sec: float,
    resume: bool,
    log_path: Path,
) -> tuple[int, int]:
    if not resume and out_path.exists():
        out_path.unlink()
    existing_ids = seen_ids(out_path) if resume else set()
    total = len(existing_ids)
    new_written = 0
    start = 0
    query = build_query(search_query)
    _log(f"arxiv_fetch_start domain={domain} mode=query target={target_doc_count} existing={total}", log_path)
    while total < target_doc_count:
        rows = fetch_page_with_retry(
            session=session,
            query=query,
            start=start,
            max_results=max_results_per_request,
            max_retries=max_retries,
            backoff_base=backoff_base,
            log_path=log_path,
        )
        if not rows:
            break
        to_write: list[dict[str, Any]] = []
        for row in rows:
            rid = str(row.get("id", "")).strip()
            if not rid or rid in existing_ids:
                continue
            row["source"] = "arxiv"
            row["domain"] = domain
            to_write.append(row)
            existing_ids.add(rid)
            total += 1
            new_written += 1
            if total >= target_doc_count:
                break
        if to_write:
            append_jsonl(out_path, to_write)
        if len(rows) < max_results_per_request:
            break
        start += len(rows)
        time.sleep(max(0.0, sleep_sec))
    _log(f"arxiv_fetch_done domain={domain} mode=query total={total} new={new_written}", log_path)
    return total, 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv metadata/abstracts to JSONL (robust large-corpus harvester)")
    parser.add_argument("--domain", default="", help="Domain label, e.g. CSM")
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--category-mode", default="true")
    parser.add_argument("--search-query", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--max-results", type=int, default=2000)
    parser.add_argument("--target-doc-count", type=int, default=15000)
    parser.add_argument("--per-category-target", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=3.0)
    parser.add_argument("--max-results-per-request", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--backoff-base", type=float, default=2.0)
    parser.add_argument("--time-slice-days", type=int, default=365)
    parser.add_argument("--max-slices", type=int, default=40)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--merge-out", default="")
    parser.add_argument("--out-dir", default="data/raw/arxiv")
    parser.add_argument("--resume", default="true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    categories = _normalize_categories(args)
    category_mode = _bool_arg(args.category_mode)
    if categories:
        category_mode = True

    if args.sleep_sec < 2.0:
        print("WARNING: sleep-sec < 2.0 is impolite for arXiv. Recommended >= 3.0")

    merge_out = Path(args.merge_out or args.out)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = _bool_arg(args.resume)
    max_results_per_request = 200 if int(args.max_results_per_request) != 200 else int(args.max_results_per_request)

    target = int(args.target_doc_count or args.max_results)
    if target <= 0:
        target = int(args.max_results)

    log_path = (
        Path(args.log_file)
        if args.log_file
        else Path("logs") / f"textmining_fetch_{now_stamp()}.log"
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "SciBabel-arXivHarvester/1.0 (mailto:maintainer@example.com)",
            "Accept": "application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        }
    )

    per_category_counts: dict[str, int] = {}
    slices_used_map: dict[str, int] = {}
    newly_written = 0

    if category_mode:
        if not categories:
            raise ValueError("Category mode requires --categories (or repeatable --category)")

        n_cat = max(1, len(categories))
        per_target = int(args.per_category_target) if int(args.per_category_target) > 0 else (target + n_cat - 1) // n_cat
        end_dt = parse_dt(args.end_date) if args.end_date else datetime.now(timezone.utc)
        if end_dt is None:
            end_dt = datetime.now(timezone.utc)

        category_files: list[Path] = []
        for cat in categories:
            cat_file = category_file(out_dir, args.domain or "domain", cat)
            category_files.append(cat_file)
            total_cat, slices_used = harvest_category(
                session=session,
                domain=args.domain or "",
                category=cat,
                out_path=cat_file,
                per_target=per_target,
                max_results_per_request=max_results_per_request,
                max_retries=args.max_retries,
                backoff_base=args.backoff_base,
                sleep_sec=args.sleep_sec,
                time_slice_days=args.time_slice_days,
                max_slices=args.max_slices,
                resume=resume,
                initial_end_date=end_dt,
                log_path=log_path,
            )
            per_category_counts[cat] = total_cat
            slices_used_map[cat] = slices_used

        unique_total, dup_removed = merge_dedup_jsonl(category_files, merge_out, args.domain or "")
        newly_written = unique_total
        print(f"docs_per_category={per_category_counts}")
        print(f"total_unique={unique_total} duplicates_removed={dup_removed} merge_out={merge_out}")
    else:
        total_query, slices_used = harvest_query_mode(
            session=session,
            domain=args.domain or "",
            search_query=args.search_query,
            out_path=merge_out,
            target_doc_count=target,
            max_results_per_request=max_results_per_request,
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            sleep_sec=args.sleep_sec,
            resume=resume,
            log_path=log_path,
        )
        per_category_counts = {"query": total_query}
        slices_used_map = {"query": slices_used}
        newly_written = total_query

    _log(
        f"arxiv_fetch_done total_unique={newly_written} per_category_counts={per_category_counts} time_slices_used={slices_used_map}",
        log_path,
    )
    print(f"total_unique={newly_written} newly_written={newly_written} per_category_counts={per_category_counts} time_slices_used={slices_used_map}")
    print(f"log={log_path}")


if __name__ == "__main__":
    main()

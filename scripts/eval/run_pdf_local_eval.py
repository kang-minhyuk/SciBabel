from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

FRAGMENT_LIKE_PATTERNS = [
    re.compile(r"^(optimize|train)\s+a\s+\w+", re.IGNORECASE),
    re.compile(r"\b(is\s+characterized|characterized\s+by)\b", re.IGNORECASE),
    re.compile(r"\bmemory\s+cost\s+on\b", re.IGNORECASE),
]


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _pdfs_from_args(pdf: str | None, pdf_dir: str | None) -> list[Path]:
    out: list[Path] = []
    if pdf:
        p = Path(pdf)
        if p.exists() and p.is_file():
            out.append(p)
    if pdf_dir:
        d = Path(pdf_dir)
        if d.exists() and d.is_dir():
            out.extend(sorted(x for x in d.glob("*.pdf") if x.is_file()))
    return out


def _term_fragment_like(term: str) -> bool:
    t = term.strip().lower()
    if len(t) <= 3:
        return True
    return any(p.search(t) is not None for p in FRAGMENT_LIKE_PATTERNS)


def _collect_page_lines(text: str) -> tuple[str, str]:
    # Extract first/last rough line-like snippets from cleaned text.
    toks = [x.strip() for x in re.split(r"\s{2,}", text) if x.strip()]
    if not toks:
        return "", ""
    first = toks[0][:80]
    last = toks[-1][:80]
    return first, last


def _annotate_pdf(api_base: str, pdf_path: Path, src: str, tgt: str, timeout: float) -> tuple[int, dict[str, Any], float]:
    with pdf_path.open("rb") as f:
        files = {"file": (pdf_path.name, f.read(), "application/pdf")}
    data = {"src": src, "tgt": tgt, "audience_level": "grad", "max_terms": "8"}
    t0 = time.perf_counter()
    r = requests.post(f"{api_base}/pdf/annotate", files=files, data=data, timeout=timeout)
    latency = time.perf_counter() - t0
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:4000]}
    return int(r.status_code), body if isinstance(body, dict) else {"value": body}, latency


def _explain_top_terms(api_base: str, doc: dict[str, Any], tgt: str, top_k: int, timeout: float) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    picks: list[tuple[int, dict[str, Any]]] = []
    for page in doc.get("pages", []):
        page_num = int(page.get("page_num", 0))
        terms = page.get("terms", []) if isinstance(page.get("terms", []), list) else []
        for t in terms:
            if isinstance(t, dict) and bool(t.get("flagged", False)):
                picks.append((page_num, t))

    out: list[dict[str, Any]] = []
    for page_num, term in picks[:top_k]:
        payload = {
            "document_id": doc.get("document_id"),
            "page_num": page_num,
            "term_id": term.get("term_id"),
            "tgt": tgt,
            "detail": "short",
        }
        t0 = time.perf_counter()
        r = requests.post(f"{api_base}/pdf/explain", json=payload, timeout=timeout)
        latency = time.perf_counter() - t0
        try:
            body = r.json()
        except Exception:
            body = {"raw": r.text[:1000]}
        out.append(
            {
                "page_num": page_num,
                "term": term.get("term"),
                "term_id": term.get("term_id"),
                "status": int(r.status_code),
                "latency_sec": round(latency, 4),
                "short_explanation": str(body.get("short_explanation", "")) if isinstance(body, dict) else "",
            }
        )
    return out


def _qa_indicators(doc: dict[str, Any]) -> dict[str, Any]:
    pages = doc.get("pages", []) if isinstance(doc.get("pages", []), list) else []
    all_terms: list[dict[str, Any]] = []
    duplicate_keys: set[tuple[int, int, int, str]] = set()
    seen_keys: set[tuple[int, int, int, str]] = set()

    pages_without_text_count = 0
    header_counter: dict[str, int] = {}

    for p in pages:
        page_num = int(p.get("page_num", 0))
        text = str(p.get("text", ""))
        if not text.strip():
            pages_without_text_count += 1
        first_line, _ = _collect_page_lines(text)
        if first_line:
            header_counter[first_line] = header_counter.get(first_line, 0) + 1

        terms = p.get("terms", []) if isinstance(p.get("terms", []), list) else []
        for t in terms:
            if not isinstance(t, dict):
                continue
            all_terms.append(t)
            key = (page_num, int(t.get("start", -1)), int(t.get("end", -1)), str(t.get("term", "")).lower())
            if key in seen_keys:
                duplicate_keys.add(key)
            seen_keys.add(key)

    flagged = [t for t in all_terms if bool(t.get("flagged", False))]
    suspicious_short = sum(1 for t in flagged if len(str(t.get("term", "")).strip()) <= 6)
    fragment_like = sum(1 for t in flagged if _term_fragment_like(str(t.get("term", ""))))
    empty_evidence = sum(1 for t in flagged if not isinstance(t.get("evidence", []), list) or len(t.get("evidence", [])) == 0)
    empty_analog = sum(1 for t in flagged if not isinstance(t.get("analogs", []), list) or len(t.get("analogs", [])) == 0)

    repeated_header_like = sum(1 for _, c in header_counter.items() if c >= 2)

    return {
        "pages_without_text_count": pages_without_text_count,
        "duplicate_span_count": len(duplicate_keys),
        "suspicious_short_flag_count": suspicious_short,
        "fragment_like_term_count": fragment_like,
        "repeated_header_like_line_count": repeated_header_like,
        "empty_evidence_count": empty_evidence,
        "empty_analog_count": empty_analog,
    }


def _render_markdown(rows: list[dict[str, Any]], path: Path, api_base: str) -> None:
    lines = [
        "# Local PDF Evaluation Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- API base: {api_base}",
        f"- PDFs evaluated: {len(rows)}",
        "",
        "## Results",
        "",
        "| PDF | status | pages | src(conf) | ambiguous | pages_with_terms | flagged | suspicious_short | fragment_like | empty_pages |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        lines.append(
            f"| {r.get('filename','')} | {r.get('status',0)} | {r.get('page_count',0)} | {r.get('predicted_src','-')} ({float(r.get('predicted_src_confidence',0.0)):.3f}) | {r.get('is_ambiguous',False)} | {r.get('pages_with_terms',0)} | {r.get('flagged_term_count',0)} | {r.get('qa',{}).get('suspicious_short_flag_count',0)} | {r.get('qa',{}).get('fragment_like_term_count',0)} | {r.get('qa',{}).get('pages_without_text_count',0)} |"
        )

    lines.extend(["", "## Per-PDF Details", ""])
    for r in rows:
        lines.append(f"### {r.get('filename','unknown.pdf')}")
        lines.append(f"- status: {r.get('status',0)}")
        lines.append(f"- page_count: {r.get('page_count',0)}")
        lines.append(f"- predicted_src: {r.get('predicted_src','-')} ({float(r.get('predicted_src_confidence',0.0)):.3f})")
        lines.append(f"- is_ambiguous: {r.get('is_ambiguous',False)}")
        lines.append(f"- top_flagged_terms: {', '.join(r.get('top_flagged_terms', []))}")
        lines.append(f"- qa_indicators: `{json.dumps(r.get('qa', {}), ensure_ascii=False)}`")
        if r.get("explanations"):
            lines.append("- explanation_samples:")
            for ex in r["explanations"]:
                lines.append(
                    f"  - page {ex.get('page_num')}, term `{ex.get('term','')}`: status={ex.get('status')} latency={ex.get('latency_sec')}s"
                )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local backend-only PDF eval against /pdf/annotate")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--pdf", default="")
    ap.add_argument("--pdf-dir", default="")
    ap.add_argument("--source", default="auto")
    ap.add_argument("--target", default="PM")
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--explain-top-k", type=int, default=0)
    ap.add_argument("--out-dir", default="reports/pdf_eval")
    args = ap.parse_args()

    pdfs = _pdfs_from_args(args.pdf or None, args.pdf_dir or None)
    if not pdfs:
        raise SystemExit("No PDFs found. Provide --pdf or --pdf-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pdf_path in pdfs:
        status, body, latency = _annotate_pdf(args.api_base.rstrip("/"), pdf_path, args.source, args.target, args.timeout)
        row: dict[str, Any] = {
            "filename": pdf_path.name,
            "status": status,
            "latency_sec": round(latency, 4),
        }

        if status != 200:
            row["error"] = body.get("detail") or body.get("error") or body.get("raw", "request_failed")
            rows.append(row)
            continue

        pages = body.get("pages", []) if isinstance(body.get("pages", []), list) else []
        flagged_terms: list[str] = []
        pages_with_terms = 0
        for p in pages:
            terms = p.get("terms", []) if isinstance(p.get("terms", []), list) else []
            if terms:
                pages_with_terms += 1
            for t in terms:
                if isinstance(t, dict) and bool(t.get("flagged", False)):
                    flagged_terms.append(str(t.get("term", "")))

        row.update(
            {
                "document_id": body.get("document_id"),
                "page_count": int(body.get("page_count", 0)),
                "predicted_src": body.get("predicted_src"),
                "predicted_src_confidence": float(body.get("predicted_src_confidence") or 0.0),
                "is_ambiguous": bool(body.get("is_ambiguous", False)),
                "src_warning": bool(body.get("src_warning", False)),
                "src_warning_reason": body.get("src_warning_reason", "none"),
                "pages_with_terms": pages_with_terms,
                "flagged_term_count": len(flagged_terms),
                "top_flagged_terms": flagged_terms[:10],
                "qa": _qa_indicators(body),
            }
        )

        row["explanations"] = _explain_top_terms(
            args.api_base.rstrip("/"),
            body,
            args.target,
            max(0, int(args.explain_top_k)),
            args.timeout,
        )
        rows.append(row)

    ts = _now_ts()
    out_json = out_dir / f"pdf_eval_{ts}.json"
    out_md = out_dir / f"pdf_eval_{ts}.md"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_base": args.api_base.rstrip("/"),
        "source": args.source,
        "target": args.target,
        "count": len(rows),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_markdown(rows, out_md, args.api_base.rstrip("/"))

    print(json.dumps({"out_md": str(out_md.resolve()), "out_json": str(out_json.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

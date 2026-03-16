from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from term_lens_eval_lib import resolve_live_api_base

BANNED = ["equivalent to", "identical to", "the same as"]
PREFERRED = ["analogous", "conceptually similar", "shares structural similarity", "similar in that"]


def _resolve_api(repo_root: Path, requested: str) -> tuple[str, str]:
    if requested:
        return requested.rstrip("/"), "arg"
    live, source = resolve_live_api_base(repo_root)
    if live:
        return live.rstrip("/"), source
    return "http://127.0.0.1:8000", "default-local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit explanation quality for flagged terms.")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--examples-json", default="")
    parser.add_argument("--out-dir", default="reports/explanations")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    api_base, source = _resolve_api(repo_root, args.api_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    examples_path = Path(args.examples_json) if args.examples_json else None
    if examples_path is None or not examples_path.exists():
        fallback = sorted((repo_root / "reports" / "manual_eval").glob("manual_eval_examples_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not fallback:
            raise SystemExit("No manual eval examples found. Run build_manual_eval_pack.py first.")
        examples_path = fallback[0]

    examples = json.loads(examples_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise SystemExit("manual eval examples json must be a list")

    rows: list[dict[str, object]] = []
    errors = 0

    for ex in examples[: max(1, int(args.max_cases))]:
        text = str(ex.get("text", ""))
        src = str(ex.get("src", "auto"))
        tgt = str(ex.get("tgt", "PM"))

        ann_payload = {"text": text, "src": src, "tgt": tgt, "same_field_mode": "normal", "max_terms": 6}
        try:
            ann = requests.post(f"{api_base}/annotate", json=ann_payload, timeout=args.timeout)
            ann_body = ann.json() if ann.headers.get("content-type", "").startswith("application/json") else {"raw": ann.text}
        except Exception as exc:
            rows.append({"id": ex.get("id", ""), "status": 0, "error": f"annotate_error: {exc}"})
            errors += 1
            continue

        terms = ann_body.get("terms", []) if isinstance(ann_body, dict) else []
        flagged = [t for t in terms if isinstance(t, dict) and bool(t.get("flagged"))]
        if not flagged:
            rows.append({"id": ex.get("id", ""), "status": 200, "note": "no_flagged_terms"})
            continue

        term = str(flagged[0].get("term", ""))
        analogs = [str(a.get("candidate", "")) for a in flagged[0].get("analogs", []) if isinstance(a, dict)]
        payload = {
            "text": text,
            "term": term,
            "src": src,
            "tgt": tgt,
            "audience_level": "grad",
            "subtrack": "",
            "analogs": analogs,
            "detail": "long",
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{api_base}/explain", json=payload, timeout=args.timeout)
            latency = time.perf_counter() - t0
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
            status = int(resp.status_code)
        except Exception as exc:
            rows.append({"id": ex.get("id", ""), "status": 0, "error": f"explain_error: {exc}"})
            errors += 1
            continue

        short = str(body.get("short_explanation", "")) if isinstance(body, dict) else ""
        long = str(body.get("long_explanation", "")) if isinstance(body, dict) else ""
        blob = (short + " " + long).lower()

        banned_hits = [p for p in BANNED if p in blob]
        preferred_hits = [p for p in PREFERRED if p in blob]

        row = {
            "id": ex.get("id", ""),
            "status": status,
            "term": term,
            "latency_sec": round(float(latency), 4),
            "short_len": len(short),
            "long_len": len(long),
            "banned_hits": banned_hits,
            "preferred_hits": preferred_hits,
            "policy_ok": bool((not banned_hits) and preferred_hits),
            "empty_output": bool(not short.strip() and not long.strip()),
        }
        if status != 200 or row["empty_output"]:
            errors += 1
        rows.append(row)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": api_base,
        "endpoint_source": source,
        "examples": str(examples_path),
        "rows": rows,
        "summary": {
            "total": len(rows),
            "errors": errors,
            "policy_violations": sum(1 for r in rows if isinstance(r, dict) and r.get("policy_ok") is False and r.get("status", 0) == 200),
            "empty_outputs": sum(1 for r in rows if isinstance(r, dict) and r.get("empty_output") is True),
            "avg_latency_sec": round(sum(float(r.get("latency_sec", 0.0)) for r in rows if isinstance(r, dict)) / max(1, len(rows)), 4),
        },
    }

    json_path = out_dir / f"explanation_audit_{ts}.json"
    md_path = out_dir / f"explanation_audit_{ts}.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Explanation Audit",
        "",
        f"- Generated (UTC): {audit['generated_at']}",
        f"- Endpoint: {api_base}",
        f"- Endpoint source: {source}",
        f"- Examples file: {examples_path}",
        "",
        "## Summary",
        "",
        f"- total: {audit['summary']['total']}",
        f"- errors: {audit['summary']['errors']}",
        f"- policy_violations: {audit['summary']['policy_violations']}",
        f"- empty_outputs: {audit['summary']['empty_outputs']}",
        f"- avg_latency_sec: {audit['summary']['avg_latency_sec']}",
        "",
        "## Case Table",
        "",
        "| id | status | term | latency(s) | short_len | long_len | banned_hits | preferred_hits | policy_ok |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        if "term" not in r:
            lines.append(f"| {r.get('id', '')} | {r.get('status', 0)} | - | - | - | - | - | - | false |")
            continue
        lines.append(
            f"| {r['id']} | {r['status']} | {r['term']} | {r['latency_sec']:.4f} | {r['short_len']} | {r['long_len']} | {', '.join(r['banned_hits'])} | {', '.join(r['preferred_hits'])} | {r['policy_ok']} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit_md": str(md_path.resolve()), "audit_json": str(json_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

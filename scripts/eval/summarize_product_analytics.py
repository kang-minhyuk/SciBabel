from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _iso(ts: int | float | None) -> str:
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _read_events(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def build_summary(events: list[dict[str, object]]) -> dict[str, object]:
    by_event: dict[str, list[dict[str, object]]] = defaultdict(list)
    for ev in events:
        event_type = str(ev.get("event_type") or "unknown")
        by_event[event_type].append(ev)

    event_summaries: dict[str, dict[str, object]] = {}
    for event_type, rows in by_event.items():
        status_counts = Counter(int(r.get("status_code") or 0) for r in rows)
        error_counts = Counter(str(r.get("error_reason") or "none") for r in rows)
        latencies = [float(r.get("latency_ms") or 0.0) for r in rows]

        event_summaries[event_type] = {
            "count": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "error_counts": dict(sorted(error_counts.items())),
            "latency_ms": {
                "avg": round(sum(latencies) / max(1, len(latencies)), 2),
                "p50": round(_percentile(latencies, 0.5), 2),
                "p95": round(_percentile(latencies, 0.95), 2),
                "p99": round(_percentile(latencies, 0.99), 2),
            },
        }

    annotate_rows = by_event.get("annotate", []) + by_event.get("profile_annotate", [])
    explain_rows = by_event.get("explain", [])

    annotate_flagged = [int((r.get("result") or {}).get("flagged_terms") or 0) for r in annotate_rows if isinstance(r, dict)]
    annotate_terms = [int((r.get("result") or {}).get("total_terms") or 0) for r in annotate_rows if isinstance(r, dict)]
    annotate_ambiguous = sum(1 for r in annotate_rows if bool((r.get("result") or {}).get("is_ambiguous", False)))

    explain_has_short = sum(1 for r in explain_rows if bool((r.get("result") or {}).get("has_short", False)))

    ts_values = [int(ev.get("ts") or 0) for ev in events if int(ev.get("ts") or 0) > 0]

    return {
        "window": {
            "start_utc": _iso(min(ts_values) if ts_values else None),
            "end_utc": _iso(max(ts_values) if ts_values else None),
        },
        "total_events": len(events),
        "by_event": event_summaries,
        "quality": {
            "annotate_avg_terms": round(sum(annotate_terms) / max(1, len(annotate_terms)), 3),
            "annotate_avg_flagged_terms": round(sum(annotate_flagged) / max(1, len(annotate_flagged)), 3),
            "annotate_ambiguity_rate": round(annotate_ambiguous / max(1, len(annotate_rows)), 4),
            "explain_short_nonempty_rate": round(explain_has_short / max(1, len(explain_rows)), 4),
        },
    }


def write_markdown(summary: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Product Analytics Summary")
    lines.append("")
    lines.append(f"- Window start: `{summary['window']['start_utc']}`")
    lines.append(f"- Window end: `{summary['window']['end_utc']}`")
    lines.append(f"- Total events: `{summary['total_events']}`")
    lines.append("")
    lines.append("## Event Breakdown")
    lines.append("")

    by_event = summary.get("by_event", {})
    if isinstance(by_event, dict):
        for event_name, detail in sorted(by_event.items()):
            if not isinstance(detail, dict):
                continue
            lat = detail.get("latency_ms", {}) if isinstance(detail.get("latency_ms", {}), dict) else {}
            lines.append(f"### {event_name}")
            lines.append(f"- Count: `{detail.get('count', 0)}`")
            lines.append(f"- Status counts: `{json.dumps(detail.get('status_counts', {}), sort_keys=True)}`")
            lines.append(f"- Error counts: `{json.dumps(detail.get('error_counts', {}), sort_keys=True)}`")
            lines.append(
                "- Latency (ms): "
                f"`avg={lat.get('avg', 0)} p50={lat.get('p50', 0)} p95={lat.get('p95', 0)} p99={lat.get('p99', 0)}`"
            )
            lines.append("")

    q = summary.get("quality", {}) if isinstance(summary.get("quality", {}), dict) else {}
    lines.append("## Quality Signals")
    lines.append("")
    lines.append(f"- Annotate avg terms: `{q.get('annotate_avg_terms', 0)}`")
    lines.append(f"- Annotate avg flagged terms: `{q.get('annotate_avg_flagged_terms', 0)}`")
    lines.append(f"- Annotate ambiguity rate: `{q.get('annotate_ambiguity_rate', 0)}`")
    lines.append(f"- Explain short non-empty rate: `{q.get('explain_short_nonempty_rate', 0)}`")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize product analytics JSONL logs")
    parser.add_argument(
        "--input",
        default="logs/product_analytics.jsonl",
        help="Path to product analytics JSONL input",
    )
    parser.add_argument(
        "--out-md",
        default="reports/product_analytics/product_analytics_summary.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--out-json",
        default="reports/product_analytics/product_analytics_summary.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    events = _read_events(input_path)
    summary = build_summary(events)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)

    print(f"[ok] Read {len(events)} events from {input_path}")
    print(f"[ok] Wrote {out_md}")
    print(f"[ok] Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

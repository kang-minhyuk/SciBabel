from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from term_lens_eval_lib import (
    load_cases,
    now_ts,
    resolve_live_api_base,
    run_regression,
    write_regression_reports,
)


def _drift(local: list[dict], prod: list[dict], field: str) -> dict[str, float]:
    local_map = {(r["id"], r["same_field_mode"]): r for r in local}
    prod_map = {(r["id"], r["same_field_mode"]): r for r in prod}
    keys = sorted(set(local_map.keys()) & set(prod_map.keys()))
    if not keys:
        return {"avg_abs": 0.0, "max_abs": 0.0}
    diffs = [abs(float(local_map[k].get(field, 0.0)) - float(prod_map[k].get(field, 0.0))) for k in keys]
    return {"avg_abs": round(sum(diffs) / len(diffs), 4), "max_abs": round(max(diffs), 4)}


def _mismatches(local: list[dict], prod: list[dict]) -> list[dict]:
    local_map = {(r["id"], r["same_field_mode"]): r for r in local}
    prod_map = {(r["id"], r["same_field_mode"]): r for r in prod}
    out: list[dict] = []
    for key in sorted(set(local_map.keys()) & set(prod_map.keys())):
        l = local_map[key]
        p = prod_map[key]
        if (
            int(l.get("status", 0)) != int(p.get("status", 0))
            or abs(float(l.get("flagged_count", 0)) - float(p.get("flagged_count", 0))) >= 2
            or bool(l.get("is_ambiguous", False)) != bool(p.get("is_ambiguous", False))
            or abs(float(l.get("analog_count", 0)) - float(p.get("analog_count", 0))) >= 2
            or abs(float(l.get("evidence_count", 0)) - float(p.get("evidence_count", 0))) >= 2
        ):
            out.append(
                {
                    "id": l["id"],
                    "mode": l["same_field_mode"],
                    "local": {
                        "status": l["status"],
                        "flagged": l["flagged_count"],
                        "ambiguous": l["is_ambiguous"],
                        "analogs": l["analog_count"],
                        "evidence": l["evidence_count"],
                    },
                    "prod": {
                        "status": p["status"],
                        "flagged": p["flagged_count"],
                        "ambiguous": p["is_ambiguous"],
                        "analogs": p["analog_count"],
                        "evidence": p["evidence_count"],
                    },
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local+prod term-lens regression and produce comparison report.")
    parser.add_argument("--local-api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--prod-api-base", default="")
    parser.add_argument("--cases", default="scripts/eval/regression_cases_term_lens.jsonl")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_ts()

    prod_base = args.prod_api_base.strip().rstrip("/")
    source = "arg"
    if not prod_base:
        resolved, source = resolve_live_api_base(repo_root)
        if not resolved:
            # Create explicit TODO report and fail clearly.
            compare_md = out_dir / f"term_lens_compare_{ts}.md"
            compare_json = out_dir / f"term_lens_compare_{ts}.json"
            todo = {
                "error": "live_api_base_missing",
                "searched_keys": ["SCIBABEL_API_BASE_URL", "NEXT_PUBLIC_API_BASE_URL", "CLOUD_RUN_BASE_URL"],
                "resolution_status": source,
                "instruction": "Set SCIBABEL_API_BASE_URL (or NEXT_PUBLIC_API_BASE_URL/CLOUD_RUN_BASE_URL) and rerun make regression-term-lens-compare",
            }
            compare_json.write_text(json.dumps(todo, ensure_ascii=False, indent=2), encoding="utf-8")
            compare_md.write_text(
                "\n".join(
                    [
                        "# Term Lens Compare Report",
                        "",
                        "## TODO: Live Endpoint Missing",
                        "",
                        "No live endpoint could be resolved.",
                        "",
                        "Set one of these and rerun:",
                        "- SCIBABEL_API_BASE_URL",
                        "- NEXT_PUBLIC_API_BASE_URL",
                        "- CLOUD_RUN_BASE_URL",
                        "",
                        f"Resolution status: {source}",
                    ]
                ),
                encoding="utf-8",
            )
            print(json.dumps({"error": "live_api_base_missing", "compare_md": str(compare_md.resolve()), "compare_json": str(compare_json.resolve())}, ensure_ascii=False))
            raise SystemExit(2)
        prod_base = resolved

    cases = load_cases(Path(args.cases))
    local_result = run_regression(args.local_api_base.rstrip("/"), cases, timeout=float(args.timeout))
    prod_result = run_regression(prod_base, cases, timeout=float(args.timeout))

    local_md, local_json = write_regression_reports(local_result, out_dir, label="local", ts=ts)
    prod_md, prod_json = write_regression_reports(prod_result, out_dir, label="prod", ts=ts)

    cmp = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_endpoint": args.local_api_base.rstrip("/"),
        "prod_endpoint": prod_base,
        "prod_source": source,
        "total_cases": len(cases),
        "local": {
            "success_count": local_result["success_count"],
            "failure_count": local_result["failure_count"],
            "p50_latency_sec": local_result["p50_latency_sec"],
            "p95_latency_sec": local_result["p95_latency_sec"],
        },
        "prod": {
            "success_count": prod_result["success_count"],
            "failure_count": prod_result["failure_count"],
            "p50_latency_sec": prod_result["p50_latency_sec"],
            "p95_latency_sec": prod_result["p95_latency_sec"],
        },
        "term_count_drift": _drift(local_result["records"], prod_result["records"], "flagged_count"),
        "ambiguity_drift": _drift(local_result["records"], prod_result["records"], "is_ambiguous"),
        "analog_drift": _drift(local_result["records"], prod_result["records"], "analog_count"),
        "evidence_drift": _drift(local_result["records"], prod_result["records"], "evidence_count"),
        "mismatches": _mismatches(local_result["records"], prod_result["records"]),
        "local_report_md": str(local_md),
        "prod_report_md": str(prod_md),
        "local_report_json": str(local_json),
        "prod_report_json": str(prod_json),
    }

    cmp_json = out_dir / f"term_lens_compare_{ts}.json"
    cmp_md = out_dir / f"term_lens_compare_{ts}.md"
    cmp_json.write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Term Lens Compare Report",
        "",
        f"- Generated (UTC): {cmp['generated_at']}",
        f"- Local endpoint: {cmp['local_endpoint']}",
        f"- Prod endpoint: {cmp['prod_endpoint']}",
        f"- Total cases: {cmp['total_cases']}",
        "",
        "## Success / Failure",
        "",
        f"- Local: success={cmp['local']['success_count']}, failure={cmp['local']['failure_count']}",
        f"- Prod: success={cmp['prod']['success_count']}, failure={cmp['prod']['failure_count']}",
        "",
        "## Latency",
        "",
        f"- Local p50/p95: {cmp['local']['p50_latency_sec']} / {cmp['local']['p95_latency_sec']} sec",
        f"- Prod p50/p95: {cmp['prod']['p50_latency_sec']} / {cmp['prod']['p95_latency_sec']} sec",
        "",
        "## Drift Summary",
        "",
        f"- term-count drift (avg_abs/max_abs): {cmp['term_count_drift']['avg_abs']} / {cmp['term_count_drift']['max_abs']}",
        f"- ambiguity drift (avg_abs/max_abs): {cmp['ambiguity_drift']['avg_abs']} / {cmp['ambiguity_drift']['max_abs']}",
        f"- analog drift (avg_abs/max_abs): {cmp['analog_drift']['avg_abs']} / {cmp['analog_drift']['max_abs']}",
        f"- evidence drift (avg_abs/max_abs): {cmp['evidence_drift']['avg_abs']} / {cmp['evidence_drift']['max_abs']}",
        "",
        "## Mismatches Worth Manual Inspection",
        "",
        "| Case | Mode | Local(status/flagged/amb/analog/evidence) | Prod(status/flagged/amb/analog/evidence) |",
        "|---|---|---|---|",
    ]

    mismatches = cmp["mismatches"]
    if not mismatches:
        lines.append("| none | - | - | - |")
    else:
        for m in mismatches[:40]:
            lines.append(
                f"| {m['id']} | {m['mode']} | {m['local']['status']}/{m['local']['flagged']}/{m['local']['ambiguous']}/{m['local']['analogs']}/{m['local']['evidence']} | {m['prod']['status']}/{m['prod']['flagged']}/{m['prod']['ambiguous']}/{m['prod']['analogs']}/{m['prod']['evidence']} |"
            )

    cmp_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "local_md": str(local_md.resolve()),
                "prod_md": str(prod_md.resolve()),
                "compare_md": str(cmp_md.resolve()),
                "compare_json": str(cmp_json.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

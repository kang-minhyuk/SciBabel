from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from term_lens_eval_lib import (
    load_cases,
    now_ts,
    resolve_live_api_base,
    run_regression,
    write_regression_reports,
)


def _write_missing_live_report(out_dir: Path, ts: str, reason: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"term_lens_regression_prod_{ts}.md"
    lines = [
        "# Term Lens Regression Report (prod)",
        "",
        "## TODO: Live Endpoint Missing",
        "",
        "Live Cloud Run URL could not be resolved.",
        "",
        "Searched keys:",
        "- SCIBABEL_API_BASE_URL",
        "- NEXT_PUBLIC_API_BASE_URL",
        "- CLOUD_RUN_BASE_URL",
        "",
        f"Resolution status: {reason}",
        "",
        "Set one of the variables above and rerun `make regression-term-lens-prod`.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run term-lens regression cases against /annotate.")
    parser.add_argument("--api-base", default="", help="Endpoint base URL. If omitted for prod/live label, auto-resolve from env/config.")
    parser.add_argument("--label", default="local", help="Report label (e.g., local, prod)")
    parser.add_argument("--cases", default="scripts/eval/regression_cases_term_lens.jsonl")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out)
    ts = now_ts()

    api_base = args.api_base.strip().rstrip("/")
    label = str(args.label).strip().lower()

    if not api_base:
        if label in {"prod", "live"}:
            resolved, source = resolve_live_api_base(repo_root)
            if not resolved:
                md_path = _write_missing_live_report(out_dir, ts, source)
                print(json.dumps({"error": "live_api_base_missing", "todo_report": str(md_path.resolve())}, ensure_ascii=False))
                raise SystemExit(2)
            api_base = resolved
        else:
            api_base = "http://127.0.0.1:8000"

    cases = load_cases(Path(args.cases))
    result = run_regression(api_base=api_base, cases=cases, timeout=float(args.timeout))
    md_path, json_path = write_regression_reports(result, out_dir=out_dir, label=label, ts=ts)

    print(
        json.dumps(
            {
                "label": label,
                "endpoint": api_base,
                "report_md": str(md_path.resolve()),
                "report_json": str(json_path.resolve()),
                "success_count": result["success_count"],
                "failure_count": result["failure_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

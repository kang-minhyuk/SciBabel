from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_extract_terms():
    root = _repo_root()
    backend_dir = root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from terms.extract import extract_terms  # pylint: disable=import-outside-toplevel

    return extract_terms


def _run_cases() -> dict[str, object]:
    extract_terms = _load_extract_terms()
    cases = [
        {
            "id": "case_1",
            "text": "We optimize a graph neural network with sparse regularization under distribution shift.",
            "must_include": ["graph neural network", "sparse regularization", "distribution shift"],
            "must_exclude": ["optimize a graph"],
            "before_bad": ["distribution shift", "graph neural network", "optimize a graph", "sparse regularization"],
        },
        {
            "id": "case_2",
            "text": "The transformer uses low-rank attention to reduce memory cost on long sequences.",
            "must_include": ["low-rank attention"],
            "must_exclude": ["memory cost on", "on long sequences", "the transformer"],
            "before_bad": ["memory cost on", "on long sequences", "the transformer"],
        },
        {
            "id": "case_3",
            "text": "The phase transition is characterized by an order parameter near criticality.",
            "must_include": ["phase transition", "order parameter", "criticality"],
            "must_exclude": ["by an", "transition is characterized", "parameter near criticality"],
            "before_bad": ["transition is characterized", "by an"],
        },
        {
            "id": "case_4",
            "text": "We train a diffusion model with classifier-free guidance for molecular generation.",
            "must_include": ["diffusion model", "classifier-free guidance", "molecular generation"],
            "must_exclude": ["train a diffusion"],
            "before_bad": ["train a diffusion"],
        },
    ]

    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for case in cases:
        items = extract_terms(case["text"], max_terms=30)
        terms = {str(x.get("term", "")).lower() for x in items}

        missing = [t for t in case["must_include"] if t not in terms]
        forbidden_present = [t for t in case["must_exclude"] if t in terms]

        if missing:
            failures.append(f"{case['id']}: missing {missing}")
        if forbidden_present:
            failures.append(f"{case['id']}: forbidden {forbidden_present}")

        rows.append(
            {
                "id": case["id"],
                "text": case["text"],
                "before_bad": case["before_bad"],
                "after_terms": sorted(terms),
                "missing": missing,
                "forbidden_present": forbidden_present,
                "pass": not missing and not forbidden_present,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(rows),
        "passed": sum(1 for r in rows if r["pass"]),
        "failed": len(failures),
        "failures": failures,
        "rows": rows,
    }


def _write_markdown(summary: dict[str, object], out_md: Path) -> None:
    lines = [
        "# Fragment Suppression Mini-Suite",
        "",
        f"- Generated (UTC): {summary['generated_at']}",
        f"- Total cases: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        "",
        "## Before / After",
        "",
        "| Case | Before bad spans | Missing required | Forbidden present | Result |",
        "|---|---|---|---|---|",
    ]

    for row in summary["rows"]:
        lines.append(
            "| "
            f"{row['id']} | "
            f"{', '.join(row['before_bad'])} | "
            f"{', '.join(row['missing']) if row['missing'] else '-'} | "
            f"{', '.join(row['forbidden_present']) if row['forbidden_present'] else '-'} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )

    lines.extend(["", "## Remaining Known Failures", ""])
    if summary["failures"]:
        for item in summary["failures"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fragment suppression mini regression suite")
    parser.add_argument("--out-json", default="reports/fragments/fragment_mini_suite_latest.json")
    parser.add_argument("--out-md", default="reports/fragments/fragment_mini_suite_latest.md")
    args = parser.parse_args()

    summary = _run_cases()
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, out_md)

    print(json.dumps({"out_json": str(out_json.resolve()), "out_md": str(out_md.resolve()), "failed": summary["failed"]}, ensure_ascii=False))
    return 1 if int(summary["failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

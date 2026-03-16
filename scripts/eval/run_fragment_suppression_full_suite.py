from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_extract_terms():
    root = _repo_root()
    backend_dir = root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from terms.extract import extract_terms  # pylint: disable=import-outside-toplevel

    return extract_terms


def _cases() -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []

    case1_texts = [
        "We optimize a graph neural network with sparse regularization under distribution shift.",
        "We optimize a graph neural network using sparse regularization under distribution shift.",
        "We optimize a graph neural network and calibrate sparse regularization under distribution shift.",
        "We optimize a graph neural network for robust sparse regularization under distribution shift.",
        "We optimize a graph neural network with adaptive sparse regularization under distribution shift.",
        "We optimize a graph neural network with stronger sparse regularization under distribution shift.",
        "We optimize a graph neural network with sparse regularization across distribution shift.",
        "We optimize a graph neural network while using sparse regularization under distribution shift.",
        "We optimize a graph neural network through sparse regularization under distribution shift.",
        "We optimize a graph neural network with sparse regularization to handle distribution shift.",
    ]
    for i, text in enumerate(case1_texts, 1):
        combos.append(
            {
                "combination": "optimize_graph_fragment",
                "id": f"opt_graph_{i:02d}",
                "text": text,
                "must_include": ["graph neural network", "sparse regularization", "distribution shift"],
                "must_exclude": ["optimize a graph"],
            }
        )

    case2_texts = [
        "The transformer uses low-rank attention to reduce memory cost on long sequences.",
        "The transformer uses low-rank attention to lower memory cost on long sequences.",
        "The transformer uses low-rank attention while reducing memory cost on long sequences.",
        "The transformer uses low-rank attention to reduce memory cost on very long sequences.",
        "The transformer uses low-rank attention to reduce memory cost on lengthy sequences.",
        "The transformer uses low-rank attention and reduces memory cost on long sequences.",
        "The transformer uses low-rank attention to reduce memory cost across long sequences.",
        "The transformer uses low-rank attention to reduce memory cost for long sequences.",
        "The transformer uses low-rank attention to reduce memory cost in long sequences.",
        "The transformer uses low-rank attention to reduce memory cost over long sequences.",
    ]
    for i, text in enumerate(case2_texts, 1):
        combos.append(
            {
                "combination": "memory_cost_connector_fragment",
                "id": f"mem_cost_{i:02d}",
                "text": text,
                "must_include": ["low-rank attention"],
                "must_exclude": ["memory cost on", "on long sequences", "the transformer"],
            }
        )

    case3_texts = [
        "The phase transition is characterized by an order parameter near criticality.",
        "The phase transition is characterized by an order parameter around criticality.",
        "The phase transition is characterized by an order parameter close to criticality.",
        "The phase transition is characterized by an order parameter at criticality.",
        "The phase transition is characterized by an order parameter in the critical regime.",
        "The phase transition is characterized by an order parameter near the critical point.",
        "The phase transition is characterized by an order parameter and criticality analysis.",
        "The phase transition is characterized by an order parameter under criticality constraints.",
        "The phase transition is characterized by an order parameter with criticality evidence.",
        "The phase transition is characterized by an order parameter for criticality studies.",
    ]
    for i, text in enumerate(case3_texts, 1):
        combos.append(
            {
                "combination": "phase_transition_fragment",
                "id": f"phase_{i:02d}",
                "text": text,
                "must_include": ["phase transition", "order parameter"],
                "must_include_any": ["criticality", "critical point", "critical regime"],
                "must_exclude": ["by an", "transition is characterized", "parameter near criticality"],
            }
        )

    case4_texts = [
        "We train a diffusion model with classifier-free guidance for molecular generation.",
        "We train a diffusion model using classifier-free guidance for molecular generation.",
        "We train a diffusion model with classifier-free guidance in molecular generation.",
        "We train a diffusion model with classifier-free guidance toward molecular generation.",
        "We train a diffusion model with classifier-free guidance under molecular generation constraints.",
        "We train a diffusion model and apply classifier-free guidance for molecular generation.",
        "We train a diffusion model with robust classifier-free guidance for molecular generation.",
        "We train a diffusion model with classifier-free guidance across molecular generation tasks.",
        "We train a diffusion model with classifier-free guidance for stable molecular generation.",
        "We train a diffusion model with classifier-free guidance in de novo molecular generation.",
    ]
    for i, text in enumerate(case4_texts, 1):
        combos.append(
            {
                "combination": "train_diffusion_fragment",
                "id": f"diff_{i:02d}",
                "text": text,
                "must_include": ["diffusion model", "classifier-free guidance", "molecular generation"],
                "must_exclude": ["train a diffusion"],
            }
        )

    return combos


def _run() -> dict[str, Any]:
    extract_terms = _load_extract_terms()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    by_combo: dict[str, dict[str, int]] = {}

    for case in _cases():
        combo = case["combination"]
        by_combo.setdefault(combo, {"total": 0, "passed": 0, "failed": 0})

        items = extract_terms(case["text"], max_terms=30)
        terms = {str(x.get("term", "")).lower() for x in items}

        missing = [t for t in case.get("must_include", []) if t not in terms]
        missing_any_group = []
        group = case.get("must_include_any", [])
        if group and not any(g in terms for g in group):
            missing_any_group = list(group)

        forbidden_present = [t for t in case.get("must_exclude", []) if t in terms]

        passed = not missing and not forbidden_present and not missing_any_group
        if not passed:
            failures.append(
                f"{case['id']} [{combo}] missing={missing or '-'} missing_any={missing_any_group or '-'} forbidden={forbidden_present or '-'}"
            )

        rows.append(
            {
                "id": case["id"],
                "combination": combo,
                "text": case["text"],
                "must_include": case.get("must_include", []),
                "must_include_any": case.get("must_include_any", []),
                "must_exclude": case.get("must_exclude", []),
                "terms": sorted(terms),
                "missing": missing,
                "missing_any": missing_any_group,
                "forbidden_present": forbidden_present,
                "pass": passed,
            }
        )

        by_combo[combo]["total"] += 1
        if passed:
            by_combo[combo]["passed"] += 1
        else:
            by_combo[combo]["failed"] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(rows),
        "total_passed": sum(1 for r in rows if r["pass"]),
        "total_failed": len(failures),
        "by_combination": by_combo,
        "failures": failures,
        "rows": rows,
    }


def _write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Fragment Suppression Full Validation",
        "",
        f"- Generated (UTC): {summary['generated_at']}",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed: {summary['total_passed']}",
        f"- Failed: {summary['total_failed']}",
        "",
        "## Combination Summary",
        "",
        "| Combination | Total | Passed | Failed | Pass Rate |",
        "|---|---:|---:|---:|---:|",
    ]

    by_combination = summary.get("by_combination", {})
    for combo, stat in sorted(by_combination.items()):
        total = int(stat.get("total", 0))
        passed = int(stat.get("passed", 0))
        failed = int(stat.get("failed", 0))
        rate = (100.0 * passed / total) if total else 0.0
        lines.append(f"| {combo} | {total} | {passed} | {failed} | {rate:.1f}% |")

    lines.extend([
        "",
        "## Failures",
        "",
    ])
    if summary.get("failures"):
        for f in summary["failures"]:
            lines.append(f"- {f}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Case Snapshot",
        "",
        "| Case | Combination | Missing | Forbidden Present | Result |",
        "|---|---|---|---|---|",
    ])

    for row in summary.get("rows", []):
        missing = ", ".join(row.get("missing", [])) if row.get("missing") else "-"
        missing_any = ", ".join(row.get("missing_any", [])) if row.get("missing_any") else ""
        if missing_any:
            missing = f"{missing}; any({missing_any})"
        forbidden = ", ".join(row.get("forbidden_present", [])) if row.get("forbidden_present") else "-"
        result = "PASS" if row.get("pass") else "FAIL"
        lines.append(f"| {row.get('id')} | {row.get('combination')} | {missing} | {forbidden} | {result} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 10-case-per-combination fragment suppression validation")
    parser.add_argument("--out-json", default="reports/fragments/fragment_full_suite_latest.json")
    parser.add_argument("--out-md", default="reports/fragments/fragment_full_suite_latest.md")
    args = parser.parse_args()

    summary = _run()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, out_md)

    print(
        json.dumps(
            {
                "out_json": str(out_json.resolve()),
                "out_md": str(out_md.resolve()),
                "total_cases": summary["total_cases"],
                "total_failed": summary["total_failed"],
            },
            ensure_ascii=False,
        )
    )

    return 1 if int(summary["total_failed"]) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

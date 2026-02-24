from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

BANNED_GENERIC = {
    "model",
    "predict",
    "prediction",
    "steps",
    "reduces",
    "method",
    "methods",
    "results",
    "study",
    "approach",
    "propose",
    "show",
    "data",
    "system",
}
DEBUG_RE = re.compile(r"updatedgpt|native=|domain-specific concept", re.IGNORECASE)


def combined_top_terms(lex: dict, domain: str, top_k: int) -> set[str]:
    node = lex.get(domain, {})
    items = (node.get("top_terms", []) + node.get("top_bigrams", []) + node.get("top_trigrams", []))[:top_k]
    return {str(x).lower() for x in items}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate text-mining artifacts")
    parser.add_argument("--lexicon", default="data/processed/domain_lexicon.json")
    parser.add_argument("--term-stats", default="data/processed/term_stats.csv")
    parser.add_argument("--clf-metrics", default="models/domain_clf_metrics.json")
    parser.add_argument("--report-out", default="reports/textmining/validation_report.md")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--macro-f1-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    lex_path = Path(args.lexicon)
    term_path = Path(args.term_stats)
    clf_path = Path(args.clf_metrics)

    lex = json.loads(lex_path.read_text(encoding="utf-8"))
    domains = [d for d in ["CSM", "PM", "CCE"] if d in lex]

    # banned terms check
    banned_hits: dict[str, list[str]] = {}
    for d in domains:
        tops = combined_top_terms(lex, d, args.top_k)
        bad = sorted([t for t in tops if t in BANNED_GENERIC])
        if bad:
            banned_hits[d] = bad
            failures.append(f"Banned generic terms in {d}: {bad[:10]}")

    # overlap check
    overlap_rows: list[str] = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            sa = combined_top_terms(lex, a, args.top_k)
            sb = combined_top_terms(lex, b, args.top_k)
            inter = len(sa & sb)
            union = max(1, len(sa | sb))
            overlap = inter / union
            overlap_rows.append(f"{a}-{b}: {overlap:.3f}")
            if overlap > args.overlap_threshold:
                failures.append(f"Overlap too high {a}-{b}: {overlap:.3f} > {args.overlap_threshold:.3f}")

    # debug artifacts in term_stats
    ts = pd.read_csv(term_path)
    debug_rows = ts[ts["term"].astype(str).str.contains(DEBUG_RE)]
    if len(debug_rows) > 0:
        failures.append(f"term_stats contains debug artifacts: {len(debug_rows)} rows")

    # classifier quality gate
    clf_metrics = json.loads(clf_path.read_text(encoding="utf-8"))
    macro_f1 = float(clf_metrics.get("macro_f1", 0.0))
    if macro_f1 < args.macro_f1_threshold:
        failures.append(f"macro_f1 too low: {macro_f1:.4f} < {args.macro_f1_threshold:.4f}")

    notes.append(f"macro_f1={macro_f1:.4f}")
    notes.extend(overlap_rows)

    lines = [
        "# Validation Report",
        "",
        f"- lexicon: {lex_path}",
        f"- term_stats: {term_path}",
        f"- clf_metrics: {clf_path}",
        "",
        "## Notes",
    ]
    lines.extend([f"- {n}" for n in notes])

    lines.append("")
    lines.append("## Failures")
    if failures:
        lines.extend([f"- {f}" for f in failures])
    else:
        lines.append("- none")

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote validation report: {out}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()

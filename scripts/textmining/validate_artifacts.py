from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from common import load_yaml

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


def combined_style_terms(lex: dict, domain: str, top_k: int) -> set[str]:
    node = lex.get(domain, {})
    if isinstance(node, list):
        items = node[:top_k]
    else:
        items = (node.get("style", []) + node.get("bigrams", []) + node.get("trigrams", []))[:top_k]
    return {str(x).lower() for x in items}


def _domain_min_style(cfg: dict, domain: str, default_min: int) -> int:
    try:
        val = cfg.get("domains", {}).get(domain, {}).get("min_style_terms_per_domain", default_min)
        return int(val)
    except Exception:
        return default_min


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate text-mining artifacts")
    parser.add_argument("--lexicon", default="data/processed/domain_lexicon.json")
    parser.add_argument("--term-stats", default="data/processed/term_stats.csv")
    parser.add_argument("--clf-metrics", default="models/domain_clf_metrics.json")
    parser.add_argument("--report-out", default="reports/textmining/validation_report.md")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--corpus-diagnostics", default="reports/textmining/corpus_report.md")
    parser.add_argument("--overlap-threshold", type=float, default=0.25)
    parser.add_argument("--macro-f1-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--default-min-style", type=int, default=30)
    args = parser.parse_args()

    failures: list[str] = []
    notes: list[str] = []

    lex_path = Path(args.lexicon)
    term_path = Path(args.term_stats)
    clf_path = Path(args.clf_metrics)
    cfg_path = Path(args.config)
    corp_diag_path = Path(args.corpus_diagnostics)

    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}

    lex = json.loads(lex_path.read_text(encoding="utf-8"))
    domains = [d for d in ["CSM", "PM", "CCE"] if d in lex]

    # banned terms check
    banned_hits: dict[str, list[str]] = {}
    for d in domains:
        tops = combined_style_terms(lex, d, args.top_k)
        bad = sorted([t for t in tops if t in BANNED_GENERIC])
        if bad:
            banned_hits[d] = bad
            failures.append(f"Banned generic terms in {d}: {bad[:10]}")

    # minimum style terms per domain
    for d in domains:
        node = lex.get(d, {})
        if isinstance(node, list):
            style_count = len(node)
        else:
            style_count = len(node.get("style", []))
        min_style = _domain_min_style(cfg, d, args.default_min_style)
        notes.append(f"style_count[{d}]={style_count} (min={min_style})")
        if style_count < min_style:
            failures.append(f"STYLE terms too few in {d}: {style_count} < {min_style}")

    # overlap check
    overlap_rows: list[str] = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            sa = combined_style_terms(lex, a, args.top_k)
            sb = combined_style_terms(lex, b, args.top_k)
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

    # corpus diagnostics context (if available)
    if corp_diag_path.exists() and corp_diag_path.suffix.lower() == ".json":
        try:
            corpus_diag = json.loads(corp_diag_path.read_text(encoding="utf-8"))
            by_domain = corpus_diag.get("by_domain", {})
            by_source = corpus_diag.get("by_domain_source", {})
            notes.append("corpus by_domain: " + ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items())))
            if by_source:
                for d in sorted(by_source):
                    row = by_source[d]
                    notes.append(f"sources[{d}]: " + ", ".join(f"{k}={v}" for k, v in sorted(row.items())))
        except Exception as exc:
            notes.append(f"WARN corpus diagnostics parse failed: {exc}")
    elif corp_diag_path.exists():
        notes.append(f"corpus diagnostics present: {corp_diag_path}")

    notes.append(f"macro_f1={macro_f1:.4f}")
    notes.extend(overlap_rows)

    lines = [
        "# Validation Report",
        "",
        f"- lexicon: {lex_path}",
        f"- term_stats: {term_path}",
        f"- clf_metrics: {clf_path}",
        f"- config: {cfg_path}",
        f"- corpus_diagnostics: {corp_diag_path}",
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

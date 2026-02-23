from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

DOMAINS = ["CSM", "PM", "CCE"]

SENTENCES: dict[str, list[str]] = {
    "CSM": [
        "We propose a graph-regularized objective that improves robustness under covariate shift.",
        "Our transformer uses sparse attention to reduce memory while preserving long-range dependencies.",
        "The optimization converges in O(n log n) with a provable approximation guarantee.",
    ],
    "PM": [
        "The phonon dispersion indicates strong anharmonic scattering at elevated temperature.",
        "We estimate carrier mobility from Hall measurements and compare with DFT-derived band structure.",
        "The model predicts crack initiation when the stress intensity exceeds the critical threshold.",
    ],
    "CCE": [
        "Catalyst deactivation is mitigated by controlling residence time and reactant partial pressure.",
        "The reactor achieves higher selectivity through staged feed and temperature ramping.",
        "Mass transfer limitations dominate at high conversion due to pore diffusion resistance.",
    ],
}


@dataclass
class EvalCase:
    src: str
    tgt: str
    text: str


def build_cases(include_same_domain: bool) -> list[EvalCase]:
    out: list[EvalCase] = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            if (not include_same_domain) and src == tgt:
                continue
            for text in SENTENCES[src]:
                out.append(EvalCase(src=src, tgt=tgt, text=text))
    return out


def post_translate(api_base: str, case: EvalCase, k: int, timeout: int) -> tuple[int, dict[str, Any]]:
    payload = {"text": case.text, "src": case.src, "tgt": case.tgt, "k": k}
    r = requests.post(f"{api_base}/translate", json=payload, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text}
    return r.status_code, body


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SciBabel translate endpoint over domain combinations")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--include-same-domain", action="store_true")
    parser.add_argument("--out-dir", default="logs")
    parser.add_argument("--cache-bust-tag", default="")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"translation_eval_{ts}.jsonl"
    summary_path = out_dir / f"translation_eval_{ts}_summary.md"

    cases = build_cases(include_same_domain=args.include_same_domain)
    by_pair: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with raw_path.open("w", encoding="utf-8") as wf:
        for idx, case in enumerate(cases, start=1):
            send_case = case
            if args.cache_bust_tag:
                send_case = EvalCase(
                    src=case.src,
                    tgt=case.tgt,
                    text=f"{case.text} [{args.cache_bust_tag}]",
                )

            status, body = post_translate(args.api_base, send_case, args.k, args.timeout)
            rec: dict[str, Any] = {
                "index": idx,
                "src": case.src,
                "tgt": case.tgt,
                "text": case.text,
                "status": status,
                "ok": status == 200,
                "response": body,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pair_key = f"{case.src}->{case.tgt}"
            by_pair[pair_key]["count"] += 1
            if status == 200:
                by_pair[pair_key]["ok"] += 1
                if body.get("used_fallback"):
                    by_pair[pair_key]["fallback"] += 1
                if body.get("src_warning"):
                    by_pair[pair_key]["src_warning"] += 1

                cands = body.get("candidates") or []
                if cands:
                    best = cands[0]
                    bd = best.get("breakdown") or {}
                    by_pair[pair_key]["semantic_sum"] += float(bd.get("semantic_sim", bd.get("meaning", 0.0)) or 0.0)
                    by_pair[pair_key]["copy_sum"] += float(bd.get("copy_score", 0.0) or 0.0)
                    by_pair[pair_key]["score_sum"] += float(best.get("total_score", 0.0) or 0.0)
            else:
                by_pair[pair_key]["error"] += 1

    lines = [
        f"# Translation Evaluation Summary ({ts})",
        "",
        f"API: {args.api_base}",
        f"k: {args.k}",
        f"include_same_domain: {args.include_same_domain}",
        "",
        "| Pair | N | OK | Fallback | Src warning | Avg semantic | Avg copy | Avg score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    problem_notes: list[str] = []
    for pair in sorted(by_pair.keys()):
        m = by_pair[pair]
        n = int(m.get("count", 0))
        ok = int(m.get("ok", 0))
        fallback = int(m.get("fallback", 0))
        src_warning = int(m.get("src_warning", 0))

        avg_sem = (m.get("semantic_sum", 0.0) / ok) if ok else 0.0
        avg_copy = (m.get("copy_sum", 0.0) / ok) if ok else 0.0
        avg_score = (m.get("score_sum", 0.0) / ok) if ok else 0.0

        lines.append(
            f"| {pair} | {n} | {ok} | {fallback} | {src_warning} | {avg_sem:.3f} | {avg_copy:.3f} | {avg_score:.3f} |"
        )

        if n and fallback / n >= 0.5:
            problem_notes.append(f"High fallback rate in {pair}: {fallback}/{n}")
        if ok and avg_copy >= 0.9:
            problem_notes.append(f"High copy behavior in {pair}: avg copy={avg_copy:.3f}")
        if ok and avg_sem < 0.55:
            problem_notes.append(f"Low semantic retention in {pair}: avg semantic={avg_sem:.3f}")

    lines.append("")
    lines.append("## Automatic problem flags")
    if problem_notes:
        lines.extend([f"- {p}" for p in problem_notes])
    else:
        lines.append("- No major automatic flags.")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote raw log: {raw_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

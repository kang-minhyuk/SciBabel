from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

LEAK_PATTERNS = [
    re.compile(r"\(domain-specific concept\)", re.IGNORECASE),
    re.compile(r"\(native=", re.IGNORECASE),
    re.compile(r"\[updatedgpt", re.IGNORECASE),
    re.compile(r"native=0\.00", re.IGNORECASE),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def leak(text: str) -> bool:
    return any(p.search(text) for p in LEAK_PATTERNS)


def safe_avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def med(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def breakdown_from_candidate(c: dict[str, Any]) -> dict[str, float]:
    bd = c.get("breakdown") or {}
    return {
        "domain": float(bd.get("domain", 0.0) or 0.0),
        "semantic_sim": float(bd.get("semantic_sim", bd.get("meaning", 0.0)) or 0.0),
        "lex": float(bd.get("lex", 0.0) or 0.0),
        "copy_score": float(bd.get("copy_score", 0.0) or 0.0),
        "strategy_penalty": float(bd.get("strategy_penalty", 0.0) or 0.0),
    }


def simulate_pick(
    candidates: list[dict[str, Any]],
    min_sem: float,
    lex_weight: float,
    copy_weight: float,
    lex_clamp: float,
) -> tuple[dict[str, Any] | None, bool]:
    eligible = []
    for c in candidates:
        bd = breakdown_from_candidate(c)
        if bd["semantic_sim"] < min_sem:
            continue
        score = bd["domain"] + lex_weight * min(bd["lex"], lex_clamp) - copy_weight * bd["copy_score"] - bd["strategy_penalty"]
        eligible.append((score, c))

    if eligible:
        eligible.sort(key=lambda x: x[0], reverse=True)
        return eligible[0][1], False

    if not candidates:
        return None, True

    best_sem = max(candidates, key=lambda c: breakdown_from_candidate(c)["semantic_sim"])
    return best_sem, True


def case_metrics_from_pick(pick: dict[str, Any] | None, fallback_by_filter: bool, lex_hack: float, sem_th: float) -> dict[str, float]:
    if not pick:
        return {
            "domain": 0.0,
            "semantic": 0.0,
            "lex": 0.0,
            "copy": 0.0,
            "annotation_leak": 0.0,
            "reward_hack": 0.0,
            "fallback_by_filter": 1.0,
        }

    bd = breakdown_from_candidate(pick)
    text = str(pick.get("text", ""))
    reward_hack = 1.0 if (bd["lex"] > lex_hack and bd["semantic_sim"] < sem_th) else 0.0

    return {
        "domain": bd["domain"],
        "semantic": bd["semantic_sim"],
        "lex": bd["lex"],
        "copy": bd["copy_score"],
        "annotation_leak": 1.0 if leak(text) else 0.0,
        "reward_hack": reward_hack,
        "fallback_by_filter": 1.0 if fallback_by_filter else 0.0,
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "domain_mean": safe_avg([r["domain"] for r in rows]),
        "semantic_mean": safe_avg([r["semantic"] for r in rows]),
        "lex_mean": safe_avg([r["lex"] for r in rows]),
        "copy_mean": safe_avg([r["copy"] for r in rows]),
        "annotation_leak_rate": safe_avg([r["annotation_leak"] for r in rows]),
        "reward_hacking_rate": safe_avg([r["reward_hack"] for r in rows]),
        "fallback_by_filter_rate": safe_avg([r["fallback_by_filter"] for r in rows]),
    }


def objective(m: dict[str, float]) -> float:
    return (
        m["domain_mean"]
        - 1.6 * m["reward_hacking_rate"]
        - 1.1 * m["annotation_leak_rate"]
        - 0.8 * m["fallback_by_filter_rate"]
        - 0.2 * m["copy_mean"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune reranker from existing eval logs (no LLM calls)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--lex-hack-threshold", type=float, default=1.0)
    parser.add_argument("--sem-threshold", type=float, default=0.75)
    args = parser.parse_args()

    in_path = Path(args.input)
    rows = load_jsonl(in_path)
    ok_rows = [r for r in rows if r.get("ok") and r.get("status") == 200]

    baseline_case_rows: list[dict[str, float]] = []
    for row in ok_rows:
        resp = row.get("response") or {}
        bd = resp.get("score_breakdown") or {}
        picked = {
            "text": resp.get("best_candidate", ""),
            "breakdown": {
                "domain": bd.get("domain", 0.0),
                "semantic_sim": bd.get("semantic_sim", bd.get("meaning", 0.0)),
                "lex": bd.get("lex", 0.0),
                "copy_score": bd.get("copy_score", 0.0),
                "strategy_penalty": bd.get("strategy_penalty", 0.0),
            },
        }
        baseline_case_rows.append(
            case_metrics_from_pick(
                picked,
                fallback_by_filter=False,
                lex_hack=args.lex_hack_threshold,
                sem_th=args.sem_threshold,
            )
        )

    baseline = aggregate(baseline_case_rows)

    min_sem_grid = [0.70, 0.75, 0.80, 0.85]
    lex_weight_grid = [0.0, 0.1, 0.25, 0.5]
    copy_weight_grid = [0.0, 0.1, 0.25]
    lex_clamp_grid = [0.5, 1.0, 2.0]

    best_params: dict[str, float] | None = {
        "MIN_SEMANTIC_SIM": -1.0,
        "LEX_WEIGHT": -1.0,
        "COPY_PENALTY_WEIGHT": -1.0,
        "LEX_CLAMP": -1.0,
    }
    best_metrics: dict[str, float] | None = baseline
    best_obj = objective(baseline)

    for min_sem, lex_weight, copy_weight, lex_clamp in itertools.product(
        min_sem_grid,
        lex_weight_grid,
        copy_weight_grid,
        lex_clamp_grid,
    ):
        case_rows: list[dict[str, float]] = []
        for row in ok_rows:
            cands = (row.get("response") or {}).get("candidates") or []
            pick, fb = simulate_pick(
                candidates=cands,
                min_sem=min_sem,
                lex_weight=lex_weight,
                copy_weight=copy_weight,
                lex_clamp=lex_clamp,
            )
            case_rows.append(
                case_metrics_from_pick(
                    pick,
                    fallback_by_filter=fb,
                    lex_hack=args.lex_hack_threshold,
                    sem_th=args.sem_threshold,
                )
            )

        m = aggregate(case_rows)
        obj = objective(m)
        if obj > best_obj:
            best_obj = obj
            best_params = {
                "MIN_SEMANTIC_SIM": min_sem,
                "LEX_WEIGHT": lex_weight,
                "COPY_PENALTY_WEIGHT": copy_weight,
                "LEX_CLAMP": lex_clamp,
            }
            best_metrics = m

    assert best_params is not None
    assert best_metrics is not None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path("reports") / f"autotune_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Reranker Autotune Report ({ts})",
        "",
        f"Input log: {in_path}",
        f"Cases used: {len(ok_rows)}",
        "",
        "## Best parameter set",
        "",
        (
            "- No grid setting beat baseline; keep current production reranker."
            if best_params["MIN_SEMANTIC_SIM"] < 0
            else f"- MIN_SEMANTIC_SIM={best_params['MIN_SEMANTIC_SIM']:.2f}"
        ),
        ("" if best_params["LEX_WEIGHT"] < 0 else f"- LEX_WEIGHT={best_params['LEX_WEIGHT']:.2f}"),
        (
            ""
            if best_params["COPY_PENALTY_WEIGHT"] < 0
            else f"- COPY_PENALTY_WEIGHT={best_params['COPY_PENALTY_WEIGHT']:.2f}"
        ),
        ("" if best_params["LEX_CLAMP"] < 0 else f"- LEX_CLAMP={best_params['LEX_CLAMP']:.2f}"),
        "",
        "## Recommended config values",
        "",
        (
            "- AUTOTUNE_RECOMMENDATION=KEEP_BASELINE"
            if best_params["MIN_SEMANTIC_SIM"] < 0
            else f"- AUTOTUNE_MIN_SEMANTIC_SIM={best_params['MIN_SEMANTIC_SIM']:.2f}"
        ),
        ("" if best_params["LEX_WEIGHT"] < 0 else f"- AUTOTUNE_LEX_WEIGHT={best_params['LEX_WEIGHT']:.2f}"),
        (
            ""
            if best_params["COPY_PENALTY_WEIGHT"] < 0
            else f"- AUTOTUNE_COPY_PENALTY_WEIGHT={best_params['COPY_PENALTY_WEIGHT']:.2f}"
        ),
        ("" if best_params["LEX_CLAMP"] < 0 else f"- AUTOTUNE_LEX_CLAMP={best_params['LEX_CLAMP']:.2f}"),
        "",
        "## Before/After summary",
        "",
        "| Metric | Baseline | Tuned |",
        "|---|---:|---:|",
        f"| mean(domain) | {baseline['domain_mean']:.3f} | {best_metrics['domain_mean']:.3f} |",
        f"| mean(semantic) | {baseline['semantic_mean']:.3f} | {best_metrics['semantic_mean']:.3f} |",
        f"| mean(lex) | {baseline['lex_mean']:.3f} | {best_metrics['lex_mean']:.3f} |",
        f"| mean(copy) | {baseline['copy_mean']:.3f} | {best_metrics['copy_mean']:.3f} |",
        f"| reward_hacking_rate | {baseline['reward_hacking_rate']:.3f} | {best_metrics['reward_hacking_rate']:.3f} |",
        f"| annotation_leak_rate | {baseline['annotation_leak_rate']:.3f} | {best_metrics['annotation_leak_rate']:.3f} |",
        f"| fallback_by_filter_rate | {baseline['fallback_by_filter_rate']:.3f} | {best_metrics['fallback_by_filter_rate']:.3f} |",
        "",
        "## Objective",
        "",
        f"- baseline objective: {objective(baseline):.4f}",
        f"- tuned objective: {objective(best_metrics):.4f}",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Autotune Summary ===")
    print(f"input: {in_path}")
    if best_params["MIN_SEMANTIC_SIM"] < 0:
        print("best params: KEEP_BASELINE")
    else:
        print(f"best params: {best_params}")
    print(f"baseline objective: {objective(baseline):.4f}")
    print(f"tuned objective: {objective(best_metrics):.4f}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()

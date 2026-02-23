from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")
LEAK_PATTERNS = [
    re.compile(r"\(domain-specific concept\)", re.IGNORECASE),
    re.compile(r"\(native=", re.IGNORECASE),
    re.compile(r"\[updatedgpt", re.IGNORECASE),
    re.compile(r"native=0\.00", re.IGNORECASE),
]
DEBUG_PATTERNS = [
    re.compile(r"updatedgpt", re.IGNORECASE),
    re.compile(r"run\d{2,}", re.IGNORECASE),
    re.compile(r"noself", re.IGNORECASE),
    re.compile(r"debug", re.IGNORECASE),
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "under",
    "with",
    "when",
    "while",
    "we",
    "our",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def jaccard(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def maybe_load_embedding_model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def embedding_sim(model: Any, a: str, b: str) -> float:
    emb = model.encode([a, b], normalize_embeddings=True)
    return float((emb[0] * emb[1]).sum())


def contains_leak(text: str) -> bool:
    return any(p.search(text) for p in LEAK_PATTERNS)


def contains_debug_term(text: str) -> bool:
    return any(p.search(text) for p in DEBUG_PATTERNS)


def is_stopword_only_term(term: str) -> bool:
    toks = [t.lower() for t in TOKEN_RE.findall(term)]
    if not toks:
        return True
    return all(t in STOPWORDS for t in toks)


def safe_med(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_avg(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SciBabel eval logs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--lex-hack-threshold", type=float, default=1.0)
    parser.add_argument("--sem-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--diversity-sim", choices=["jaccard", "embedding-auto"], default="jaccard")
    args = parser.parse_args()

    in_path = Path(args.input)
    rows = load_jsonl(in_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path("reports") / f"diagnosis_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok_rows = [r for r in rows if r.get("ok") and r.get("status") == 200]
    responses = [r.get("response") or {} for r in ok_rows]
    embed_model = maybe_load_embedding_model() if args.diversity_sim == "embedding-auto" else None

    fallback_rate = safe_avg([1.0 if r.get("used_fallback") else 0.0 for r in responses])

    sem_vals = [float((r.get("score_breakdown") or {}).get("semantic_sim", 0.0) or 0.0) for r in responses]
    copy_vals = [float((r.get("score_breakdown") or {}).get("copy_score", 0.0) or 0.0) for r in responses]
    domain_vals = [float((r.get("score_breakdown") or {}).get("domain", 0.0) or 0.0) for r in responses]
    lex_vals = [float((r.get("score_breakdown") or {}).get("lex", 0.0) or 0.0) for r in responses]

    pairwise_sims: list[float] = []
    low_diversity_cases: list[tuple[float, dict[str, Any], list[str]]] = []
    for row in ok_rows:
        resp = row.get("response") or {}
        cands = resp.get("candidates") or []
        texts = [c.get("text", "") for c in cands if c.get("text")]
        if len(texts) < 2:
            continue
        sims: list[float] = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sims.append(jaccard(texts[i], texts[j]))
                if embed_model is not None:
                    try:
                        sims[-1] = embedding_sim(embed_model, texts[i], texts[j])
                    except Exception:
                        pass
        if sims:
            avg_sim = safe_avg(sims)
            pairwise_sims.append(avg_sim)
            low_diversity_cases.append((avg_sim, row, texts[:2]))

    annotation_cases: list[dict[str, Any]] = []
    for row in ok_rows:
        text = str((row.get("response") or {}).get("best_candidate", ""))
        if contains_leak(text):
            annotation_cases.append(row)
    annotation_leak_rate = (len(annotation_cases) / len(ok_rows)) if ok_rows else 0.0

    total_terms = 0
    stopword_only = 0
    debug_terms = 0
    noisy_rows: list[tuple[float, dict[str, Any], list[str]]] = []
    for row in ok_rows:
        terms = (row.get("response") or {}).get("term_strategies") or []
        row_terms = [str(t.get("term", "")) for t in terms]
        total_terms += len(row_terms)
        row_noise = 0.0
        for term in row_terms:
            if is_stopword_only_term(term):
                stopword_only += 1
                row_noise += 1.0
            if contains_debug_term(term):
                debug_terms += 1
                row_noise += 1.0
        if row_terms:
            noisy_rows.append((row_noise / len(row_terms), row, row_terms[:4]))

    term_count_avg = (total_terms / len(ok_rows)) if ok_rows else 0.0
    frac_stopword_only = (stopword_only / total_terms) if total_terms else 0.0
    frac_debug_suffix = (debug_terms / total_terms) if total_terms else 0.0

    reward_hacks: list[dict[str, Any]] = []
    for row in ok_rows:
        bd = (row.get("response") or {}).get("score_breakdown") or {}
        lex = float(bd.get("lex", 0.0) or 0.0)
        sem = float(bd.get("semantic_sim", 0.0) or 0.0)
        if lex > args.lex_hack_threshold and sem < args.sem_threshold:
            reward_hacks.append(row)

    low_sem = sorted(
        ok_rows,
        key=lambda r: float(((r.get("response") or {}).get("score_breakdown") or {}).get("semantic_sim", 0.0) or 0.0),
    )[: args.top_k]
    high_copy = sorted(
        ok_rows,
        key=lambda r: float(((r.get("response") or {}).get("score_breakdown") or {}).get("copy_score", 0.0) or 0.0),
        reverse=True,
    )[: args.top_k]
    low_div = sorted(low_diversity_cases, key=lambda x: x[0], reverse=True)[: args.top_k]
    noisy = sorted(noisy_rows, key=lambda x: x[0], reverse=True)[: args.top_k]
    hacks = reward_hacks[: args.top_k]

    lines: list[str] = []
    lines.append(f"# SciBabel Diagnosis Report ({ts})")
    lines.append("")
    lines.append(f"Input log: {in_path}")
    lines.append(f"Total cases: {len(rows)}")
    lines.append(f"Successful cases: {len(ok_rows)}")
    lines.append("")
    lines.append("## Summary metrics")
    lines.append("")
    lines.append(f"- fallback_rate: {fallback_rate:.3f}")
    lines.append(f"- semantic_sim avg/median: {safe_avg(sem_vals):.3f} / {safe_med(sem_vals):.3f}")
    lines.append(f"- copy_score avg/median: {safe_avg(copy_vals):.3f} / {safe_med(copy_vals):.3f}")
    lines.append(f"- domain avg/median: {safe_avg(domain_vals):.3f} / {safe_med(domain_vals):.3f}")
    lines.append(f"- lex avg/median: {safe_avg(lex_vals):.3f} / {safe_med(lex_vals):.3f}")
    lines.append(
        f"- candidate_diversity (avg pairwise Jaccard similarity; lower is better): {safe_avg(pairwise_sims):.3f}"
    )
    lines.append(f"- annotation_leak_rate: {annotation_leak_rate:.3f}")
    lines.append(f"- term_noise_score: avg_count={term_count_avg:.2f}, stopword_only_frac={frac_stopword_only:.3f}, debug_suffix_frac={frac_debug_suffix:.3f}")
    lines.append(f"- reward_hacking_count: {len(reward_hacks)}")
    lines.append("")

    lines.append("## Failure modes (explicit)")
    lines.append("")
    lines.append(f"- reward hacking (lex dominates): {'YES' if len(reward_hacks) > 0 else 'NO'}")
    lines.append(f"- annotation leaks: {'YES' if annotation_leak_rate > 0 else 'NO'}")
    lines.append(f"- term noise: {'YES' if (frac_stopword_only > 0 or frac_debug_suffix > 0) else 'NO'}")
    lines.append(f"- low diversity: {'YES' if safe_avg(pairwise_sims) > 0.7 else 'NO'}")
    lines.append("")
    lines.append("### What to fix first")
    lines.append("1. Raise semantic floor / reduce lex weight where reward hacking appears.")
    lines.append("2. Sanitize term strategy extraction (drop stopword-only/debug tokens).")
    lines.append("3. Block annotation artifacts from final output post-processing.")
    lines.append("4. Increase action diversity or deduplicate near-identical candidates.")
    lines.append("")

    def add_cases(title: str, case_rows: list[dict[str, Any]]) -> None:
        lines.append(f"### {title}")
        if not case_rows:
            lines.append("- none")
            lines.append("")
            return
        for row in case_rows:
            resp = row.get("response") or {}
            bd = resp.get("score_breakdown") or {}
            lines.append(
                f"- {row.get('id')} ({row.get('src')}->{row.get('tgt')}): "
                f"sem={float(bd.get('semantic_sim',0.0) or 0.0):.3f}, "
                f"lex={float(bd.get('lex',0.0) or 0.0):.3f}, "
                f"copy={float(bd.get('copy_score',0.0) or 0.0):.3f}"
            )
            lines.append(f"  - input: {row.get('text','')}")
            lines.append(f"  - output: {resp.get('best_candidate','')}")
        lines.append("")

    add_cases("Top Failures: reward hacking", hacks)
    add_cases("Top Failures: low semantic", low_sem)
    add_cases("Top Failures: high copy", high_copy)

    lines.append("### Top Failures: annotation leaks")
    if not annotation_cases:
        lines.append("- none")
    else:
        for row in annotation_cases[: args.top_k]:
            resp = row.get("response") or {}
            lines.append(f"- {row.get('id')} ({row.get('src')}->{row.get('tgt')}): {resp.get('best_candidate','')}")
    lines.append("")

    lines.append("### Top Failures: term noise")
    if not noisy:
        lines.append("- none")
    else:
        for ratio, row, terms in noisy:
            lines.append(f"- {row.get('id')} ({row.get('src')}->{row.get('tgt')}): noise_ratio={ratio:.3f}, terms={terms}")
    lines.append("")

    lines.append("### Top Failures: low diversity")
    if not low_div:
        lines.append("- none")
    else:
        for sim, row, texts in low_div:
            lines.append(f"- {row.get('id')} ({row.get('src')}->{row.get('tgt')}): pairwise_sim={sim:.3f}")
            lines.append(f"  - c1: {texts[0]}")
            lines.append(f"  - c2: {texts[1]}")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Diagnosis Summary ===")
    print(f"input: {in_path}")
    print(f"cases: {len(rows)} | ok: {len(ok_rows)}")
    print(f"fallback_rate: {fallback_rate:.3f}")
    print(f"avg semantic/copy/domain/lex: {safe_avg(sem_vals):.3f} / {safe_avg(copy_vals):.3f} / {safe_avg(domain_vals):.3f} / {safe_avg(lex_vals):.3f}")
    print(f"annotation_leak_rate: {annotation_leak_rate:.3f}")
    print(f"term_noise: avg_count={term_count_avg:.2f}, stopword_only_frac={frac_stopword_only:.3f}, debug_suffix_frac={frac_debug_suffix:.3f}")
    print(f"reward_hacking_count: {len(reward_hacks)}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()

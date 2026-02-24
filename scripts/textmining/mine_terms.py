from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ACADEMIC_STOPWORDS = {
    "model",
    "models",
    "method",
    "methods",
    "results",
    "study",
    "approach",
    "approaches",
    "propose",
    "proposed",
    "demonstrate",
    "show",
    "predict",
    "prediction",
    "paper",
    "data",
    "dataset",
    "datasets",
    "system",
    "performance",
    "steps",
    "reduces",
    "using",
    "based",
    "analysis",
    "novel",
    "work",
}

DEBUG_TOKENS = ("updatedgpt", "native=", "domain-specific concept")


def load_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Corpus must be parquet or jsonl")


def ngram_len(term: str) -> int:
    return len(term.split())


def valid_term(term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return False
    if any(x in t for x in DEBUG_TOKENS):
        return False
    toks = t.split()
    if any(tok in ACADEMIC_STOPWORDS for tok in toks):
        return False
    if all(tok in ACADEMIC_STOPWORDS for tok in toks):
        return False
    if all(not any(ch.isalpha() for ch in tok) for tok in toks):
        return False
    if len(t.replace(" ", "")) < 3:
        return False
    return True


def log_odds_with_prior(counts_d: np.ndarray, counts_o: np.ndarray, prior_mass: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    total = counts_d + counts_o
    total_sum = float(total.sum())
    if total_sum <= 0:
        z = np.zeros_like(total, dtype=float)
        delta = np.zeros_like(total, dtype=float)
        return z, delta

    prior = (total / total_sum) * prior_mass
    alpha0 = float(prior.sum())

    n_d = float(counts_d.sum())
    n_o = float(counts_o.sum())

    num_d = counts_d + prior
    den_d = np.maximum((n_d + alpha0) - num_d, 1e-9)
    num_o = counts_o + prior
    den_o = np.maximum((n_o + alpha0) - num_o, 1e-9)

    delta = np.log(num_d / den_d) - np.log(num_o / den_o)
    var = (1.0 / np.maximum(num_d, 1e-9)) + (1.0 / np.maximum(num_o, 1e-9))
    z = delta / np.sqrt(np.maximum(var, 1e-12))
    return z, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine robust domain lexicon + term stats")
    parser.add_argument("--corpus", default="data/processed/corpus.parquet")
    parser.add_argument("--term-stats-out", default="data/processed/term_stats.csv")
    parser.add_argument("--lexicon-out", default="data/processed/domain_lexicon.json")
    parser.add_argument("--report-out", default="reports/textmining/lexicon_report.md")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.4)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    df = load_corpus(Path(args.corpus)).dropna(subset=["text", "domain"]).copy()
    texts = df["text"].astype(str).tolist()
    domains = sorted(df["domain"].astype(str).unique().tolist())

    vec = CountVectorizer(
        ngram_range=(1, 3),
        lowercase=True,
        min_df=args.min_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-_/\.\(\)]*\b",
    )
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    keep_mask = np.array([valid_term(t) for t in terms], dtype=bool)
    filtered_generic = sorted([str(t) for t, k in zip(terms, keep_mask) if not k and (t in ACADEMIC_STOPWORDS)])

    X = X[:, keep_mask]
    terms = terms[keep_mask]

    df_total = np.asarray((X > 0).sum(axis=0)).ravel()
    global_keep = df_total <= int(args.max_df * X.shape[0])
    removed_global = sorted([str(t) for t, gk in zip(terms, global_keep) if not gk])
    X = X[:, global_keep]
    terms = terms[global_keep]
    df_total = df_total[global_keep]

    domain_masks = {d: (df["domain"].astype(str).values == d) for d in domains}
    total_counts = np.asarray(X.sum(axis=0)).ravel().astype(float)

    rows: list[dict[str, Any]] = []
    lexicon: dict[str, Any] = {}

    for d in domains:
        mask_d = domain_masks[d]
        counts_d = np.asarray(X[mask_d].sum(axis=0)).ravel().astype(float)
        counts_o = total_counts - counts_d
        z, delta = log_odds_with_prior(counts_d, counts_o)

        df_d = np.asarray((X[mask_d] > 0).sum(axis=0)).ravel()

        stats = pd.DataFrame(
            {
                "term": terms,
                "domain": d,
                "z": z,
                "delta": delta,
                "count_domain": counts_d,
                "count_total": total_counts,
                "docfreq_domain": df_d,
                "docfreq_total": df_total,
                "ngram_len": [ngram_len(t) for t in terms],
            }
        ).sort_values("z", ascending=False)

        rows.extend(stats.to_dict(orient="records"))

        uni = stats[stats["ngram_len"] == 1].head(args.top_n)["term"].tolist()
        bi = stats[stats["ngram_len"] == 2].head(args.top_n)["term"].tolist()
        tri = stats[stats["ngram_len"] == 3].head(args.top_n)["term"].tolist()
        lexicon[d] = {
            "top_terms": uni,
            "top_bigrams": bi,
            "top_trigrams": tri,
        }

    term_stats = pd.DataFrame(rows)
    out_stats = Path(args.term_stats_out)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    term_stats.to_csv(out_stats, index=False)

    lexicon["meta"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "min_df": args.min_df,
        "filters": {
            "max_df_ratio": args.max_df,
            "academic_stopwords_count": len(ACADEMIC_STOPWORDS),
            "removed_global_frequent_count": len(removed_global),
        },
    }

    out_lex = Path(args.lexicon_out)
    out_lex.parent.mkdir(parents=True, exist_ok=True)
    out_lex.write_text(json.dumps(lexicon, indent=2), encoding="utf-8")

    # report
    overlap_rows = []
    combined = {}
    for d in domains:
        combined[d] = set((lexicon[d]["top_terms"] + lexicon[d]["top_bigrams"] + lexicon[d]["top_trigrams"])[:200])
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            inter = len(combined[a] & combined[b])
            union = max(1, len(combined[a] | combined[b]))
            overlap_rows.append((a, b, inter / union))

    rep_lines = [
        "# Lexicon Report",
        "",
        f"Rows: {len(df)}",
        f"Vocabulary size: {len(terms)}",
        "",
        "## Top 50 terms per domain",
        "",
    ]
    for d in domains:
        rep_lines.append(f"### {d}")
        top50 = (lexicon[d]["top_bigrams"][:25] + lexicon[d]["top_trigrams"][:25])[:50]
        for t in top50:
            rep_lines.append(f"- {t}")
        rep_lines.append("")

    rep_lines.append("## Overlap statistics (top200 combined)")
    for a, b, ov in overlap_rows:
        rep_lines.append(f"- {a} vs {b}: {ov:.3f}")
    rep_lines.append("")
    rep_lines.append("## Filtered generic terms")
    for t in filtered_generic[:200]:
        rep_lines.append(f"- {t}")
    rep_lines.append("")
    rep_lines.append("## Global frequent filtered terms")
    for t in removed_global[:200]:
        rep_lines.append(f"- {t}")

    out_rep = Path(args.report_out)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep.write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    print(f"Wrote term stats: {out_stats}")
    print(f"Wrote lexicon: {out_lex}")
    print(f"Wrote report: {out_rep}")


if __name__ == "__main__":
    main()

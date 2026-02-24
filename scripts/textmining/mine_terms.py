from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_stoplist
from phrase_extract import extract_phrases, normalize_phrase, valid_phrase


def load_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Corpus must be parquet or jsonl")


def ngram_len(term: str) -> int:
    return len(term.split())


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


def ctfidf_score(count_domain: np.ndarray, count_total: np.ndarray) -> np.ndarray:
    tf = count_domain / np.maximum(float(count_domain.sum()), 1e-12)
    idf = np.log(1.0 + (float(count_total.sum()) / np.maximum(count_total, 1.0)))
    return tf * idf


def is_valid_term(term: str, academic_stop: set[str], debug_terms: set[str], generic_stop: set[str]) -> bool:
    t = normalize_phrase(term)
    if not t:
        return False
    if not valid_phrase(t):
        return False
    if any(d in t for d in debug_terms):
        return False
    if t in academic_stop or t in generic_stop:
        return False
    return True


def classify_style_vs_topic(term: str, topic_nouns: set[str], academic_stop: set[str]) -> str:
    toks = term.lower().split()
    if len(toks) == 1:
        if toks[0] in topic_nouns or toks[0] in academic_stop:
            return "topic"
        return "style"

    # multiword phrases with methodological shape are style; concrete-noun heavy phrases are topic
    noun_hits = sum(1 for t in toks if t in topic_nouns)
    if noun_hits >= 2:
        return "topic"
    return "style"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine STYLE/TOPIC domain lexicons from spaCy+YAKE phrases")
    parser.add_argument("--corpus", default="data/processed/corpus.parquet")
    parser.add_argument("--term-stats-out", default="data/processed/term_stats.csv")
    parser.add_argument("--lexicon-out", default="data/processed/domain_lexicon.json")
    parser.add_argument("--report-out", default="reports/textmining/lexicon_report.md")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.4)
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()

    academic_stop = load_stoplist("academic_stopwords.txt")
    generic_terms = load_stoplist("generic_science_terms.txt")
    domain_generic = load_stoplist("domain_generic.txt")
    debug_terms = load_stoplist("debug_artifacts.txt")

    df = load_corpus(Path(args.corpus)).dropna(subset=["text", "domain"]).copy()
    domains = sorted(df["domain"].astype(str).unique().tolist())

    domain_counts: dict[str, Counter[str]] = {d: Counter() for d in domains}
    domain_docfreq: dict[str, Counter[str]] = {d: Counter() for d in domains}
    total_counts: Counter[str] = Counter()
    total_docfreq: Counter[str] = Counter()

    for _, row in df.iterrows():
        d = str(row["domain"])
        text = str(row.get("text", ""))
        phrases = extract_phrases(text, top_k=20)
        kept = [p for p in phrases if is_valid_term(p, academic_stop, debug_terms, domain_generic)]
        if not kept:
            continue
        local_count = Counter(kept)
        domain_counts[d].update(local_count)
        total_counts.update(local_count)

        uniq = set(kept)
        domain_docfreq[d].update(uniq)
        total_docfreq.update(uniq)

    vocab = sorted([t for t in total_counts if total_docfreq[t] >= args.min_df])
    if not vocab:
        raise RuntimeError("No phrase vocabulary after filtering. Check corpus and stoplists.")

    # global frequent phrase filtering
    n_docs = len(df)
    removed_global = sorted([t for t in vocab if total_docfreq[t] > int(args.max_df * n_docs)])
    vocab = [t for t in vocab if t not in set(removed_global)]

    rows: list[dict[str, Any]] = []
    lexicon: dict[str, Any] = {}

    for d in domains:
        counts_d = np.array([float(domain_counts[d].get(t, 0.0)) for t in vocab], dtype=float)
        total_arr = np.array([float(total_counts.get(t, 0.0)) for t in vocab], dtype=float)
        counts_o = total_arr - counts_d
        z, delta = log_odds_with_prior(counts_d, counts_o)
        ctfidf = ctfidf_score(counts_d, total_arr)
        df_d = np.array([float(domain_docfreq[d].get(t, 0.0)) for t in vocab], dtype=float)
        df_total = np.array([float(total_docfreq.get(t, 0.0)) for t in vocab], dtype=float)

        stats = pd.DataFrame(
            {
                "term": vocab,
                "domain": d,
                "z": z,
                "delta": delta,
                "ctfidf": ctfidf,
                "count_domain": counts_d,
                "count_total": total_arr,
                "docfreq_domain": df_d,
                "docfreq_total": df_total,
                "ngram_len": [ngram_len(t) for t in vocab],
            }
        )
        stats["combined"] = 0.7 * stats["z"] + 0.3 * np.log1p(np.maximum(stats["ctfidf"], 0))
        stats["kind"] = stats["term"].map(lambda t: classify_style_vs_topic(str(t), generic_terms, academic_stop))
        stats = stats.sort_values("combined", ascending=False)

        rows.extend(stats.drop(columns=["combined"]).to_dict(orient="records"))

        style = stats[stats["kind"] == "style"].copy()
        # Prefer multiword style phrases
        style = style.sort_values(["ngram_len", "z"], ascending=[False, False])
        style_terms = style.head(args.top_n)["term"].astype(str).tolist()

        topic = stats[stats["kind"] == "topic"].copy()
        topic_terms = topic.head(args.top_n)["term"].astype(str).tolist()

        bi = style[style["ngram_len"] == 2].head(args.top_n)["term"].astype(str).tolist()
        tri = style[style["ngram_len"] == 3].head(args.top_n)["term"].astype(str).tolist()

        lexicon[d] = {
            "style": style_terms,
            "topic": topic_terms,
            "bigrams": bi,
            "trigrams": tri,
            "meta": {
                "n_style": len(style_terms),
                "n_topic": len(topic_terms),
            },
        }

    term_stats = pd.DataFrame(rows)
    out_stats = Path(args.term_stats_out)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    term_stats.to_csv(out_stats, index=False)

    lexicon["meta"] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "min_df": args.min_df,
        "max_df": args.max_df,
        "filters": {
            "academic_stopwords": len(academic_stop),
            "generic_science_terms": len(generic_terms),
            "domain_generic_terms": len(domain_generic),
            "removed_global_frequent_count": len(removed_global),
        },
    }
    out_lex = Path(args.lexicon_out)
    out_lex.parent.mkdir(parents=True, exist_ok=True)
    out_lex.write_text(json.dumps(lexicon, indent=2), encoding="utf-8")

    combined = {d: set((lexicon[d]["style"])[:200]) for d in domains}
    overlap_rows = []
    for i, a in enumerate(domains):
        for b in domains[i + 1 :]:
            inter = len(combined[a] & combined[b])
            union = max(1, len(combined[a] | combined[b]))
            overlap_rows.append((a, b, inter / union))

    rep = [
        "# Lexicon Report",
        "",
        f"Rows: {len(df)}",
        f"Vocabulary size: {len(vocab)}",
        "",
        "## Top 50 STYLE/TOPIC per domain",
        "",
    ]
    for d in domains:
        rep.append(f"### {d} STYLE")
        for t in lexicon[d]["style"][:50]:
            rep.append(f"- {t}")
        rep.append("")
        rep.append(f"### {d} TOPIC")
        for t in lexicon[d]["topic"][:50]:
            rep.append(f"- {t}")
        rep.append("")

    rep.append("## STYLE overlap statistics (top200)")
    for a, b, ov in overlap_rows:
        rep.append(f"- {a} vs {b}: {ov:.3f}")
    rep.append("")
    rep.append("## Global frequent filtered terms")
    for t in removed_global[:200]:
        rep.append(f"- {t}")

    out_rep = Path(args.report_out)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"Wrote term stats: {out_stats}")
    print(f"Wrote lexicon: {out_lex}")
    print(f"Wrote report: {out_rep}")


if __name__ == "__main__":
    main()

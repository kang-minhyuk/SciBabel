from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Corpus must be .parquet or .jsonl")


def token_counts(texts: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        for tok in text.lower().split():
            tok = tok.strip(".,;:!?()[]{}\"'")
            if tok.isalpha() and len(tok) > 2:
                counter[tok] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine domain lexicon and term stats")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--lexicon-out", required=True)
    parser.add_argument("--term-stats-out", required=True)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    corpus = load_corpus(Path(args.corpus))
    if "text" not in corpus.columns or "domain" not in corpus.columns:
        raise ValueError("Corpus must contain columns: text, domain")

    corpus = corpus.dropna(subset=["text", "domain"]).copy()
    domains = sorted(corpus["domain"].unique().tolist())

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), min_df=2)
    x = vectorizer.fit_transform(corpus["text"].astype(str))
    terms = vectorizer.get_feature_names_out()

    tfidf_by_domain: dict[str, pd.Series] = {}
    for d in domains:
        idx_mask = (corpus["domain"] == d).to_numpy()
        mean_scores = x[idx_mask].mean(axis=0).A1
        tfidf_by_domain[d] = pd.Series(mean_scores, index=terms).sort_values(ascending=False)

    # Unigram log-odds with add-one smoothing.
    counts_by_domain: dict[str, Counter[str]] = {}
    total_counter: Counter[str] = Counter()
    for d in domains:
        c = token_counts(corpus.loc[corpus["domain"] == d, "text"])
        counts_by_domain[d] = c
        total_counter.update(c)

    vocab = set(total_counter.keys())
    vocab_size = max(len(vocab), 1)

    rows: list[dict[str, float | str]] = []
    lexicon: dict[str, list[str]] = {}

    for d in domains:
        in_counts = counts_by_domain[d]
        out_counts = total_counter - in_counts
        n_in = sum(in_counts.values()) + vocab_size
        n_out = sum(out_counts.values()) + vocab_size

        scores: list[tuple[str, float]] = []
        for term in vocab:
            c_in = in_counts.get(term, 0) + 1
            c_out = out_counts.get(term, 0) + 1
            p_in = c_in / n_in
            p_out = c_out / n_out
            log_odds = math.log(p_in / p_out)
            scores.append((term, log_odds))

        top_log_odds = sorted(scores, key=lambda t: t[1], reverse=True)[: args.top_n * 3]
        top_tfidf_terms = tfidf_by_domain[d].head(args.top_n * 3).index.tolist()

        combined = []
        seen = set()
        for t, _ in top_log_odds:
            if t not in seen:
                combined.append(t)
                seen.add(t)
        for t in top_tfidf_terms:
            if t not in seen:
                combined.append(t)
                seen.add(t)

        lexicon[d] = combined[: args.top_n]

        log_odds_map = dict(top_log_odds)
        for term in lexicon[d]:
            rows.append(
                {
                    "domain": d,
                    "term": term,
                    "log_odds": float(log_odds_map.get(term, 0.0)),
                    "tfidf_mean": float(tfidf_by_domain[d].get(term, 0.0)),
                }
            )

    lexicon_out = Path(args.lexicon_out)
    stats_out = Path(args.term_stats_out)
    lexicon_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    lexicon_out.write_text(json.dumps(lexicon, indent=2), encoding="utf-8")
    pd.DataFrame(rows).sort_values(["domain", "log_odds"], ascending=[True, False]).to_csv(
        stats_out, index=False
    )

    print(f"Wrote lexicon: {lexicon_out}")
    print(f"Wrote term stats: {stats_out}")


if __name__ == "__main__":
    main()

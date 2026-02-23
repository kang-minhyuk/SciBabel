from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Corpus must be .parquet or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train domain classifier")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model-out", required=True)
    args = parser.parse_args()

    df = load_corpus(Path(args.corpus)).dropna(subset=["text", "domain"]).copy()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["domain"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=df["domain"].astype(str),
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)

    preds = pipeline.predict(x_test)
    print(classification_report(y_test, preds))

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()

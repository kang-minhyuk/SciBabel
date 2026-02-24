from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


def load_corpus(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Corpus must be parquet or jsonl")


def build_pipeline(seed: int) -> Pipeline:
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        sublinear_tf=True,
        lowercase=True,
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        lowercase=True,
    )
    feats = FeatureUnion([
        ("word", tfidf_word),
        ("char", tfidf_char),
    ])
    base = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("feats", feats), ("clf", calibrated)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated domain classifier")
    parser.add_argument("--corpus", default="data/processed/corpus.parquet")
    parser.add_argument("--model-out", default="models/domain_clf.joblib")
    parser.add_argument("--report-out", default="reports/textmining/classifier_report.md")
    parser.add_argument("--metrics-out", default="models/domain_clf_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_corpus(Path(args.corpus)).dropna(subset=["text", "domain"]).copy()

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].astype(str),
        df["domain"].astype(str),
        test_size=0.2,
        random_state=args.seed,
        stratify=df["domain"].astype(str),
    )

    pipe = build_pipeline(args.seed)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    labels = sorted(y_val.unique().tolist())
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    report_text = classification_report(y_val, y_pred, digits=4)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rep = [
        "# Classifier Report",
        "",
        f"- accuracy: {acc:.4f}",
        f"- macro_f1: {macro_f1:.4f}",
        f"- train_size: {len(X_train)}",
        f"- val_size: {len(X_val)}",
        "",
        "## Confusion matrix",
        "",
        "labels: " + ", ".join(labels),
        "",
    ]
    for row in cm:
        rep.append("- " + " ".join([str(int(x)) for x in row]))
    rep.extend(["", "## Classification report", "", "```", report_text, "```"])

    rep_out = Path(args.report_out)
    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_out.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved metrics: {metrics_out}")
    print(f"Saved report: {rep_out}")


if __name__ == "__main__":
    main()

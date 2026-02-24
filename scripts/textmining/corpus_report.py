from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate corpus scale report from corpus_full")
    parser.add_argument("--corpus", default="data/processed/corpus_full.parquet")
    parser.add_argument("--out", default="reports/textmining/corpus_scale_report.md")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise RuntimeError(f"Missing corpus file: {corpus_path}")

    df = pd.read_parquet(corpus_path)
    if df.empty:
        raise RuntimeError("Corpus is empty")

    for col in ["domain", "source", "abstract", "year", "published"]:
        if col not in df.columns:
            df[col] = ""

    by_domain = df.groupby("domain").size().to_dict()
    by_source = df.groupby("source").size().to_dict()
    abs_len = df["abstract"].astype(str).map(len)

    year = pd.to_numeric(df["year"], errors="coerce")
    if year.dropna().empty:
        year = pd.to_datetime(df["published"], errors="coerce").dt.year
    year = year.dropna()

    y_min = int(year.min()) if len(year) else 0
    y_max = int(year.max()) if len(year) else 0
    y_med = float(year.median()) if len(year) else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Corpus Scale Report",
        "",
        f"Total docs: {len(df)}",
        f"Average abstract length: {round(float(abs_len.mean()), 2) if len(abs_len) else 0}",
        f"Year distribution (min/max/median): {y_min}/{y_max}/{y_med}",
        "",
        "## Counts per domain",
    ]
    for d, n in sorted(by_domain.items()):
        lines.append(f"- {d}: {int(n)}")

    lines += ["", "## Counts per source"]
    for s, n in sorted(by_source.items()):
        lines.append(f"- {s}: {int(n)}")

    lines += ["", "## CHEM vs CHEME", f"- CHEM: {int(by_domain.get('CHEM', 0))}", f"- CHEME: {int(by_domain.get('CHEME', 0))}"]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={out}")
    for d, n in sorted(by_domain.items()):
        print(f"{d}={int(n)}")
    print(f"TOTAL={len(df)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_input_mapping(items: list[str]) -> list[tuple[Path, str]]:
    parsed: list[tuple[Path, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --inputs entry: {item}. Expected path=DOMAIN")
        path_str, domain = item.split("=", 1)
        parsed.append((Path(path_str), domain))
    return parsed


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge domain JSONL files into a single corpus")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of path=DOMAIN for JSONL files",
    )
    parser.add_argument("--out", required=True, help="Output corpus path (.parquet or .jsonl)")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for path, domain in parse_input_mapping(args.inputs):
        df = load_jsonl(path)
        if "domain" not in df.columns:
            df["domain"] = domain
        df["text"] = (df.get("title", "") + " " + df.get("abstract", "")).astype(str)
        keep_cols = [c for c in ["id", "title", "abstract", "text", "domain", "query"] if c in df.columns]
        frames.append(df[keep_cols])

    merged = pd.concat(frames, ignore_index=True).dropna(subset=["text", "domain"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix == ".parquet":
        merged.to_parquet(out, index=False)
    elif out.suffix == ".jsonl":
        merged.to_json(out, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("--out must end with .parquet or .jsonl")

    print(f"Wrote corpus with {len(merged)} rows to {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import english_like, load_yaml, normalize_for_vectorizer, read_jsonl, sanitize_text, set_seed


def parse_inputs(inputs: list[str]) -> list[Path]:
    return [Path(x) for x in inputs]


def default_input_paths() -> list[Path]:
    roots = [Path("data/raw/arxiv"), Path("data/raw/chemrxiv")]
    out: list[Path] = []
    for root in roots:
        if root.exists():
            out.extend(sorted(root.glob("*.jsonl")))
    return out


def load_raw(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        for row in read_jsonl(path):
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["id", "domain", "source", "title", "abstract", "categories", "published"])
    return pd.DataFrame(rows)


def pick_target(domain: str, cfg: dict, default_target: int) -> int:
    dom = cfg.get("domains", {}).get(domain, {})
    return int(dom.get("target_doc_count", default_target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned, balanced text-mining corpus")
    parser.add_argument("--inputs", nargs="*", default=[], help="Raw JSONL inputs")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--out", default="data/processed/corpus.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg.get("seed", args.seed)))

    input_paths = parse_inputs(args.inputs) if args.inputs else default_input_paths()
    if not input_paths:
        raise RuntimeError("No raw inputs found. Provide --inputs or fetch into data/raw/{arxiv,chemrxiv}.")

    df = load_raw(input_paths)
    if df.empty:
        raise RuntimeError("No records loaded from inputs.")

    for col in ["id", "domain", "source", "title", "abstract", "categories", "published"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str).map(sanitize_text)
    df["abstract"] = df["abstract"].fillna("").astype(str).map(sanitize_text)
    df["text_original"] = (df["title"] + " " + df["abstract"]).str.strip()
    df["text"] = [normalize_for_vectorizer(t, a) for t, a in zip(df["title"], df["abstract"])]

    min_chars = int(cfg.get("corpus", {}).get("min_abstract_chars", 200))
    df = df[df["abstract"].str.len() >= min_chars].copy()
    df = df[df["abstract"].map(english_like)].copy()

    # Optional per-domain keyword filtering (useful for CCE/ChemE split).
    for domain, spec in (cfg.get("domains", {}) or {}).items():
        kws = [str(k).lower() for k in (spec.get("keyword_filters", []) or []) if str(k).strip()]
        if not kws:
            continue
        mask_d = df["domain"].astype(str) == str(domain)
        if mask_d.any():
            text_series = df.loc[mask_d, "text"].astype(str).str.lower()
            keep = text_series.map(lambda t: any(k in t for k in kws))
            df = pd.concat([df.loc[~mask_d], df.loc[mask_d].loc[keep]], ignore_index=True)

    default_target = int(cfg.get("corpus", {}).get("target_doc_count_per_domain", 20000))
    domain_groups = {str(d): g for d, g in df.groupby("domain", sort=True)}
    if len(domain_groups) < 2:
        raise RuntimeError("Need at least two domains for balanced corpus build.")

    max_per_domain = {
        d: min(len(g), pick_target(d, cfg, default_target))
        for d, g in domain_groups.items()
    }
    n_equal = min(max_per_domain.values())
    if n_equal <= 0:
        raise RuntimeError("No rows available after filtering for balanced sampling.")

    balanced_parts = []
    for domain in sorted(domain_groups.keys()):
        g = domain_groups[domain]
        balanced_parts.append(g.sample(n=n_equal, random_state=int(cfg.get("seed", args.seed))))

    if not balanced_parts:
        raise RuntimeError("No rows remained after filtering.")

    out_df = pd.concat(balanced_parts, ignore_index=True)
    out_df = out_df[["id", "domain", "source", "title", "abstract", "text", "text_original", "categories", "published"]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_json(out_path, orient="records", lines=True, force_ascii=False)

    print(f"rows={len(out_df)} out={out_path}")


if __name__ == "__main__":
    main()

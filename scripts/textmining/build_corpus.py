from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    english_like,
    load_yaml,
    normalize_for_vectorizer,
    read_jsonl,
    sanitize_text,
    set_seed,
)


def parse_inputs(inputs: list[str]) -> list[Path]:
    return [Path(x) for x in inputs]


def default_input_paths() -> list[Path]:
    roots = [Path("data/raw/arxiv"), Path("data/raw/openalex")]
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
        return pd.DataFrame(columns=["id", "domain", "source", "title", "abstract", "categories", "concepts", "published", "year"])
    return pd.DataFrame(rows)


def pick_target(domain: str, cfg: dict, default_target: int) -> int:
    dom = cfg.get("domains", {}).get(domain, {})
    return int(dom.get("target_doc_count", default_target))


def summarize_counts(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    by_domain = df.groupby("domain").size().to_dict() if not df.empty else {}
    by_domain_source: dict[str, dict[str, int]] = {}
    if not df.empty:
        grouped = df.groupby(["domain", "source"]).size()
        for (d, s), n in grouped.items():
            by_domain_source.setdefault(str(d), {})[str(s)] = int(n)
    return {str(k): int(v) for k, v in by_domain.items()}, by_domain_source


def to_year_series(df: pd.DataFrame) -> pd.Series:
    if "year" in df.columns:
        year = pd.to_numeric(df["year"], errors="coerce")
    else:
        pub = df.get("published", pd.Series([""] * len(df)))
        year = pd.to_datetime(pub, errors="coerce").dt.year
    return year.dropna().astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned full+balanced text-mining corpus")
    parser.add_argument("--inputs", nargs="*", default=[], help="Raw JSONL inputs")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--out-full", default="data/processed/corpus_full.parquet")
    parser.add_argument("--out-balanced", default="data/processed/corpus_balanced.parquet")
    parser.add_argument("--diagnostics-out", default="reports/textmining/corpus_scale_report.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg.get("seed", args.seed)))

    input_paths = parse_inputs(args.inputs) if args.inputs else default_input_paths()
    if not input_paths:
        raise RuntimeError("No raw inputs found. Provide --inputs or fetch into data/raw/{arxiv,openalex}.")

    df = load_raw(input_paths)
    if df.empty:
        raise RuntimeError("No records loaded from inputs.")

    for col in ["id", "domain", "source", "title", "abstract", "categories", "concepts", "published", "year"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str).map(sanitize_text)
    df["abstract"] = df["abstract"].fillna("").astype(str).map(sanitize_text)
    df["text_original"] = (df["title"] + " " + df["abstract"]).str.strip()
    df["text"] = [normalize_for_vectorizer(t, a) for t, a in zip(df["title"], df["abstract"])]

    min_chars = int(cfg.get("corpus", {}).get("min_abstract_chars", 200))
    stats = {
        "loaded": int(len(df)),
        "drop_short": 0,
        "drop_non_english": 0,
        "drop_keyword": 0,
    }

    short_mask = df["abstract"].str.len() < min_chars
    stats["drop_short"] = int(short_mask.sum())
    df = df[~short_mask].copy()

    eng_mask = df["abstract"].map(english_like)
    stats["drop_non_english"] = int((~eng_mask).sum())
    df = df[eng_mask].copy()

    for domain, spec in (cfg.get("domains", {}) or {}).items():
        kws = [str(k).lower() for k in (spec.get("keyword_filters", []) or []) if str(k).strip()]
        if not kws:
            continue
        mask_d = df["domain"].astype(str) == str(domain)
        if mask_d.any():
            text_series = df.loc[mask_d, "text"].astype(str).str.lower()
            keep = text_series.map(lambda t: any(k in t for k in kws))
            stats["drop_keyword"] += int((~keep).sum())
            df = pd.concat([df.loc[~mask_d], df.loc[mask_d].loc[keep]], ignore_index=True)

    by_domain, by_domain_source = summarize_counts(df)
    min_docs = max(5000, int(cfg.get("corpus", {}).get("min_docs_per_domain", 5000)))
    missing_domains = []
    for domain in (cfg.get("domains", {}) or {}).keys():
        cnt = int(by_domain.get(domain, 0))
        if cnt < min_docs:
            missing_domains.append((domain, cnt, by_domain_source.get(domain, {})))

    abs_len = df["abstract"].astype(str).map(len)
    year_series = to_year_series(df)
    year_min = int(year_series.min()) if len(year_series) else 0
    year_max = int(year_series.max()) if len(year_series) else 0
    year_med = float(year_series.median()) if len(year_series) else 0.0

    diagnostics_out = Path(args.diagnostics_out)
    diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corpus Scale Report",
        "",
        f"Loaded rows: {stats['loaded']}",
        f"After cleaning rows: {len(df)}",
        f"Dropped short abstracts: {stats['drop_short']}",
        f"Dropped non-English heuristic: {stats['drop_non_english']}",
        f"Dropped by keyword filters: {stats['drop_keyword']}",
        f"Average abstract length: {round(float(abs_len.mean()), 2) if len(abs_len) else 0}",
        f"Year distribution (min/max/median): {year_min}/{year_max}/{year_med}",
        "",
        "## Counts by domain",
    ]
    for d in sorted((cfg.get("domains", {}) or {}).keys()):
        lines.append(f"- {d}: {by_domain.get(d, 0)}")
    lines.append("")
    lines.append("## Counts by domain/source")
    for d in sorted((cfg.get("domains", {}) or {}).keys()):
        lines.append(f"### {d}")
        source_counts = by_domain_source.get(d, {})
        if not source_counts:
            lines.append("- none")
        for s, n in sorted(source_counts.items()):
            lines.append(f"- {s}: {n}")
    lines.append("")
    lines.append("## CHEM vs CHEME")
    lines.append(f"- CHEM: {by_domain.get('CHEM', 0)}")
    lines.append(f"- CHEME: {by_domain.get('CHEME', 0)}")
    lines.append("")

    print("[build_corpus] counts by domain:")
    for d in sorted((cfg.get("domains", {}) or {}).keys()):
        print(f"  - {d}: {by_domain.get(d, 0)}")
    print("[build_corpus] counts by domain/source:")
    for d in sorted((cfg.get("domains", {}) or {}).keys()):
        source_counts = by_domain_source.get(d, {})
        if not source_counts:
            print(f"  - {d}: none")
            continue
        pairs = ", ".join(f"{s}={n}" for s, n in sorted(source_counts.items()))
        print(f"  - {d}: {pairs}")
    print("[build_corpus] CHEM vs CHEME:")
    print(f"  - CHEM: {by_domain.get('CHEM', 0)}")
    print(f"  - CHEME: {by_domain.get('CHEME', 0)}")

    if missing_domains:
        lines.append("## Threshold failures")
        for domain, cnt, src_counts in missing_domains:
            lines.append(f"- {domain}: {cnt} < MIN_DOCS_PER_DOMAIN({min_docs})")
            lines.append(f"  - source_counts: {json.dumps(src_counts)}")
            lines.append("  - suggestion: fetch more records for this domain and rerun build")

    diagnostics_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if missing_domains:
        raise RuntimeError(
            "Domain corpus threshold unmet. "
            + "; ".join([f"{d}={c} (<{min_docs})" for d, c, _ in missing_domains])
        )

    default_target = int(cfg.get("corpus", {}).get("target_doc_count_per_domain", 15000))
    domain_groups = {str(d): g for d, g in df.groupby("domain", sort=True)}
    if len(domain_groups) < 2:
        raise RuntimeError("Need at least two domains for balanced corpus build.")

    max_per_domain = {d: min(len(g), pick_target(d, cfg, default_target)) for d, g in domain_groups.items()}
    n_equal = min(max_per_domain.values())
    if n_equal <= 0:
        raise RuntimeError("No rows available after filtering for balanced sampling.")

    balanced_parts = []
    for domain in sorted(domain_groups.keys()):
        g = domain_groups[domain]
        balanced_parts.append(g.sample(n=n_equal, random_state=int(cfg.get("seed", args.seed))))

    if not balanced_parts:
        raise RuntimeError("No rows remained after filtering.")

    base_cols = ["id", "domain", "source", "title", "abstract", "text", "text_original", "categories", "concepts", "published", "year"]
    full_df = df[[c for c in base_cols if c in df.columns]].copy()
    out_balanced_df = pd.concat(balanced_parts, ignore_index=True)
    out_balanced_df = out_balanced_df[[c for c in base_cols if c in out_balanced_df.columns]].copy()

    out_full_path = Path(args.out_full)
    out_full_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_full_path, index=False)

    out_balanced_path = Path(args.out_balanced)
    out_balanced_path.parent.mkdir(parents=True, exist_ok=True)
    out_balanced_df.to_parquet(out_balanced_path, index=False)

    print(f"rows_full={len(full_df)} out_full={out_full_path}")
    print(f"rows_balanced={len(out_balanced_df)} out_balanced={out_balanced_path}")
    print(f"report={diagnostics_out}")


if __name__ == "__main__":
    main()

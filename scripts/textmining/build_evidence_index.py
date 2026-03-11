from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

TOKEN_RE = re.compile(r"\w+")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _has_all_tokens(haystack: str, phrase: str) -> bool:
    ht = set(TOKEN_RE.findall(_normalize(haystack)))
    pt = [t for t in TOKEN_RE.findall(_normalize(phrase)) if t]
    if not pt:
        return False
    return all(t in ht for t in pt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight evidence index for annotate term evidence.")
    parser.add_argument("--corpus", required=True, help="Input corpus parquet path")
    parser.add_argument("--lexicon", default="data/processed/domain_lexicon.json", help="Domain lexicon json path")
    parser.add_argument("--out", default="data/processed/evidence_index.jsonl", help="Output index jsonl path")
    parser.add_argument("--max-phrases-per-domain", type=int, default=300)
    parser.add_argument("--max-snippets-per-phrase", type=int, default=3)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    lexicon_path = Path(args.lexicon)
    out_path = Path(args.out)

    if not corpus_path.exists():
        raise SystemExit(f"Missing corpus: {corpus_path}")
    if not lexicon_path.exists():
        raise SystemExit(f"Missing lexicon: {lexicon_path}")

    raw_lex = json.loads(lexicon_path.read_text(encoding="utf-8"))
    by_domain: dict[str, list[str]] = {}
    for domain, node in raw_lex.items():
        terms: list[str] = []
        if isinstance(node, list):
            terms = [str(x) for x in node]
        elif isinstance(node, dict):
            terms = [str(x) for x in node.get("style", [])]
            if not terms:
                terms = [str(x) for x in node.get("top_terms", [])]
        seen: set[str] = set()
        dedup: list[str] = []
        for term in terms:
            norm = _normalize(term)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            dedup.append(term.strip())
        by_domain[str(domain).upper()] = dedup[: max(1, int(args.max_phrases_per_domain))]

    df = pd.read_parquet(corpus_path)
    text_col = "text" if "text" in df.columns else "abstract" if "abstract" in df.columns else None
    if text_col is None:
        raise SystemExit("Corpus parquet must include `text` or `abstract` column")

    doc_id_col = "doc_id" if "doc_id" in df.columns else None
    source_col = "source" if "source" in df.columns else None

    rows_out: list[dict[str, str]] = []

    for domain, phrases in by_domain.items():
        for phrase in phrases:
            hits = 0
            for _, row in df.iterrows():
                text = str(row.get(text_col, "") or "")
                if not text:
                    continue
                if not _has_all_tokens(text, phrase):
                    continue

                snippet = text.strip()
                if len(snippet) > 220:
                    snippet = snippet[:217].rstrip() + "..."

                rows_out.append(
                    {
                        "key": f"{domain}::{_normalize(phrase)}",
                        "tgt": domain,
                        "phrase": _normalize(phrase),
                        "snippet": snippet,
                        "doc_id": str(row.get(doc_id_col, "")) if doc_id_col else "",
                        "source": str(row.get(source_col, "")) if source_col else "",
                    }
                )
                hits += 1
                if hits >= max(1, int(args.max_snippets_per_phrase)):
                    break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(out_path.resolve()), "rows": len(rows_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

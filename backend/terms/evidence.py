from __future__ import annotations

import re

import pandas as pd

WS_RE = re.compile(r"\s+")
DEBUG_RE = re.compile(r"updatedgpt|native=|domain-specific concept", re.IGNORECASE)


def _clean(text: str) -> str:
    text = DEBUG_RE.sub(" ", text or "")
    return WS_RE.sub(" ", text).strip()


def _window_around_phrase(text: str, phrase: str, max_words: int = 25) -> str:
    text = _clean(text)
    if not text:
        return ""

    t_low = text.lower()
    p_low = phrase.lower().strip()
    idx = t_low.find(p_low)
    if idx < 0:
        words = text.split()
        return " ".join(words[:max_words])

    left = text[:idx]
    right = text[idx + len(phrase) :]
    left_words = left.split()
    right_words = right.split()

    keep_left = max(0, max_words // 2)
    keep_right = max_words - min(len(left_words), keep_left)

    snippet_words = left_words[-keep_left:] + [text[idx : idx + len(phrase)]] + right_words[: max(0, keep_right - 1)]
    return " ".join(snippet_words[:max_words])


def find_evidence_snippets(
    corpus_df: pd.DataFrame,
    tgt: str,
    phrase: str,
    max_hits: int = 2,
) -> list[dict[str, str]]:
    if corpus_df.empty or not phrase.strip():
        return []

    mask = corpus_df["domain"].astype(str).str.upper() == tgt.upper()
    sub = corpus_df.loc[mask].copy()
    if sub.empty:
        return []

    col = "abstract" if "abstract" in sub.columns else "text"
    phrase_low = phrase.lower()
    has_phrase = sub[col].astype(str).str.lower().str.contains(re.escape(phrase_low), regex=True)
    matched = sub.loc[has_phrase].head(max_hits)

    out: list[dict[str, str]] = []
    for _, row in matched.iterrows():
        body = str(row.get(col, ""))
        snippet = _window_around_phrase(body, phrase, max_words=25)
        if not snippet:
            continue
        out.append(
            {
                "snippet": snippet,
                "doc_id": str(row.get("id", "")),
                "source": str(row.get("source", "")),
            }
        )
    return out

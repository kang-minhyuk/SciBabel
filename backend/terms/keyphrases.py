from __future__ import annotations

import re

from terms.clean import clean_text_for_mining
from terms.stoplist import load_all_stoplists, load_stoplist

_STOP = load_all_stoplists()
_DEBUG = load_stoplist("debug_artifacts.txt")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
VERB_LIKE = {"use", "uses", "using", "reduce", "reduces", "preserve", "preserving", "improve", "improves"}
CONNECTORS = {"to", "while", "under", "in", "for", "with", "and", "or"}


def _contains_debug(text: str) -> bool:
    low = text.lower()
    return any(d and d in low for d in _DEBUG)


def _stopword_only(text: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if not toks:
        return True
    return all(t in _STOP for t in toks)


def _low_quality(text: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return any(t in VERB_LIKE or t in CONNECTORS for t in toks)


def _find_spans(text: str, phrase: str, max_occurrences: int = 2) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    low = text.lower()
    p = phrase.lower().strip()
    if not p:
        return spans
    start = 0
    while len(spans) < max_occurrences:
        idx = low.find(p, start)
        if idx < 0:
            break
        end = idx + len(p)
        left_ok = idx == 0 or not low[idx - 1].isalnum()
        right_ok = end >= len(low) or not low[end].isalnum()
        if left_ok and right_ok:
            spans.append((idx, end))
        start = idx + 1
    return spans


def extract_yake_keyphrases_with_spans(text: str, top_k: int = 20) -> list[dict[str, object]]:
    cleaned = clean_text_for_mining(text)
    if not cleaned:
        return []

    import importlib

    yake = importlib.import_module("yake")

    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
    phrases = extractor.extract_keywords(cleaned)

    out: list[dict[str, object]] = []
    for phrase, _score in phrases:
        p = str(phrase).strip()
        if len(p) < 3:
            continue
        if _contains_debug(p) or _stopword_only(p) or _low_quality(p):
            continue
        for s, e in _find_spans(cleaned, p, max_occurrences=2):
            out.append({"term": cleaned[s:e], "start": int(s), "end": int(e), "source": "yake"})
    return out

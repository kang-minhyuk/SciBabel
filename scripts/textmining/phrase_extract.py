from __future__ import annotations

import os
import re

from common import clean_text_for_mining
from stoplist import load_all_stoplists, load_stoplist

_STOP = load_all_stoplists()
_DEBUG = load_stoplist("debug_artifacts.txt")
_NLP = None
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
VERB_LIKE = {"use", "uses", "using", "reduce", "reduces", "preserve", "preserving", "improve", "improves"}
CONNECTORS = {"to", "while", "under", "in", "for", "with", "and", "or"}
TECH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z]{1,4}\(\d+(?:,\d+)*\)-[A-Za-z]+(?:-[A-Za-z]+)*\b"),
    re.compile(r"\bk-space\b", re.IGNORECASE),
]


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    import importlib

    spacy = importlib.import_module("spacy")

    model = os.getenv("SPACY_MODEL", "en_core_web_sm")
    try:
        _NLP = spacy.load(model)
    except Exception:
        _NLP = spacy.blank("en")
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


def normalize_phrase(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip().lower()
    return t.strip(".,;:!?\"'`()[]{}")


def _contains_debug(text: str) -> bool:
    low = text.lower()
    return any(d and d in low for d in _DEBUG)


def _stopword_only(text: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if not toks:
        return True
    return all(t in _STOP for t in toks)


def valid_phrase(text: str) -> bool:
    norm = normalize_phrase(text)
    if len(norm) < 3:
        return False
    if _contains_debug(norm):
        return False
    if _stopword_only(norm):
        return False
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(norm)]
    if any(t in VERB_LIKE or t in CONNECTORS for t in toks):
        return False
    return True


def extract_phrases(text: str, top_k: int = 20) -> list[str]:
    cleaned = clean_text_for_mining(text)
    if not cleaned:
        return []

    nlp = _get_nlp()
    doc = nlp(cleaned)

    terms: list[str] = []
    for ent in getattr(doc, "ents", []):
        t = cleaned[ent.start_char : ent.end_char]
        if valid_phrase(t):
            terms.append(normalize_phrase(t))

    if hasattr(doc, "noun_chunks"):
        try:
            for nc in doc.noun_chunks:
                if any(tok.pos_ == "VERB" for tok in nc):
                    continue
                if len(nc) > 0 and nc[-1].pos_ not in {"NOUN", "PROPN"}:
                    continue
                t = cleaned[nc.start_char : nc.end_char]
                if valid_phrase(t):
                    terms.append(normalize_phrase(t))
        except Exception:
            pass

    for tok in doc:
        if tok.dep_ != "compound":
            continue
        left = min(tok.i, tok.head.i)
        right = max(tok.i, tok.head.i)
        span = doc[left : right + 1]
        t = cleaned[span.start_char : span.end_char]
        if valid_phrase(t):
            terms.append(normalize_phrase(t))

    for tok in doc:
        if tok.dep_ != "amod":
            continue
        head = tok.head
        if getattr(head, "pos_", "") not in {"NOUN", "PROPN"}:
            continue
        left = min(tok.i, head.i)
        right = max(tok.i, head.i)
        span = doc[left : right + 1]
        t = cleaned[span.start_char : span.end_char]
        if valid_phrase(t):
            terms.append(normalize_phrase(t))

    for patt in TECH_PATTERNS:
        for m in patt.finditer(cleaned):
            t = cleaned[m.start() : m.end()]
            if valid_phrase(t):
                terms.append(normalize_phrase(t))

    matches = list(TOKEN_RE.finditer(cleaned))
    for i in range(0, max(0, len(matches) - 1)):
        w1 = matches[i].group(0).lower()
        w2 = matches[i + 1].group(0).lower()
        if w1 in _STOP or w2 in _STOP:
            continue
        if w1 in VERB_LIKE or w2 in VERB_LIKE:
            continue
        if matches[i].end() + 1 != matches[i + 1].start():
            continue
        terms.append(normalize_phrase(cleaned[matches[i].start() : matches[i + 1].end()]))
        if i + 2 < len(matches):
            w3 = matches[i + 2].group(0).lower()
            if w3 in _STOP or w3 in VERB_LIKE:
                continue
            if matches[i + 1].end() + 1 != matches[i + 2].start():
                continue
            terms.append(normalize_phrase(cleaned[matches[i].start() : matches[i + 2].end()]))

    import importlib

    yake = importlib.import_module("yake")
    kw = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
    for phrase, _ in kw.extract_keywords(cleaned):
        p = normalize_phrase(str(phrase))
        if valid_phrase(p):
            terms.append(p)

    # deterministic dedup, preserve first occurrence
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

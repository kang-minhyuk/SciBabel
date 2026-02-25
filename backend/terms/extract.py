from __future__ import annotations

import os
import re
from dataclasses import dataclass

from terms.clean import clean_text_for_mining
from terms.keyphrases import extract_yake_keyphrases_with_spans
from terms.stoplist import load_all_stoplists, load_stoplist

_SOURCE_RANK = {
    "spacy_entity": 3,
    "spacy_noun_chunk": 2,
    "spacy_compound": 1,
    "yake": 0,
}

_NLP = None
_STOP_ALL = load_all_stoplists()
_DEBUG = load_stoplist("debug_artifacts.txt")
_ACADEMIC = load_stoplist("academic_stopwords.txt")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
VERB_LIKE = {"use", "uses", "using", "reduce", "reduces", "preserve", "preserving", "improve", "improves"}
BASIC_CONNECTORS = {"to", "while", "under", "in", "for", "with", "and", "or"}
TECH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z]{1,4}\(\d+(?:,\d+)*\)-[A-Za-z]+(?:-[A-Za-z]+)*\b"),
    re.compile(r"\bk-space\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class SpanTerm:
    term: str
    start: int
    end: int
    source: str


def set_nlp(nlp_obj: object) -> None:
    global _NLP
    _NLP = nlp_obj


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    import importlib

    spacy = importlib.import_module("spacy")

    model = os.getenv("SPACY_MODEL", "en_core_web_sm")
    default_env = "production" if os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"} else "dev"
    env = os.getenv("SCIBABEL_ENV", default_env).strip().lower()
    prefer_blank = env == "production" and os.getenv("SPACY_LOAD_MODEL_IN_PROD", "false").strip().lower() not in {"1", "true", "yes", "on"}
    try:
        if prefer_blank:
            _NLP = spacy.blank("en")
            if "sentencizer" not in _NLP.pipe_names:
                _NLP.add_pipe("sentencizer")
        else:
            _NLP = spacy.load(model)
    except Exception:
        _NLP = spacy.blank("en")
        if "sentencizer" not in _NLP.pipe_names:
            _NLP.add_pipe("sentencizer")
    return _NLP


def _normalize_phrase(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip().lower()
    return t.strip(".,;:!?\"'`()[]{}")


def _is_stopword_only(text: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if not toks:
        return True
    return all(t in _STOP_ALL for t in toks)


def _contains_debug(text: str) -> bool:
    low = text.lower()
    return any(d and d in low for d in _DEBUG)


def _valid_candidate(text: str) -> bool:
    norm = _normalize_phrase(text)
    if len(norm) < 3:
        return False
    if _contains_debug(norm):
        return False
    if norm in _ACADEMIC:
        return False
    if _is_stopword_only(norm):
        return False
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(norm)]
    if any(t in VERB_LIKE for t in toks):
        return False
    if any(t in BASIC_CONNECTORS for t in toks):
        return False
    return True


def _refine_span(cleaned: str, start: int, end: int) -> tuple[int, int, str]:
    segment = cleaned[start:end]
    toks = list(TOKEN_RE.finditer(segment))
    if not toks:
        return start, end, segment

    left = 0
    right = len(toks) - 1
    while left <= right and toks[left].group(0).lower() in (_STOP_ALL | VERB_LIKE | BASIC_CONNECTORS):
        left += 1
    while right >= left and toks[right].group(0).lower() in (_STOP_ALL | VERB_LIKE | BASIC_CONNECTORS):
        right -= 1
    if left > right:
        return start, end, segment

    new_start = start + toks[left].start()
    new_end = start + toks[right].end()
    return new_start, new_end, cleaned[new_start:new_end]


def _collect_spacy_candidates(text: str) -> list[SpanTerm]:
    nlp = _get_nlp()
    cleaned = clean_text_for_mining(text)
    doc = nlp(cleaned)

    items: list[SpanTerm] = []
    for ent in getattr(doc, "ents", []):
        term = cleaned[ent.start_char : ent.end_char]
        s, e, term = _refine_span(cleaned, ent.start_char, ent.end_char)
        if _valid_candidate(term):
            items.append(SpanTerm(term=term, start=s, end=e, source="spacy_entity"))

    if hasattr(doc, "noun_chunks"):
        try:
            for nc in doc.noun_chunks:
                if any(tok.pos_ == "VERB" for tok in nc):
                    continue
                if len(nc) > 0 and nc[-1].pos_ not in {"NOUN", "PROPN"}:
                    continue
                term = cleaned[nc.start_char : nc.end_char]
                s, e, term = _refine_span(cleaned, nc.start_char, nc.end_char)
                if _valid_candidate(term):
                    items.append(SpanTerm(term=term, start=s, end=e, source="spacy_noun_chunk"))
        except Exception:
            pass

    for tok in doc:
        if tok.dep_ != "compound":
            continue
        left = min(tok.i, tok.head.i)
        right = max(tok.i, tok.head.i)
        span = doc[left : right + 1]
        term = cleaned[span.start_char : span.end_char]
        s, e, term = _refine_span(cleaned, span.start_char, span.end_char)
        if _valid_candidate(term):
            items.append(SpanTerm(term=term, start=s, end=e, source="spacy_compound"))

    # adjective+noun technical phrases (e.g., sparse attention)
    for tok in doc:
        if tok.dep_ != "amod":
            continue
        head = tok.head
        if getattr(head, "pos_", "") not in {"NOUN", "PROPN"}:
            continue
        left = min(tok.i, head.i)
        right = max(tok.i, head.i)
        span = doc[left : right + 1]
        term = cleaned[span.start_char : span.end_char]
        s, e, term = _refine_span(cleaned, span.start_char, span.end_char)
        if _valid_candidate(term):
            items.append(SpanTerm(term=term, start=s, end=e, source="spacy_noun_chunk"))

    # regex technical tokens to preserve forms like SE(3)-equivariant
    for patt in TECH_PATTERNS:
        for m in patt.finditer(cleaned):
            term = cleaned[m.start() : m.end()]
            s, e, term = _refine_span(cleaned, m.start(), m.end())
            if _valid_candidate(term):
                items.append(SpanTerm(term=term, start=s, end=e, source="spacy_entity"))

    # heuristic adjacent bigram/trigram phrases for technical style (deterministic fallback)
    matches = list(TOKEN_RE.finditer(cleaned))
    for i in range(0, max(0, len(matches) - 1)):
        w1 = matches[i].group(0).lower()
        w2 = matches[i + 1].group(0).lower()
        if w1 in _STOP_ALL or w2 in _STOP_ALL:
            continue
        if w1 in VERB_LIKE or w2 in VERB_LIKE:
            continue
        if matches[i].end() + 1 != matches[i + 1].start():
            continue
        s = matches[i].start()
        e = matches[i + 1].end()
        s, e, term = _refine_span(cleaned, s, e)
        if _valid_candidate(term):
            items.append(SpanTerm(term=term, start=s, end=e, source="spacy_noun_chunk"))

        if i + 2 < len(matches):
            w3 = matches[i + 2].group(0).lower()
            if w3 in _STOP_ALL or w3 in VERB_LIKE:
                continue
            if matches[i + 1].end() + 1 != matches[i + 2].start():
                continue
            s3 = matches[i].start()
            e3 = matches[i + 2].end()
            s3, e3, term3 = _refine_span(cleaned, s3, e3)
            if _valid_candidate(term3):
                items.append(SpanTerm(term=term3, start=s3, end=e3, source="spacy_noun_chunk"))

    return items


def _overlap_ratio(a: SpanTerm, b: SpanTerm) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    shorter = min(a.end - a.start, b.end - b.start)
    return inter / max(1, shorter)


def _is_technical(text: str) -> int:
    score = 0
    if re.search(r"[\-()]", text):
        score += 1
    if re.search(r"\b[A-Z]{2,}\b", text):
        score += 1
    return score


def _dedup_merge(candidates: list[SpanTerm]) -> list[SpanTerm]:
    ranked = sorted(
        candidates,
        key=lambda s: (-_is_technical(s.term), -(s.end - s.start), -_SOURCE_RANK.get(s.source, 0), s.start),
    )
    kept: list[SpanTerm] = []
    for c in ranked:
        skip = False
        for k in kept:
            if _overlap_ratio(c, k) <= 0.5:
                continue
            len_c = c.end - c.start
            len_k = k.end - k.start
            if len_k >= len_c:
                skip = True
                break
            if c.source == "yake" and len_c > len_k:
                continue
            if _SOURCE_RANK.get(k.source, 0) >= _SOURCE_RANK.get(c.source, 0):
                skip = True
                break
        if not skip:
            kept.append(c)

    uniq: list[SpanTerm] = []
    seen: set[tuple[str, int, int]] = set()
    for k in sorted(kept, key=lambda x: x.start):
        key = (_normalize_phrase(k.term), k.start, k.end)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(k)
    return uniq


def extract_terms(text: str, max_terms: int = 12) -> list[dict[str, object]]:
    if not text.strip():
        return []

    spacy_terms = _collect_spacy_candidates(text)
    yake_terms = [
        SpanTerm(term=str(x["term"]), start=int(x["start"]), end=int(x["end"]), source="yake")
        for x in extract_yake_keyphrases_with_spans(text, top_k=20)
    ]
    merged = _dedup_merge(spacy_terms + yake_terms)

    prioritized = sorted(
        merged,
        key=lambda s: (-_is_technical(s.term), -(s.end - s.start), -_SOURCE_RANK.get(s.source, 0), s.start),
    )[:max_terms]
    prioritized.sort(key=lambda s: s.start)

    return [{"term": s.term, "start": int(s.start), "end": int(s.end), "source": s.source} for s in prioritized]


def extract_terms_with_spans(text: str, lexicon_phrases: list[str] | None = None, max_terms: int = 24) -> list[dict[str, object]]:
    _ = lexicon_phrases
    return extract_terms(text=text, max_terms=max_terms)

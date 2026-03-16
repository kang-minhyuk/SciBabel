from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from terms.canonicalize import canonicalize_span
from terms.clean import clean_text_for_mining
from terms.keyphrases import extract_yake_keyphrases_with_spans
from terms.score import concept_likeness_score
from terms.stoplist import load_all_stoplists, load_stoplist

_SOURCE_RANK = {
    "spacy_entity": 3,
    "spacy_noun_chunk": 2,
    "spacy_compound": 1,
    "yake": 0,
}

_NLP = None
_NLP_LOCK = threading.RLock()
_STOP_ALL = load_all_stoplists()
_DEBUG = load_stoplist("debug_artifacts.txt")
_ACADEMIC = load_stoplist("academic_stopwords.txt")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
VERB_LIKE = {
    "use",
    "uses",
    "using",
    "reduce",
    "reduces",
    "preserve",
    "preserving",
    "improve",
    "improves",
    "optimize",
    "optimizes",
    "derive",
    "derives",
    "estimating",
    "estimate",
    "estimates",
    "train",
    "trains",
    "training",
}
BASIC_CONNECTORS = {"to", "while", "under", "in", "for", "with", "and", "or", "on", "of", "by", "near"}
LEADING_STRIP = {
    "the",
    "our",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "we",
    "i",
    "they",
    "by",
    "of",
    "on",
    "in",
    "with",
    "under",
    "for",
    "to",
}
TRAILING_STRIP = {
    "of",
    "on",
    "in",
    "by",
    "with",
    "under",
    "to",
    "for",
    "from",
    "and",
    "or",
    "the",
    "a",
    "an",
}
GENERIC_ACADEMIC = {
    "method",
    "methods",
    "approach",
    "approaches",
    "system",
    "systems",
    "model",
    "models",
    "pattern",
    "analysis",
    "study",
    "result",
    "results",
}
GENERIC_SINGLE_BLOCK = {
    "transformer",
    "model",
    "method",
    "system",
    "approach",
}
TECH_SINGLE_SUFFIXES = ("ity", "tion", "sion", "lysis", "kinetics", "dynamics")
TECH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z]{1,4}\(\d+(?:,\d+)*\)-[A-Za-z]+(?:-[A-Za-z]+)*\b"),
    re.compile(r"\bk-space\b", re.IGNORECASE),
]
ACRONYM_RE = re.compile(r"^[A-Z]{2,}(?:\d+)?$")
SYMBOLIC_RE = re.compile(r"[()\[\]{}\/+*]")
FRAGMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:optimize|optimizes|optimizing|train|trains|training)\s+(?:a|an|the)\s+\w+$", re.IGNORECASE),
    re.compile(r"^memory\s+cost\s+on$", re.IGNORECASE),
    re.compile(r"^on\s+long\s+sequences$", re.IGNORECASE),
    re.compile(r"^(?:by\s+an|of\s+the)$", re.IGNORECASE),
    re.compile(r"\b(?:is\s+characterized|characterized\s+by)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class SpanTerm:
    term: str
    surface_term: str
    canonical_term: str
    start: int
    end: int
    source: str
    concept_score: float = 0.0
    noun_headed: bool = False
    noun_adj_compound: bool = False
    leading_pos: str = ""
    leading_lemma: str = ""
    overlapped_by_stronger: bool = False


def set_nlp(nlp_obj: object) -> None:
    global _NLP
    _NLP = nlp_obj


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    with _NLP_LOCK:
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


def _is_yake_enabled() -> bool:
    default_env = "production" if os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"} else "dev"
    env = os.getenv("SCIBABEL_ENV", default_env).strip().lower()
    default = "false" if env == "production" else "true"
    return os.getenv("YAKE_ENABLED", default).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_phrase(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip().lower()
    return t.strip(".,;:!?\"'`()[]{}")


def _tokenize_with_span(text: str) -> list[re.Match[str]]:
    return list(TOKEN_RE.finditer(text))


def _is_acronym_or_symbolic(term: str) -> bool:
    clean = term.strip()
    if not clean:
        return False
    if ACRONYM_RE.match(clean):
        return True
    if SYMBOLIC_RE.search(clean):
        return True
    low = clean.lower()
    if any(x in low for x in ["o(", "se(", "so(", "su(", "log("]):
        return True
    if "k-space" in clean.lower():
        return True
    return False


def _has_fragment_pattern(term: str) -> bool:
    t = _normalize_phrase(term)
    if not t:
        return True
    if any(p.search(t) for p in FRAGMENT_PATTERNS):
        return True
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(t)]
    if len(toks) >= 2 and toks[0] in VERB_LIKE:
        return True
    if len(toks) >= 2 and toks[0] in BASIC_CONNECTORS:
        return True
    if len(toks) >= 3 and any(tok in BASIC_CONNECTORS for tok in toks[1:-1]):
        return True
    if toks[:2] in (["by", "an"], ["of", "the"], ["on", "long"]):
        return True
    return False


def _is_technical_single_word(term: str) -> bool:
    t = _normalize_phrase(term)
    if not t:
        return False
    if t in GENERIC_SINGLE_BLOCK:
        return False
    if _is_acronym_or_symbolic(term):
        return True
    if any(t.endswith(sfx) for sfx in TECH_SINGLE_SUFFIXES):
        return True
    return False


def _span_linguistic_signals(doc: object | None, start: int, end: int) -> tuple[str, str, bool, bool]:
    if doc is None:
        return "", "", False, False
    try:
        span = doc.char_span(start, end, alignment_mode="expand")
        if span is None or len(span) == 0:
            return "", "", False, False
        lead = span[0]
        head = span[-1]
        leading_pos = str(getattr(lead, "pos_", ""))
        leading_lemma = str(getattr(lead, "lemma_", "") or getattr(lead, "text", "")).lower()
        noun_headed = str(getattr(head, "pos_", "")) in {"NOUN", "PROPN"}
        noun_adj_compound = all(str(getattr(tok, "pos_", "")) in {"NOUN", "PROPN", "ADJ", "NUM"} for tok in span)
        return leading_pos, leading_lemma, noun_headed, noun_adj_compound
    except Exception:
        return "", "", False, False


def normalize_phrase_span(text: str, start: int, end: int) -> Optional[tuple[int, int, str]]:
    segment = text[start:end]
    toks = _tokenize_with_span(segment)
    if not toks:
        return None

    left = 0
    right = len(toks) - 1
    while left <= right and toks[left].group(0).lower() in LEADING_STRIP:
        left += 1
    while right >= left and toks[right].group(0).lower() in (TRAILING_STRIP | VERB_LIKE | BASIC_CONNECTORS):
        right -= 1
    if left > right:
        return None

    new_start = start + toks[left].start()
    new_end = start + toks[right].end()
    cleaned = text[new_start:new_end].strip(" ,.;:")
    if not cleaned:
        return None

    clean_tokens = [m.group(0) for m in TOKEN_RE.finditer(cleaned)]
    if not clean_tokens:
        return None
    token_lowers = [t.lower() for t in clean_tokens]

    stop_ratio = sum(1 for t in token_lowers if t in _STOP_ALL) / max(1, len(token_lowers))
    stop_cap = 0.6 if len(token_lowers) >= 2 else 0.4
    if stop_ratio > stop_cap:
        return None

    if len(clean_tokens) == 1 and not _is_technical_single_word(cleaned):
        return None
    if len(clean_tokens) > 5 and not _is_acronym_or_symbolic(cleaned):
        return None

    generic_ratio = sum(1 for t in token_lowers if t in GENERIC_ACADEMIC) / max(1, len(token_lowers))
    if generic_ratio >= 0.6:
        return None

    if _has_fragment_pattern(cleaned):
        return None

    return new_start, new_end, cleaned


def _is_stopword_only(text: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if not toks:
        return True
    return all(t in _STOP_ALL for t in toks)


def _contains_debug(text: str) -> bool:
    low = text.lower()
    return any(d and d in low for d in _DEBUG)


def _valid_candidate(
    text: str,
    *,
    leading_pos: str = "",
    leading_lemma: str = "",
    noun_headed: bool = False,
    noun_adj_compound: bool = False,
) -> bool:
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
    if _has_fragment_pattern(norm):
        return False
    if len(toks) == 1 and not _is_technical_single_word(text):
        return False
    if len(toks) > 5 and not _is_acronym_or_symbolic(text):
        return False
    stop_ratio = sum(1 for t in toks if t in _STOP_ALL) / max(1, len(toks))
    stop_cap = 0.6 if len(toks) >= 2 else 0.4
    if stop_ratio > stop_cap:
        return False
    concept_score = concept_likeness_score(
        text,
        leading_pos=leading_pos,
        leading_lemma=leading_lemma,
        noun_headed=noun_headed,
        noun_adj_compound=noun_adj_compound,
        lexicon_support=False,
        evidence_support=False,
        stronger_overlap=False,
    )
    if concept_score < 0.34:
        return False
    return True


def _refine_span(cleaned: str, start: int, end: int) -> tuple[int, int, str]:
    normalized = normalize_phrase_span(cleaned, start, end)
    if normalized is None:
        return start, end, cleaned[start:end]
    return normalized


def _collect_spacy_candidates(text: str) -> list[SpanTerm]:
    cleaned = clean_text_for_mining(text)
    doc = None
    try:
        nlp = _get_nlp()
        if nlp is not None:
            doc = nlp(cleaned)
    except Exception:
        doc = None

    items: list[SpanTerm] = []

    def _append_candidate(s: int, e: int, source: str) -> None:
        s2, e2, term = _refine_span(cleaned, s, e)
        leading_pos, leading_lemma, noun_headed, noun_adj_compound = _span_linguistic_signals(doc, s2, e2)
        canonical_terms = canonicalize_span(term)
        if not canonical_terms and _valid_candidate(
            term,
            leading_pos=leading_pos,
            leading_lemma=leading_lemma,
            noun_headed=noun_headed,
            noun_adj_compound=noun_adj_compound,
        ):
            canonical_terms = [term]
        for canonical in canonical_terms:
            if not _valid_candidate(
                canonical,
                leading_pos=leading_pos,
                leading_lemma=leading_lemma,
                noun_headed=noun_headed,
                noun_adj_compound=noun_adj_compound,
            ):
                continue
            concept_score = concept_likeness_score(
                canonical,
                leading_pos=leading_pos,
                leading_lemma=leading_lemma,
                noun_headed=noun_headed,
                noun_adj_compound=noun_adj_compound,
                lexicon_support=False,
                evidence_support=False,
                stronger_overlap=False,
            )
            items.append(
                SpanTerm(
                    term=canonical,
                    surface_term=term,
                    canonical_term=canonical,
                    start=s2,
                    end=e2,
                    source=source,
                    concept_score=concept_score,
                    noun_headed=noun_headed,
                    noun_adj_compound=noun_adj_compound,
                    leading_pos=leading_pos,
                    leading_lemma=leading_lemma,
                )
            )

    if doc is not None:
        for ent in getattr(doc, "ents", []):
            _append_candidate(ent.start_char, ent.end_char, "spacy_entity")

        try:
            for nc in doc.noun_chunks:
                if any(tok.pos_ == "VERB" for tok in nc):
                    continue
                if len(nc) > 0 and nc[-1].pos_ not in {"NOUN", "PROPN"}:
                    continue
                _append_candidate(nc.start_char, nc.end_char, "spacy_noun_chunk")
        except Exception:
            pass

        for tok in doc:
            if tok.dep_ != "compound":
                continue
            left = min(tok.i, tok.head.i)
            right = max(tok.i, tok.head.i)
            span = doc[left : right + 1]
            _append_candidate(span.start_char, span.end_char, "spacy_compound")

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
            _append_candidate(span.start_char, span.end_char, "spacy_noun_chunk")

    # regex technical tokens to preserve forms like SE(3)-equivariant
    for patt in TECH_PATTERNS:
        for m in patt.finditer(cleaned):
            _append_candidate(m.start(), m.end(), "spacy_entity")

    # heuristic adjacent bigram/trigram phrases for technical style (deterministic fallback)
    matches = list(TOKEN_RE.finditer(cleaned))
    allowed_stop_heads = {"model", "models"}
    for i in range(0, max(0, len(matches) - 1)):
        w1 = matches[i].group(0).lower()
        w2 = matches[i + 1].group(0).lower()
        if w1 in LEADING_STRIP or w1 in BASIC_CONNECTORS:
            continue
        if (w2 in BASIC_CONNECTORS or w2 in TRAILING_STRIP) and w2 not in allowed_stop_heads:
            continue
        if w1 in VERB_LIKE or w2 in VERB_LIKE:
            continue
        if matches[i].end() + 1 != matches[i + 1].start():
            continue
        s = matches[i].start()
        e = matches[i + 1].end()
        _append_candidate(s, e, "spacy_noun_chunk")

        if i + 2 < len(matches):
            w3 = matches[i + 2].group(0).lower()
            if ((w3 in BASIC_CONNECTORS or w3 in TRAILING_STRIP) and w3 not in allowed_stop_heads) or w3 in VERB_LIKE:
                continue
            if matches[i + 1].end() + 1 != matches[i + 2].start():
                continue
            s3 = matches[i].start()
            e3 = matches[i + 2].end()
            _append_candidate(s3, e3, "spacy_noun_chunk")

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
        key=lambda s: (-int(s.noun_headed), -_is_technical(s.term), -(s.end - s.start), -round(float(s.concept_score), 4), -_SOURCE_RANK.get(s.source, 0), s.start),
    )
    kept: list[SpanTerm] = []
    for c in ranked:
        skip = False
        for k in kept:
            if c.start == k.start and c.end == k.end and _normalize_phrase(c.canonical_term) != _normalize_phrase(k.canonical_term):
                continue
            overlap = _overlap_ratio(c, k)
            if overlap <= 0.5:
                continue
            len_c = c.end - c.start
            len_k = k.end - k.start
            if overlap >= 0.6 and len_k >= len_c and k.concept_score >= c.concept_score + 0.08:
                skip = True
                break
            if len_k >= len_c:
                skip = True
                break
            if c.source == "yake" and len_c > len_k:
                continue
            if len_k == len_c and _SOURCE_RANK.get(k.source, 0) >= _SOURCE_RANK.get(c.source, 0):
                skip = True
                break
        if not skip:
            kept.append(c)

    uniq: list[SpanTerm] = []
    seen: set[tuple[str, int, int]] = set()
    for k in sorted(kept, key=lambda x: x.start):
        key = (_normalize_phrase(k.canonical_term), k.start, k.end)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(k)
    return uniq


def extract_terms(text: str, max_terms: int = 12, yake_enabled: bool | None = None) -> list[dict[str, object]]:
    return extract_terms_profiled(text=text, max_terms=max_terms, yake_enabled=yake_enabled)["terms"]


def extract_terms_profiled(text: str, max_terms: int = 12, yake_enabled: bool | None = None) -> dict[str, object]:
    if not text.strip():
        return {
            "terms": [],
            "timings": {
                "spacy_extract_sec": 0.0,
                "yake_extract_sec": 0.0,
                "merge_dedupe_sec": 0.0,
            },
        }

    t0 = time.perf_counter()
    spacy_terms = _collect_spacy_candidates(text)
    t_spacy = time.perf_counter() - t0

    t0 = time.perf_counter()
    use_yake = _is_yake_enabled() if yake_enabled is None else bool(yake_enabled)
    if use_yake:
        yake_terms = []
        for x in extract_yake_keyphrases_with_spans(text, top_k=20):
            normalized = normalize_phrase_span(text, int(x["start"]), int(x["end"]))
            if normalized is None:
                continue
            s, e, term = normalized
            canonical_terms = canonicalize_span(term)
            if not canonical_terms and _valid_candidate(term):
                canonical_terms = [term]
            for canonical in canonical_terms:
                if not _valid_candidate(canonical):
                    continue
                concept_score = concept_likeness_score(canonical)
                yake_terms.append(
                    SpanTerm(
                        term=canonical,
                        surface_term=term,
                        canonical_term=canonical,
                        start=s,
                        end=e,
                        source="yake",
                        concept_score=concept_score,
                    )
                )
    else:
        yake_terms = []
    t_yake = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged = _dedup_merge(spacy_terms + yake_terms)
    t_merge = time.perf_counter() - t0

    prioritized = sorted(
        merged,
        key=lambda s: (-_is_technical(s.term), -(s.end - s.start), -_SOURCE_RANK.get(s.source, 0), s.start),
    )[:max_terms]
    prioritized.sort(key=lambda s: s.start)

    return {
        "terms": [
            {
                "term": s.term,
                "surface_term": s.surface_term,
                "canonical_term": s.canonical_term,
                "start": int(s.start),
                "end": int(s.end),
                "source": s.source,
                "concept_score": round(float(s.concept_score), 4),
                "noun_headed": bool(s.noun_headed),
                "noun_adj_compound": bool(s.noun_adj_compound),
                "leading_pos": s.leading_pos,
                "leading_lemma": s.leading_lemma,
                "overlapped_by_stronger": bool(s.overlapped_by_stronger),
            }
            for s in prioritized
        ],
        "timings": {
            "spacy_extract_sec": round(t_spacy, 4),
            "yake_extract_sec": round(t_yake, 4),
            "merge_dedupe_sec": round(t_merge, 4),
        },
    }


def extract_terms_with_spans(text: str, lexicon_phrases: list[str] | None = None, max_terms: int = 24) -> list[dict[str, object]]:
    _ = lexicon_phrases
    return extract_terms(text=text, max_terms=max_terms)

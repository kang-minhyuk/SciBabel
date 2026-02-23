from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")


@dataclass
class ScoreBreakdown:
    domain: float
    meaning: float
    lex: float


@dataclass
class RewardResult:
    total: float
    breakdown: ScoreBreakdown


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def jaccard_overlap(text_a: str, text_b: str) -> float:
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lexicon_coverage(candidate: str, tgt_lexicon: list[str]) -> float:
    cand_tokens = tokenize(candidate)
    if not tgt_lexicon:
        return 0.0

    normalized_lex = {term.lower() for term in tgt_lexicon}
    hits = len(cand_tokens & normalized_lex)
    return hits / max(1, len(normalized_lex))


def domain_probability(candidate: str, tgt: str, clf: Any) -> float:
    labels = list(getattr(clf, "classes_", []))
    if tgt not in labels:
        return 0.0

    probs = clf.predict_proba([candidate])[0]
    idx = labels.index(tgt)
    return float(probs[idx])


def compute_reward(
    source_text: str,
    candidate: str,
    tgt: str,
    clf: Any,
    lexicon_by_domain: dict[str, list[str]],
    w_domain: float = 0.5,
    w_meaning: float = 0.3,
    w_lex: float = 0.2,
) -> RewardResult:
    domain = domain_probability(candidate, tgt, clf)
    meaning = jaccard_overlap(source_text, candidate)
    lex = lexicon_coverage(candidate, lexicon_by_domain.get(tgt, []))

    total = (w_domain * domain) + (w_meaning * meaning) + (w_lex * lex)
    return RewardResult(total=total, breakdown=ScoreBreakdown(domain=domain, meaning=meaning, lex=lex))

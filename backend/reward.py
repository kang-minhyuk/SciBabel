from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")


@dataclass
class ScoreBreakdown:
    domain: float
    meaning: float
    lex: float
    semantic_sim: float
    copy_score: float
    copy_penalty: float
    lex_terms_hit: list[str]


@dataclass
class RewardResult:
    total: float
    breakdown: ScoreBreakdown
    eligible: bool
    discard_reason: str | None = None


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def ngram_set(text: str, max_n: int = 3) -> set[str]:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    out: set[str] = set(toks)
    n_tok = len(toks)
    for n in range(2, max_n + 1):
        for i in range(0, max(0, n_tok - n + 1)):
            out.add(" ".join(toks[i : i + n]))
    return out


def jaccard_overlap(text_a: str, text_b: str) -> float:
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rouge_l_copy_score(source_text: str, candidate: str) -> float:
    """Approximate copy score via ROUGE-L F1 over token sequence."""
    src = [t.lower() for t in TOKEN_RE.findall(source_text)]
    cand = [t.lower() for t in TOKEN_RE.findall(candidate)]
    if not src or not cand:
        return 0.0

    # LCS dynamic programming
    dp = [[0] * (len(cand) + 1) for _ in range(len(src) + 1)]
    for i in range(1, len(src) + 1):
        for j in range(1, len(cand) + 1):
            if src[i - 1] == cand[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    precision = lcs / len(cand)
    recall = lcs / len(src)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def lexicon_coverage(candidate: str, tgt_lexicon: list[str]) -> float:
    cand_tokens = tokenize(candidate)
    if not tgt_lexicon:
        return 0.0

    normalized_lex = {term.lower() for term in tgt_lexicon}
    hits = len(cand_tokens & normalized_lex)
    return hits / max(1, len(normalized_lex))


def lex_weighted_evidence(
    candidate: str,
    tgt: str,
    term_log_odds: dict[tuple[str, str], dict[str, float]] | None,
    tgt_lexicon: list[str],
    lex_score_clamp: float = 2.0,
) -> tuple[float, list[str]]:
    """Weighted lexical evidence from term log-odds, fallback to coverage."""
    cand_ngrams = ngram_set(candidate, max_n=3)

    if not term_log_odds:
        return lexicon_coverage(candidate, tgt_lexicon), []

    contributions: list[tuple[str, float]] = []
    for (domain, term), meta in term_log_odds.items():
        if domain != tgt:
            continue
        if not term or term not in cand_ngrams:
            continue
        z = float(meta.get("z", 0.0))
        delta = float(meta.get("delta", 0.0))
        ngram_len = int(meta.get("ngram_len", max(1, len(term.split()))))
        if z <= 0 and delta <= 0:
            continue

        # Make reward harder to game: prefer multi-word domain voice.
        ngram_weight = {1: 0.25, 2: 0.7, 3: 1.0}.get(ngram_len, 1.0)
        raw_score = max(0.0, min(z, 6.0)) * ngram_weight
        if raw_score > 0:
            contributions.append((term, float(raw_score)))

    if not contributions:
        return lexicon_coverage(candidate, tgt_lexicon), []

    contributions.sort(key=lambda x: x[1], reverse=True)
    lex_score = min(lex_score_clamp, sum(v for _, v in contributions))
    lex_terms = [f"{t}:{v:.2f}" for t, v in contributions[:8]]
    return float(lex_score), lex_terms


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
    semantic_similarity_fn: Callable[[str, str], float],
    term_log_odds: dict[tuple[str, str], dict[str, float]] | None = None,
    min_semantic_sim: float = 0.78,
    alpha_lex: float = 0.35,
    beta_copy: float = 0.5,
    copy_threshold: float = 0.86,
    lex_score_clamp: float = 2.0,
) -> RewardResult:
    domain = domain_probability(candidate, tgt, clf)
    semantic_sim = float(max(0.0, min(1.0, semantic_similarity_fn(source_text, candidate))))
    copy_score = rouge_l_copy_score(source_text, candidate)
    copy_penalty = max(0.0, copy_score - copy_threshold)

    lex, lex_terms_hit = lex_weighted_evidence(
        candidate=candidate,
        tgt=tgt,
        term_log_odds=term_log_odds,
        tgt_lexicon=lexicon_by_domain.get(tgt, []),
        lex_score_clamp=lex_score_clamp,
    )

    breakdown = ScoreBreakdown(
        domain=domain,
        meaning=semantic_sim,
        lex=lex,
        semantic_sim=semantic_sim,
        copy_score=copy_score,
        copy_penalty=copy_penalty,
        lex_terms_hit=lex_terms_hit,
    )

    if semantic_sim < min_semantic_sim:
        return RewardResult(
            total=-1e9,
            breakdown=breakdown,
            eligible=False,
            discard_reason=f"semantic_sim<{min_semantic_sim:.2f}",
        )

    total = domain + (alpha_lex * lex) - (beta_copy * copy_penalty)
    return RewardResult(total=total, breakdown=breakdown, eligible=True)

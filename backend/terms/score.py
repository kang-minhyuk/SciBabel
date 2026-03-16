from __future__ import annotations

import math
import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
STOP = {
    "the",
    "a",
    "an",
    "of",
    "on",
    "in",
    "by",
    "with",
    "under",
    "to",
    "for",
    "and",
    "or",
    "we",
    "our",
}
GENERIC_VERBS = {
    "use",
    "uses",
    "using",
    "reduce",
    "reduces",
    "optimize",
    "optimizes",
    "optimizing",
    "derive",
    "derives",
    "train",
    "trains",
    "training",
}
TECH_SINGLE_SUFFIXES = ("ity", "tion", "sion", "lysis", "genesis", "dynamics", "kinetics")


@dataclass
class TermScoreConfig:
    src_threshold: float = 0.35
    tgt_threshold: float = 0.45


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _normalize_term(term: str) -> str:
    t = re.sub(r"\s+", " ", term.strip().lower())
    t = t.strip(".,;:!?\"'`()[]{}")
    return t


def _variants(term: str) -> list[str]:
    base = _normalize_term(term)
    vals = [base]
    vals.append(re.sub(r"[-_/]+", " ", base))
    vals.append(re.sub(r"\s+", " ", vals[-1]).strip())
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _components(term: str) -> list[str]:
    parts = [_normalize_term(m.group(0)) for m in TOKEN_RE.finditer(term)]
    meaningful = [p for p in parts if p and p not in STOP]
    return meaningful


def _lookup_domain_z(term_stats: dict[str, dict[str, float]], domain: str, term: str) -> tuple[float, str]:
    dom = term_stats.get(domain, {})
    if not dom:
        return 0.0, "default"

    raw = term.strip().lower()
    if raw in dom:
        return float(dom[raw]), "exact"

    for v in _variants(term):
        if v in dom:
            return float(dom[v]), "normalized"

    comps = _components(term)
    comp_vals = [float(dom[c]) for c in comps if c in dom]
    if comp_vals:
        return sum(comp_vals) / len(comp_vals), "component_fallback"

    return 0.0, "default"


def _is_concept_like(term: str) -> bool:
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(term)]
    if not toks:
        return False
    if len(toks) == 1:
        one = toks[0]
        if one.upper() == one and len(one) >= 2:
            return True
        if any(ch in term for ch in ["(", ")", "-", "/"]):
            return True
        return False
    if len(toks) > 5:
        return False
    stop_ratio = sum(1 for t in toks if t in STOP) / max(1, len(toks))
    return stop_ratio <= 0.4


def concept_likeness_score(
    term: str,
    *,
    leading_pos: str = "",
    leading_lemma: str = "",
    noun_headed: bool = False,
    noun_adj_compound: bool = False,
    lexicon_support: bool = False,
    evidence_support: bool = False,
    stronger_overlap: bool = False,
) -> float:
    """Return [0,1] concept-likeness for a span.

    Higher scores indicate noun-centered technical concepts over generic fragments.
    """
    toks = [m.group(0).lower() for m in TOKEN_RE.finditer(term)]
    if not toks:
        return 0.0

    score = 0.35
    stop_ratio = sum(1 for t in toks if t in STOP) / max(1, len(toks))
    score += max(-0.35, 0.22 - stop_ratio * 0.6)

    lead = leading_lemma.strip().lower() or toks[0]
    if len(toks) > 1 and lead in GENERIC_VERBS:
        score -= 0.45
    if len(toks) > 1 and toks[0] in STOP:
        score -= 0.30

    if leading_pos in {"NOUN", "PROPN", "ADJ"}:
        score += 0.12
    elif leading_pos == "VERB":
        score -= 0.18

    if noun_headed:
        score += 0.14
    if noun_adj_compound:
        score += 0.12

    if len(toks) == 1:
        t = toks[0]
        if any(t.endswith(suf) for suf in TECH_SINGLE_SUFFIXES):
            score += 0.12
        else:
            score -= 0.35
        if leading_pos == "ADJ":
            score -= 0.25
    elif len(toks) > 5:
        score -= 0.20

    if lexicon_support:
        score += 0.15
    if evidence_support:
        score += 0.10
    if stronger_overlap:
        score -= 0.22

    return max(0.0, min(1.0, round(score, 4)))


def _familiarity(
    term: str,
    tgt: str,
    term_stats: dict[str, dict[str, float]],
    lexicon_lower_by_domain: dict[str, set[str]],
) -> tuple[float, str]:
    z, source = _lookup_domain_z(term_stats, tgt, term)
    if source != "default":
        return float(_sigmoid(z / 2.0)), source

    norm = _normalize_term(term)
    if norm in lexicon_lower_by_domain.get(tgt, set()):
        return 0.6, "normalized"
    return 0.1, "default"


def _distinctiveness(
    term: str,
    src: str,
    tgt: str,
    all_domains: list[str],
    term_stats: dict[str, dict[str, float]],
    lexicon_lower_by_domain: dict[str, set[str]],
) -> tuple[float, str]:
    src_z, src_source = _lookup_domain_z(term_stats, src, term)
    if src_source == "default" and _normalize_term(term) in lexicon_lower_by_domain.get(src, set()):
        src_z = 0.8
        src_source = "normalized"

    other = [d for d in all_domains if d != src]
    max_other = 0.0
    for d in other:
        z, z_source = _lookup_domain_z(term_stats, d, term)
        if z_source == "default" and _normalize_term(term) in lexicon_lower_by_domain.get(d, set()):
            z = max(z, 0.8)
        if z > max_other:
            max_other = z

    raw = src_z - max_other
    return float(_sigmoid(raw / 2.0)), src_source


def score_terms(
    extracted_terms: list[dict[str, object]],
    src: str,
    tgt: str,
    all_domains: list[str],
    term_stats: dict[str, dict[str, float]],
    lexicon_by_domain: dict[str, list[str]],
    lexicon_lower_by_domain: dict[str, set[str]] | None = None,
    cfg: TermScoreConfig | None = None,
    same_field_mode: str = "normal",
) -> list[dict[str, object]]:
    cfg = cfg or TermScoreConfig()
    lexicon_lower = lexicon_lower_by_domain or {
        d: {_normalize_term(t) for t in terms}
        for d, terms in lexicon_by_domain.items()
    }
    out: list[dict[str, object]] = []

    for item in extracted_terms:
        surface_term = str(item.get("surface_term", item.get("term", ""))).strip()
        canonical_term = str(item.get("canonical_term", item.get("term", ""))).strip()
        if not canonical_term:
            continue

        fam_tgt, fam_source = _familiarity(canonical_term, tgt, term_stats, lexicon_lower)
        dist_src, dist_source = _distinctiveness(canonical_term, src, tgt, all_domains, term_stats, lexicon_lower)
        lexicon_support = _normalize_term(canonical_term) in lexicon_lower.get(src, set()) or _normalize_term(canonical_term) in lexicon_lower.get(tgt, set())
        concept_score = concept_likeness_score(
            canonical_term,
            leading_pos=str(item.get("leading_pos", "")),
            leading_lemma=str(item.get("leading_lemma", "")),
            noun_headed=bool(item.get("noun_headed", False)),
            noun_adj_compound=bool(item.get("noun_adj_compound", False)),
            lexicon_support=lexicon_support,
            evidence_support=False,
            stronger_overlap=bool(item.get("overlapped_by_stronger", False)),
        )

        concept_like = _is_concept_like(canonical_term)
        if src == tgt and same_field_mode == "normal":
            flagged = bool(concept_like and concept_score >= 0.45 and dist_src >= 0.70 and fam_tgt <= 0.20)
            src_th = 0.70
            tgt_th = 0.20
        else:
            flagged = bool(concept_score >= 0.35 and dist_src >= cfg.src_threshold and fam_tgt <= cfg.tgt_threshold)
            src_th = cfg.src_threshold
            tgt_th = cfg.tgt_threshold

        reasons: list[str] = []
        if dist_src >= src_th:
            reasons.append("src_distinctive")
        if fam_tgt <= tgt_th:
            reasons.append("low_tgt_familiarity")
        if src == tgt and same_field_mode == "normal" and not concept_like:
            reasons.append("not_concept_like")
        if concept_score < 0.35:
            reasons.append("low_concept_likeness")
        reason = "+".join(reasons) if reasons else "not_flagged"

        out.append(
            {
                "term": canonical_term,
                "surface_term": surface_term,
                "canonical_term": canonical_term,
                "start": int(item.get("start", -1)),
                "end": int(item.get("end", -1)),
                "familiarity_tgt": round(float(fam_tgt), 4),
                "distinctiveness_src": round(float(dist_src), 4),
                "familiarity_source": fam_source,
                "distinctiveness_source": dist_source,
                "concept_likeness": round(float(concept_score), 4),
                "flagged": flagged,
                "reason": reason,
            }
        )

        if fam_source == "default" or dist_source == "default":
            print(
                "[score_default_fallback]",
                {
                    "term": canonical_term,
                    "src": src,
                    "tgt": tgt,
                    "familiarity_source": fam_source,
                    "distinctiveness_source": dist_source,
                },
            )

    out.sort(key=lambda r: (not bool(r["flagged"]), -float(r["distinctiveness_src"]), float(r["familiarity_tgt"])))
    deduped: list[dict[str, object]] = []
    seen_can: set[str] = set()
    for row in out:
        can = _normalize_term(str(row.get("canonical_term", row.get("term", ""))))
        if can in seen_can:
            continue
        seen_can.add(can)
        deduped.append(row)
    return deduped

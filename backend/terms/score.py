from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TermScoreConfig:
    src_threshold: float = 0.35
    tgt_threshold: float = 0.45


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _domain_z(term_stats: dict[tuple[str, str], float], domain: str, term: str) -> float:
    return float(term_stats.get((domain, term.lower().strip()), 0.0))


def _familiarity(term: str, tgt: str, term_stats: dict[tuple[str, str], float], lexicon_by_domain: dict[str, list[str]]) -> float:
    z = _domain_z(term_stats, tgt, term)
    if z != 0:
        return float(_sigmoid(z / 2.0))
    return 0.6 if term.lower() in {t.lower() for t in lexicon_by_domain.get(tgt, [])} else 0.1


def _distinctiveness(term: str, src: str, tgt: str, all_domains: list[str], term_stats: dict[tuple[str, str], float], lexicon_by_domain: dict[str, list[str]]) -> float:
    src_z = _domain_z(term_stats, src, term)
    if src_z == 0:
        src_z = 0.8 if term.lower() in {t.lower() for t in lexicon_by_domain.get(src, [])} else 0.0
    other = [d for d in all_domains if d != src]
    max_other = max((_domain_z(term_stats, d, term) for d in other), default=0.0)
    raw = src_z - max_other
    return float(_sigmoid(raw / 2.0))


def score_terms(
    extracted_terms: list[dict[str, object]],
    src: str,
    tgt: str,
    all_domains: list[str],
    term_stats: dict[tuple[str, str], float],
    lexicon_by_domain: dict[str, list[str]],
    cfg: TermScoreConfig | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or TermScoreConfig()
    out: list[dict[str, object]] = []

    for item in extracted_terms:
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        fam_tgt = _familiarity(term, tgt, term_stats, lexicon_by_domain)
        dist_src = _distinctiveness(term, src, tgt, all_domains, term_stats, lexicon_by_domain)
        flagged = bool(dist_src >= cfg.src_threshold and fam_tgt <= cfg.tgt_threshold)

        reasons: list[str] = []
        if dist_src >= cfg.src_threshold:
            reasons.append("src_distinctive")
        if fam_tgt <= cfg.tgt_threshold:
            reasons.append("low_tgt_familiarity")
        reason = "+".join(reasons) if reasons else "not_flagged"

        out.append(
            {
                "term": term,
                "start": int(item.get("start", -1)),
                "end": int(item.get("end", -1)),
                "familiarity_tgt": round(float(fam_tgt), 4),
                "distinctiveness_src": round(float(dist_src), 4),
                "flagged": flagged,
                "reason": reason,
            }
        )

    out.sort(key=lambda r: (not bool(r["flagged"]), -float(r["distinctiveness_src"]), float(r["familiarity_tgt"])))
    return out

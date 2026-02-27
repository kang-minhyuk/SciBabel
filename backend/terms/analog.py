from __future__ import annotations

import re
import os
from pathlib import Path

GENERIC_BLOCK = {
    "model",
    "method",
    "methods",
    "approach",
    "approaches",
    "results",
    "study",
    "analysis",
    "data",
    "system",
}


def _load_generic_block_from_stoplist() -> set[str]:
    path = Path(__file__).resolve().parents[2] / "scripts" / "textmining" / "stoplists" / "academic_stopwords.txt"
    if not path.exists():
        return set()
    vals = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return vals


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def _jaccard(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class AnalogSuggester:
    def __init__(self, analog_sim_threshold: float = 0.2) -> None:
        self.analog_sim_threshold = analog_sim_threshold
        self._generic = GENERIC_BLOCK | _load_generic_block_from_stoplist()
        self._rapidfuzz = None
        try:
            from rapidfuzz import fuzz  # type: ignore

            self._rapidfuzz = fuzz
        except Exception:
            self._rapidfuzz = None

    def _score_candidates(self, term: str, candidates: list[str], max_candidates: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        capped = candidates[:max(1, int(max_candidates))]
        if self._rapidfuzz is None:
            return [(c, _jaccard(term, c)) for c in capped]
        return [
            (c, (0.7 * _jaccard(term, c)) + (0.3 * (float(self._rapidfuzz.token_set_ratio(term, c)) / 100.0)))
            for c in capped
        ]

    def suggest(self, term: str, target_candidates: list[str], top_k: int = 5, max_candidates: int = 300) -> list[dict[str, object]]:
        clean_pool: list[str] = []
        seen: set[str] = set()
        term_low = term.lower().strip()
        for c in target_candidates:
            c1 = str(c).strip()
            c_low = c1.lower()
            if not c1 or c_low == term_low:
                continue
            if c_low in seen:
                continue
            if c_low in self._generic:
                continue
            if len(c_low) < 3:
                continue
            seen.add(c_low)
            clean_pool.append(c1)

            if len(clean_pool) >= max(1, int(max_candidates)):
                break

        scored = self._score_candidates(term, clean_pool, max_candidates=max_candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        out = [
            {"candidate": c, "score": round(float(s), 4)}
            for c, s in scored
            if float(s) >= self.analog_sim_threshold
        ]
        return out[:top_k]

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

DOMAINS = ["CSM", "PM", "CHEM", "CHEME"]
CHEME_HINTS = {
    "reactor",
    "separation",
    "distillation",
    "membrane",
    "adsorption",
    "mass transfer",
    "transport",
    "process control",
    "reaction engineering",
    "heat transfer",
    "column",
    "reflux",
}
CHEM_HINTS = {
    "transition state",
    "substituent",
    "synthesis",
    "molecule",
    "molecular",
    "ligand",
    "spectroscopy",
    "reaction mechanism",
}


@dataclass
class DetectorConfig:
    min_words: int = int(os.getenv("SRC_AUTO_MIN_WORDS", "18"))
    min_conf: float = float(os.getenv("SRC_AUTO_MIN_CONF", "0.55"))
    min_gap: float = float(os.getenv("SRC_AUTO_MIN_GAP", "0.10"))


class SourceDetector:
    def __init__(self, model_path: Path, cfg: DetectorConfig | None = None) -> None:
        self.model_path = model_path
        self.cfg = cfg or DetectorConfig()
        self.clf = joblib.load(model_path)

    @staticmethod
    def _word_count(text: str) -> int:
        return len([w for w in text.strip().split() if w])

    @staticmethod
    def _normalize(probs: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, float(v)) for v in probs.values())
        if total <= 0:
            uni = 1.0 / max(1, len(probs))
            return {k: uni for k in probs}
        return {k: max(0.0, float(v)) / total for k, v in probs.items()}

    @staticmethod
    def _keyword_score(text: str, keywords: set[str]) -> float:
        t = text.lower()
        return float(sum(1 for k in keywords if k in t))

    def _expand_cce(self, base_probs: dict[str, float], text: str) -> dict[str, float]:
        p_cce = float(base_probs.get("CCE", 0.0))
        p_chem = float(base_probs.get("CHEM", 0.0))
        p_cheme = float(base_probs.get("CHEME", 0.0))

        if p_cce <= 0:
            out = {
                "CSM": float(base_probs.get("CSM", 0.0)),
                "PM": float(base_probs.get("PM", 0.0)),
                "CHEM": p_chem,
                "CHEME": p_cheme,
            }
            return self._normalize(out)

        s_cheme = self._keyword_score(text, CHEME_HINTS)
        s_chem = self._keyword_score(text, CHEM_HINTS)
        if s_cheme <= 0 and s_chem <= 0:
            w_chem, w_cheme = 0.45, 0.55
        else:
            tot = max(1e-6, s_chem + s_cheme)
            w_chem = s_chem / tot
            w_cheme = s_cheme / tot

        out = {
            "CSM": float(base_probs.get("CSM", 0.0)),
            "PM": float(base_probs.get("PM", 0.0)),
            "CHEM": p_chem + p_cce * w_chem,
            "CHEME": p_cheme + p_cce * w_cheme,
        }
        return self._normalize(out)

    def detect_source(self, text: str) -> dict[str, Any]:
        labels = [str(x) for x in list(getattr(self.clf, "classes_", []))]
        probs_arr = self.clf.predict_proba([text])[0]
        raw = {labels[i]: float(probs_arr[i]) for i in range(min(len(labels), len(probs_arr)))}
        probs = self._expand_cce(raw, text)

        for d in DOMAINS:
            probs.setdefault(d, 0.0)
        probs = self._normalize(probs)

        top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top1_domain, top1_prob = top[0]
        top2_prob = top[1][1] if len(top) > 1 else 0.0
        gap = float(top1_prob - top2_prob)

        words = self._word_count(text)
        reason = "none"
        ambiguous = False
        if words < self.cfg.min_words:
            ambiguous = True
            reason = "too_short"
        elif top1_prob < self.cfg.min_conf:
            ambiguous = True
            reason = "low_confidence"
        elif gap < self.cfg.min_gap:
            ambiguous = True
            reason = "small_gap"

        return {
            "predicted_src": top1_domain,
            "confidence": float(top1_prob),
            "probs": {k: float(v) for k, v in probs.items()},
            "is_ambiguous": bool(ambiguous),
            "top2_gap": float(gap),
            "reason": reason,
        }

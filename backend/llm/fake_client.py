from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.openai_client import BANNED_PHRASES, MUST_INCLUDE_ANY


@dataclass
class FakeExplainClient:
    model: str = "fake-llm"

    def explain(self, req: Any) -> dict[str, Any]:
        analog = req.analogs[0] if getattr(req, "analogs", None) else None
        phrase = MUST_INCLUDE_ANY[0]
        short = f"In {req.tgt}, '{req.term}' is {phrase} to '{analog or 'a related concept'}', but not equivalent."
        long = (
            f"Within the given sentence context, '{req.term}' can be framed using {phrase} language for {req.tgt}. "
            "This is an analogy, not a one-to-one identity."
        )
        blob = (short + " " + long).lower()
        assert not any(bp in blob for bp in BANNED_PHRASES)
        return {
            "term": req.term,
            "short_explanation": short,
            "long_explanation": long,
            "closest_analog": analog,
            "caution_label": "analogous_not_equivalent",
            "cache_hit": False,
            "model": self.model,
            "semantic_policy": {
                "banned_phrases": BANNED_PHRASES,
                "must_include_any": MUST_INCLUDE_ANY,
            },
        }

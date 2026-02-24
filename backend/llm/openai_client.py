from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BANNED_PHRASES = ["equivalent to", "identical to", "is the same as"]
MUST_INCLUDE_ANY = ["analogous", "conceptually similar", "shares structural similarity"]


@dataclass
class ExplainRequest:
    text: str
    term: str
    src: str
    tgt: str
    audience_level: str
    subtrack: str
    analogs: list[str]
    detail: str


class ExplainPolicyError(RuntimeError):
    pass


class OpenAIExplainClient:
    def __init__(self, db_path: Path, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Explanations require GPT API access.")

        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS explain_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _cache_key(self, req: ExplainRequest) -> str:
        context_hash = hashlib.sha256(req.text.encode("utf-8")).hexdigest()
        raw = "|".join([
            req.term,
            req.tgt,
            req.audience_level,
            req.subtrack,
            req.detail,
            context_hash,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_cache(self, key: str) -> dict[str, Any] | None:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT payload_json FROM explain_cache WHERE cache_key=?", (key,)).fetchone()
            if not row:
                return None
            return json.loads(str(row[0]))
        finally:
            conn.close()

    def _set_cache(self, key: str, payload: dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "REPLACE INTO explain_cache(cache_key, payload_json, created_at) VALUES(?,?,?)",
                (key, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def _build_prompt(self, req: ExplainRequest) -> str:
        analogs_text = ", ".join(req.analogs[:5]) if req.analogs else "none"
        return (
            "You are a careful cross-domain explainer. Return ONLY JSON object with keys "
            "short_explanation, long_explanation, closest_analog.\n"
            "Rules:\n"
            f"- NEVER use banned phrases: {BANNED_PHRASES}\n"
            f"- MUST include at least one phrase from: {MUST_INCLUDE_ANY}\n"
            "- Use analogy framing, not equivalence.\n"
            "- Do not introduce claims outside provided sentence context.\n"
            "- Keep audience level appropriate.\n\n"
            f"Context sentence: {req.text}\n"
            f"Source domain: {req.src}\n"
            f"Target domain: {req.tgt}\n"
            f"Term: {req.term}\n"
            f"Audience level: {req.audience_level}\n"
            f"Subtrack: {req.subtrack or 'none'}\n"
            f"Analog candidates: {analogs_text}\n"
            f"Detail preference: {req.detail}\n"
        )

    @staticmethod
    def _validate_policy(payload: dict[str, Any]) -> None:
        text_blob = (str(payload.get("short_explanation", "")) + " " + str(payload.get("long_explanation", ""))).lower()
        if any(bp in text_blob for bp in BANNED_PHRASES):
            raise ExplainPolicyError("Banned equivalence phrase detected in explanation.")
        if not any(mp in text_blob for mp in MUST_INCLUDE_ANY):
            raise ExplainPolicyError("Explanation must include analogy language.")

    def _parse_json(self, content: str) -> dict[str, Any]:
        obj = json.loads(content)
        if not isinstance(obj, dict):
            raise ValueError("Expected JSON object.")
        return obj

    def explain(self, req: ExplainRequest) -> dict[str, Any]:
        key = self._cache_key(req)
        cached = self._get_cache(key)
        if cached:
            out = dict(cached)
            out["cache_hit"] = True
            return out

        prompt = self._build_prompt(req)
        content = self._call_responses(prompt)
        try:
            parsed = self._parse_json(content)
        except Exception:
            repair_prompt = (
                "Repair the following into valid JSON object with keys short_explanation,long_explanation,closest_analog only:\n"
                + content
            )
            repaired = self._call_responses(repair_prompt)
            parsed = self._parse_json(repaired)

        parsed.setdefault("short_explanation", "")
        parsed.setdefault("long_explanation", "")
        parsed.setdefault("closest_analog", None)

        self._validate_policy(parsed)

        out = {
            "term": req.term,
            "short_explanation": str(parsed.get("short_explanation", "")),
            "long_explanation": str(parsed.get("long_explanation", "")),
            "closest_analog": parsed.get("closest_analog"),
            "caution_label": "analogous_not_equivalent",
            "cache_hit": False,
            "model": self.model,
            "semantic_policy": {
                "banned_phrases": BANNED_PHRASES,
                "must_include_any": MUST_INCLUDE_ANY,
            },
        }
        self._set_cache(key, out)
        return out

    def _call_responses(self, prompt: str) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
            text={"format": {"type": "json_object"}},
            temperature=0.2,
        )
        text_out = getattr(resp, "output_text", "")
        if text_out:
            return str(text_out)

        # fallback extraction
        try:
            chunks = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    maybe = getattr(c, "text", None)
                    if maybe:
                        chunks.append(str(maybe))
            joined = "\n".join(chunks).strip()
            if joined:
                return joined
        except Exception:
            pass
        raise RuntimeError("OpenAI Responses API returned empty output.")

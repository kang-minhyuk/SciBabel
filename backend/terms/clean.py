from __future__ import annotations

import re

DEBUG_PATTERNS = [
    re.compile(r"\[\s*updatedgpt[^\]]*\]", re.IGNORECASE),
    re.compile(r"updatedgpt[_\-a-z0-9]*", re.IGNORECASE),
    re.compile(r"\(\s*native\s*=\s*[^\)]*\)", re.IGNORECASE),
    re.compile(r"\(\s*domain-specific concept\s*\)", re.IGNORECASE),
]
WS_RE = re.compile(r"\s+")


def clean_text_for_mining(text: str) -> str:
    out = text or ""
    for patt in DEBUG_PATTERNS:
        out = patt.sub(" ", out)
    return WS_RE.sub(" ", out).strip()

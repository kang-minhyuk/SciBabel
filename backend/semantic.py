from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")

_semantic_model: Any | None = None
_semantic_enabled: bool = False


def get_semantic_mode() -> str:
    return "embedding" if _semantic_enabled and _semantic_model is not None else "overlap"


def _token_jaccard(a: str, b: str) -> float:
    ta = {m.group(0).lower() for m in TOKEN_RE.finditer(a)}
    tb = {m.group(0).lower() for m in TOKEN_RE.finditer(b)}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def init_semantic_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> bool:
    """Initialize sentence-transformers model once.

    Returns True if loaded, False if unavailable.
    """
    global _semantic_model, _semantic_enabled
    if _semantic_model is not None:
        return _semantic_enabled

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _semantic_model = SentenceTransformer(model_name)
        _semantic_enabled = True
        return True
    except Exception:
        _semantic_model = None
        _semantic_enabled = False
        return False


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity using sentence embeddings, with token-overlap fallback."""
    if _semantic_enabled and _semantic_model is not None:
        try:
            emb = _semantic_model.encode([a, b], normalize_embeddings=True)
            sim = float((emb[0] * emb[1]).sum())
            return max(0.0, min(1.0, sim))
        except Exception:
            pass

    return _token_jaccard(a, b)

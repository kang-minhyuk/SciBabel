from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PDF_CACHE_DIR = Path(os.getenv("PDF_CACHE_DIR", str(ROOT / "backend" / "pdf_cache")))


def _cache_path(document_id: str) -> Path:
    return PDF_CACHE_DIR / f"{document_id}.json"


def create_document_id() -> str:
    return uuid.uuid4().hex


def save_document(payload: dict[str, Any]) -> None:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doc_id = str(payload.get("document_id") or "")
    if not doc_id:
        raise ValueError("missing document_id")
    payload = dict(payload)
    payload["updated_at"] = time.time()
    _cache_path(doc_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_document(document_id: str) -> dict[str, Any] | None:
    p = _cache_path(document_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

DEBUG_PATTERNS = [
    re.compile(r"\[\s*updatedgpt[^\]]*\]", re.IGNORECASE),
    re.compile(r"updatedgpt[_\-a-z0-9]*", re.IGNORECASE),
    re.compile(r"\(\s*native\s*=\s*[^\)]*\)", re.IGNORECASE),
    re.compile(r"\(\s*domain-specific concept\s*\)", re.IGNORECASE),
]

WHITESPACE_RE = re.compile(r"\s+")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seen_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(path):
        rid = str(row.get("id", "")).strip()
        if rid:
            out.add(rid)
    return out


def sanitize_text(text: str) -> str:
    out = text or ""
    for patt in DEBUG_PATTERNS:
        out = patt.sub(" ", out)
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def english_like(text: str) -> bool:
    if not text:
        return False
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return False
    return (ascii_letters / letters) >= 0.8


def normalize_for_vectorizer(title: str, abstract: str) -> str:
    text = sanitize_text(f"{title} {abstract}")
    return text.lower()

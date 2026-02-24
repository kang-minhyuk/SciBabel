from __future__ import annotations

from pathlib import Path


def stoplists_dir() -> Path:
    return Path(__file__).resolve().parent / "stoplists"


def load_stoplist(name: str) -> set[str]:
    path = stoplists_dir() / name
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_all_stoplists() -> set[str]:
    merged: set[str] = set()
    for name in ["academic_stopwords.txt", "domain_generic.txt", "debug_artifacts.txt"]:
        merged |= load_stoplist(name)
    return merged

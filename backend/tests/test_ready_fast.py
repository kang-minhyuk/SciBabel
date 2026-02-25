from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module


def test_ready_fast_and_no_corpus_open(monkeypatch) -> None:
    real_open = Path.open

    def _guard_open(self: Path, *args, **kwargs):
        p = str(self).lower()
        if "corpus.parquet" in p or "corpus.jsonl" in p or "corpus_full.parquet" in p:
            raise AssertionError("/ready must not open corpus files")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _guard_open)

    client = TestClient(app_module.app)
    t0 = time.perf_counter()
    resp = client.get("/ready")
    elapsed = time.perf_counter() - t0

    assert resp.status_code == 200
    body = resp.json()
    assert "ready" in body
    assert "missing" in body
    assert "artifacts" in body
    assert elapsed < 1.0

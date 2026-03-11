from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module


async def _always_busy() -> bool:
    return False


def test_annotate_returns_429_when_busy(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_acquire_annotate_slot", _always_busy)

    client = TestClient(app_module.app)
    resp = client.post(
        "/annotate",
        json={"text": "short sample sentence", "src": "auto", "tgt": "PM", "max_terms": 3},
    )
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"] == "busy"
    assert body["hint"] == "try again"

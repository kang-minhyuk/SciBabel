from __future__ import annotations

import time

from fastapi.testclient import TestClient

import app as app_module


async def _always_acquired() -> bool:
    return True


def _slow_impl(payload, include_profile: bool = False):
    _ = (payload, include_profile)
    time.sleep(0.2)
    return {"terms": []}


def test_annotate_timeout_budget(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_acquire_annotate_slot", _always_acquired)
    monkeypatch.setattr(app_module, "_annotate_impl_sync", _slow_impl)
    monkeypatch.setattr(app_module, "ANNOTATE_TIMEOUT_SEC", 0.05)

    client = TestClient(app_module.app)
    resp = client.post(
        "/annotate",
        json={"text": "short sample sentence", "src": "auto", "tgt": "PM", "max_terms": 3},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "timeout_budget_exceeded"
    assert float(body["timeout_sec"]) == 0.05

from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module
from resources import ArtifactsMissingError


MISSING = [
    "data/processed/domain_lexicon.json",
    "data/processed/term_stats.csv",
    "models/domain_clf.joblib",
]


def _raise_missing(*, load_explain: bool = False):
    _ = load_explain
    raise ArtifactsMissingError(MISSING)


def test_annotate_returns_503_with_hint(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_resources", _raise_missing)

    client = TestClient(app_module.app)
    resp = client.post(
        "/annotate",
        json={"text": "short sample sentence", "src": "auto", "tgt": "PM", "max_terms": 3},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "artifacts_missing"
    assert body["missing"] == MISSING
    assert "build_artifacts.py" in body["hint"]


def test_explain_returns_503_with_hint(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_resources", _raise_missing)

    client = TestClient(app_module.app)
    resp = client.post(
        "/explain",
        json={
            "text": "Sample sentence for explain.",
            "term": "sample",
            "src": "auto",
            "tgt": "PM",
            "detail": "short",
        },
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "artifacts_missing"
    assert body["missing"] == MISSING

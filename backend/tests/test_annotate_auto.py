from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module


class _DummyEngine:
    def annotate(self, text: str, src: str, tgt: str, max_terms: int = 8) -> dict[str, object]:
        return {
            "src_effective": src,
            "terms": [
                {
                    "term": "sparse attention",
                    "start": 10,
                    "end": 26,
                    "familiarity_tgt": 0.2,
                    "distinctiveness_src": 0.8,
                    "flagged": True,
                    "reason": "src_distinctive+low_tgt_familiarity",
                    "analogs": [],
                    "evidence": [],
                    "explain_available": True,
                }
            ],
        }


class _DummyDetector:
    def detect_source(self, text: str) -> dict[str, object]:
        return {
            "predicted_src": "CSM",
            "confidence": 0.82,
            "probs": {"CSM": 0.82, "PM": 0.1, "CHEM": 0.05, "CHEME": 0.03},
            "is_ambiguous": False,
            "top2_gap": 0.72,
            "reason": "none",
        }


def test_annotate_auto_includes_source_metadata(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "_ensure_annotation_ready", lambda: _DummyEngine())
    monkeypatch.setattr(app_module, "_ensure_source_detector", lambda: _DummyDetector())

    client = TestClient(app_module.app)
    resp = client.post(
        "/annotate",
        json={
            "text": "We optimize a regularized loss using gradient descent and sparse attention in a neural architecture.",
            "src": "auto",
            "tgt": "PM",
            "max_terms": 6,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["predicted_src"] == "CSM"
    assert body["src_used"] == "CSM"
    assert isinstance(body["predicted_src_probs"], dict)
    assert "terms" in body and isinstance(body["terms"], list)

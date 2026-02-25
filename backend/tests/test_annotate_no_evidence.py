from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module


class _DummyEngine:
    def annotate(self, text: str, src: str, tgt: str, max_terms: int = 8) -> dict[str, object]:
        return {
            "src_effective": src,
            "_timings": {
                "extract_terms_sec": 0.01,
                "score_terms_sec": 0.01,
                "analog_search_sec": 0.01,
                "evidence_sec": 0.0,
                "total_sec": 0.03,
            },
            "terms": [
                {
                    "term": "sparse attention",
                    "start": 10,
                    "end": 26,
                    "familiarity_tgt": 0.2,
                    "distinctiveness_src": 0.8,
                    "flagged": True,
                    "reason": "src_distinctive+low_tgt_familiarity",
                    "analogs": [{"candidate": "phonon mode", "score": 0.7}],
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


class _DummyResources:
    annotation_engine = _DummyEngine()
    source_detector = _DummyDetector()
    explain_client = None


def test_annotate_evidence_disabled_returns_empty_evidence(monkeypatch) -> None:
    monkeypatch.setenv("EVIDENCE_ENABLED", "false")
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(app_module, "get_spacy", lambda: object())

    client = TestClient(app_module.app)
    resp = client.post(
        "/annotate",
        json={
            "text": "We optimize a regularized loss using sparse attention.",
            "src": "auto",
            "tgt": "PM",
            "max_terms": 8,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    terms = body.get("terms", [])
    assert isinstance(terms, list)
    assert all(isinstance(t.get("evidence", []), list) and len(t.get("evidence", [])) == 0 for t in terms)

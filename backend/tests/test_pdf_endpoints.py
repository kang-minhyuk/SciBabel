from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import app as app_module
import pdf.store as pdf_store


class _DummyEngine:
    def annotate(self, text: str, src: str, tgt: str, max_terms: int = 8, same_field_mode: str = "normal") -> dict[str, object]:
        _ = (src, tgt, max_terms, same_field_mode)
        needle = "sparse attention"
        pos = text.lower().find(needle)
        if pos < 0:
            return {"terms": []}
        return {
            "terms": [
                {
                    "term": "sparse attention",
                    "surface_term": "sparse attention",
                    "canonical_term": "sparse attention",
                    "start": pos,
                    "end": pos + len(needle),
                    "flagged": True,
                    "familiarity_tgt": 0.2,
                    "distinctiveness_src": 0.9,
                    "reason": "src_distinctive+low_tgt_familiarity",
                    "analogs": [{"candidate": "selective focus", "score": 0.8}],
                    "evidence": [],
                    "explain_available": True,
                }
            ]
        }


class _DummyDetector:
    def detect_source(self, text: str) -> dict[str, object]:
        _ = text
        return {
            "predicted_src": "CSM",
            "confidence": 0.9,
            "probs": {"CSM": 0.9, "PM": 0.05, "CHEM": 0.03, "CHEME": 0.02},
            "is_ambiguous": False,
            "top2_gap": 0.85,
            "reason": "none",
        }


class _DummyResources:
    annotation_engine = _DummyEngine()
    source_detector = _DummyDetector()
    explain_client = None


class _DummyExplainClient:
    def explain(self, req):
        term = getattr(req, "term", "term")
        return {
            "term": term,
            "short_explanation": "Short local fake explanation",
            "long_explanation": "Long local fake explanation",
            "closest_analog": None,
            "caution_label": "none",
            "cache_hit": False,
            "model": "fake-local",
        }


class _DummyResourcesWithExplain:
    annotation_engine = _DummyEngine()
    source_detector = _DummyDetector()
    explain_client = _DummyExplainClient()


def _make_pdf_bytes() -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 760, "A graph neural network uses sparse attention for inference.")
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_two_page_pdf_bytes() -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 760, "A graph neural network uses sparse attention for inference.")
    c.showPage()
    c.drawString(72, 760, "Sparse attention also appears on the second page.")
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_empty_text_pdf_bytes() -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_pdf_annotate_and_fetch_document(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(app_module, "get_spacy", lambda: object())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)

    client = TestClient(app_module.app)

    resp = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("sample.pdf", _make_pdf_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"]
    assert body["page_count"] == 1
    assert body["pages"][0]["terms"]

    doc_id = body["document_id"]
    fetched = client.get(f"/pdf/document/{doc_id}")
    assert fetched.status_code == 200
    fetched_body = fetched.json()
    assert fetched_body["document_id"] == doc_id


def test_pdf_page_numbers_and_term_ids_stable_within_document(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(app_module, "get_spacy", lambda: object())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)
    client = TestClient(app_module.app)

    resp = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("two_pages.pdf", _make_two_page_pdf_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()

    pages = body["pages"]
    assert [p["page_num"] for p in pages] == [1, 2]
    first_ids = [t.get("term_id") for t in pages[0].get("terms", [])]
    assert all(first_ids), "all terms should include term_id"

    doc_id = body["document_id"]
    fetched = client.get(f"/pdf/document/{doc_id}")
    assert fetched.status_code == 200
    fetched_body = fetched.json()
    fetched_ids = [t.get("term_id") for t in fetched_body["pages"][0].get("terms", [])]
    assert first_ids == fetched_ids


def test_pdf_explain_happy_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)

    doc = {
        "document_id": "doc12345678",
        "src_used": "CSM",
        "pages": [
            {
                "page_num": 1,
                "text": "Sparse attention speeds inference.",
                "terms": [{"term_id": "abc123", "term": "sparse attention", "analogs": []}],
            }
        ],
    }
    pdf_store.save_document(doc)

    monkeypatch.setattr(
        app_module,
        "explain_term_from_document",
        lambda d, r: {
            "term": "sparse attention",
            "short_explanation": "A targeted focus mechanism.",
            "long_explanation": "Detailed explanation",
            "closest_analog": "selective focus",
            "caution_label": "none",
            "cache_hit": False,
            "model": "fake",
            "page_num": r["page_num"],
            "term_id": r["term_id"],
        },
    )

    client = TestClient(app_module.app)
    resp = client.post(
        "/pdf/explain",
        json={
            "document_id": "doc12345678",
            "page_num": 1,
            "term_id": "abc123",
            "tgt": "PM",
            "detail": "short",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["term"] == "sparse attention"
    assert body["term_id"] == "abc123"


def test_pdf_explain_with_fake_client(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(app_module, "get_spacy", lambda: object())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)
    client = TestClient(app_module.app)

    ann = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("sample.pdf", _make_pdf_bytes(), "application/pdf")},
    )
    assert ann.status_code == 200
    out = ann.json()
    term = out["pages"][0]["terms"][0]

    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=True: _DummyResourcesWithExplain())
    ex = client.post(
        "/pdf/explain",
        json={
            "document_id": out["document_id"],
            "page_num": 1,
            "term_id": term["term_id"],
            "tgt": "PM",
            "detail": "short",
        },
    )
    assert ex.status_code == 200
    body = ex.json()
    assert body["short_explanation"]


def test_pdf_annotate_invalid_file_type_fails_cleanly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)
    client = TestClient(app_module.app)

    resp = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("sample.txt", b"not pdf", "text/plain")},
    )
    assert resp.status_code == 422
    assert "file must be a .pdf" in str(resp.json())


def test_pdf_annotate_broken_pdf_fails_cleanly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)
    client = TestClient(app_module.app)

    resp = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("broken.pdf", b"not_a_pdf", "application/pdf")},
    )
    assert resp.status_code == 422
    assert "invalid_pdf" in str(resp.json())


def test_pdf_annotate_empty_text_pdf_fails_cleanly(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(app_module, "get_resources", lambda load_explain=False: _DummyResources())
    monkeypatch.setattr(pdf_store, "PDF_CACHE_DIR", tmp_path)
    client = TestClient(app_module.app)

    resp = client.post(
        "/pdf/annotate",
        data={"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"},
        files={"file": ("empty.pdf", _make_empty_text_pdf_bytes(), "application/pdf")},
    )
    assert resp.status_code == 422
    assert "no_extractable_text" in str(resp.json())


def test_pdf_document_not_found() -> None:
    client = TestClient(app_module.app)
    resp = client.get("/pdf/document/does_not_exist")
    assert resp.status_code == 404

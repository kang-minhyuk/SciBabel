from __future__ import annotations

import argparse
from io import BytesIO

import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _sample_pdf_bytes() -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(72, 760, "Graph neural networks with sparse attention improve scalability.")
    c.showPage()
    c.save()
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    files = {"file": ("demo.pdf", _sample_pdf_bytes(), "application/pdf")}
    data = {"src": "auto", "tgt": "PM", "audience_level": "grad", "max_terms": "8"}
    base = args.api_base.rstrip("/")
    r = requests.post(f"{base}/pdf/annotate", data=data, files=files, timeout=30)
    r.raise_for_status()
    out = r.json()

    doc_id = out.get("document_id")
    assert doc_id, "missing document_id"
    assert isinstance(out.get("page_count"), int), "missing page_count"
    assert isinstance(out.get("predicted_src"), str), "missing predicted_src"
    pages = out.get("pages", [])
    assert pages and isinstance(pages, list), "missing pages"
    assert int(pages[0].get("page_num", 0)) == 1, "bad page numbering"

    doc_r = requests.get(f"{base}/pdf/document/{doc_id}", timeout=20)
    doc_r.raise_for_status()
    doc = doc_r.json()
    assert doc.get("document_id") == doc_id, "cache retrieval mismatch"

    first_terms = pages[0].get("terms", []) if isinstance(pages[0].get("terms", []), list) else []
    explain_status = "skipped"
    if first_terms:
        t0 = first_terms[0]
        if t0.get("term_id"):
            ex_payload = {
                "document_id": doc_id,
                "page_num": 1,
                "term_id": t0.get("term_id"),
                "tgt": "PM",
                "detail": "short",
            }
            ex_r = requests.post(f"{base}/pdf/explain", json=ex_payload, timeout=30)
            # In local fake/no-key mode this may be 503, which still verifies route behavior.
            assert ex_r.status_code in {200, 503}, f"unexpected explain status: {ex_r.status_code}"
            explain_status = str(ex_r.status_code)

    print(
        {
            "status": "ok",
            "document_id": doc_id,
            "page_count": out.get("page_count", 0),
            "explain_status": explain_status,
        }
    )


if __name__ == "__main__":
    main()

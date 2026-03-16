from __future__ import annotations

from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from pdf.extract import PdfEmptyTextError, PdfExtractError, extract_pdf_pages


def _make_pdf_bytes(pages: list[list[str]]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for lines in pages:
        y = 760
        for line in lines:
            c.drawString(72, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def test_extract_pdf_pages_basic_text() -> None:
    data = _make_pdf_bytes(
        [
            ["A graph neural network uses sparse attention.", "Gradient descent tunes the objective."],
            ["References", "[1] Smith et al."],
        ]
    )

    pages = extract_pdf_pages(data, max_pages=10)
    assert len(pages) == 2
    assert "graph neural network" in pages[0].text.lower()
    # Reference section should be truncated once found.
    assert "smith" not in pages[1].text.lower()


def test_extract_pdf_pages_rejects_page_overflow() -> None:
    data = _make_pdf_bytes([["Only one page"]])
    try:
        extract_pdf_pages(data, max_pages=0)
        assert False, "Expected overflow error"
    except Exception as exc:
        assert "pdf_too_many_pages" in str(exc)


def test_extract_pdf_pages_rejects_empty_text_pdf() -> None:
    data = _make_pdf_bytes([[]])
    try:
        extract_pdf_pages(data, max_pages=10)
        assert False, "Expected empty-text error"
    except PdfEmptyTextError as exc:
        assert "no_extractable_text" in str(exc)


def test_extract_pdf_pages_rejects_invalid_pdf_bytes() -> None:
    try:
        extract_pdf_pages(b"this is not a pdf", max_pages=10)
        assert False, "Expected invalid-pdf error"
    except PdfExtractError as exc:
        assert "invalid_pdf" in str(exc)

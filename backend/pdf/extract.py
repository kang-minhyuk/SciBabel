from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class ExtractedPage:
    page_num: int
    text: str


class PdfExtractError(RuntimeError):
    pass


class PdfEncryptedError(PdfExtractError):
    pass


class PdfEmptyTextError(PdfExtractError):
    pass


def _clean_page_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Dehyphenate wrapped words before flattening whitespace.
    t = re.sub(r"([A-Za-z])\-\n([a-z])", r"\1\2", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _trim_repeated_headers_footers(pages: list[str]) -> list[str]:
    if len(pages) < 3:
        return pages

    first_line_counts: dict[str, int] = {}
    last_line_counts: dict[str, int] = {}

    split_pages = [p.split("\n") if "\n" in p else [p] for p in pages]
    for lines in split_pages:
        if not lines:
            continue
        first = lines[0].strip().lower()
        last = lines[-1].strip().lower()
        if first:
            first_line_counts[first] = first_line_counts.get(first, 0) + 1
        if last:
            last_line_counts[last] = last_line_counts.get(last, 0) + 1

    threshold = max(2, int(0.6 * len(pages)))
    common_first = {k for k, v in first_line_counts.items() if v >= threshold and len(k) < 120}
    common_last = {k for k, v in last_line_counts.items() if v >= threshold and len(k) < 120}

    out: list[str] = []
    for lines in split_pages:
        if not lines:
            out.append("")
            continue
        if lines and lines[0].strip().lower() in common_first:
            lines = lines[1:]
        if lines and lines[-1].strip().lower() in common_last:
            lines = lines[:-1]
        out.append("\n".join(lines).strip())
    return out


def _truncate_references(pages: list[str]) -> list[str]:
    out: list[str] = []
    cut = False
    for text in pages:
        if cut:
            out.append("")
            continue
        m = re.search(r"\b(references|bibliography)\b", text, flags=re.IGNORECASE)
        if m and m.start() <= 2000:
            maybe_header = text[max(0, m.start() - 40): m.start() + 40].lower()
            if "reference" in maybe_header or "bibliography" in maybe_header:
                out.append(text[:m.start()].strip())
                cut = True
                continue
        out.append(text)
    return out


def extract_pdf_pages(pdf_bytes: bytes, *, max_pages: int = 40, timeout_sec: float = 15.0) -> list[ExtractedPage]:
    t0 = time.perf_counter()
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise PdfExtractError(f"invalid_pdf: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        raise PdfEncryptedError("encrypted_pdf")

    n_pages = len(reader.pages)
    if n_pages == 0:
        raise PdfEmptyTextError("empty_pdf")
    if n_pages > max_pages:
        raise PdfExtractError(f"pdf_too_many_pages: {n_pages}>{max_pages}")

    raw_pages: list[str] = []
    for page in reader.pages:
        if (time.perf_counter() - t0) > timeout_sec:
            raise PdfExtractError("pdf_extract_timeout")
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        raw_pages.append(_clean_page_text(text))

    raw_pages = _trim_repeated_headers_footers(raw_pages)
    raw_pages = _truncate_references(raw_pages)

    pages = [ExtractedPage(page_num=i + 1, text=t.strip()) for i, t in enumerate(raw_pages)]
    if not any(p.text for p in pages):
        raise PdfEmptyTextError("no_extractable_text")

    return pages

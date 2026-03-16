from __future__ import annotations

import hashlib
import re
from typing import Any

from llm.openai_client import ExplainRequest
from resources import get_resources


def _term_id(page_num: int, canonical_term: str, start: int, end: int) -> str:
    # Keep ids stable for repeated local evaluations of the same content.
    raw = f"{page_num}|{canonical_term}|{start}|{end}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _sanitize_output_text(text: str) -> str:
    out = text
    out = re.sub(r"\(\s*domain-specific concept\s*\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\(\s*native\s*=\s*[^\)]*\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\[\s*updatedgpt[^\]]*\]", "", out, flags=re.IGNORECASE)
    out = re.sub(r"updatedgpt[_\-a-z0-9]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def annotate_pages(
    *,
    document_id: str,
    filename: str,
    pages: list[dict[str, Any]],
    payload: dict[str, Any],
    max_terms_per_page: int,
    max_terms_total: int,
) -> dict[str, Any]:
    resources = get_resources(load_explain=False)
    detector = resources.source_detector
    engine = resources.annotation_engine

    full_text = "\n\n".join(_sanitize_output_text(str(p.get("text", ""))) for p in pages if str(p.get("text", "")).strip())
    det = detector.detect_source(full_text if full_text else "empty text")

    src_used = str(payload.get("src") or "auto")
    src_warning = False
    src_warning_reason = "none"
    if src_used == "auto":
        src_used = str(det.get("predicted_src") or "CSM")
        if bool(det.get("is_ambiguous", False)):
            src_warning = True
            src_warning_reason = str(det.get("reason") or "ambiguous")
    else:
        pred = str(det.get("predicted_src") or "")
        conf = float(det.get("confidence") or 0.0)
        if pred and pred != src_used and conf >= 0.65:
            src_warning = True
            src_warning_reason = "mismatch"

    out_pages: list[dict[str, Any]] = []
    flagged_total = 0
    pages_with_flags = 0
    total_emitted = 0

    for page in pages:
        page_num = int(page["page_num"])
        page_text = _sanitize_output_text(str(page.get("text", "")))
        if not page_text.strip():
            out_pages.append(
                {
                    "page_num": page_num,
                    "text": "",
                    "blocks": [],
                    "terms": [],
                }
            )
            continue

        ann = engine.annotate(
            text=page_text,
            src=src_used,
            tgt=str(payload.get("tgt") or "PM"),
            max_terms=max(1, int(max_terms_per_page)),
            same_field_mode=str(payload.get("same_field_mode") or "normal"),
        )
        terms = ann.get("terms", []) if isinstance(ann, dict) else []
        if not isinstance(terms, list):
            terms = []

        dedup: dict[tuple[str, int, int], dict[str, Any]] = {}
        for t in terms:
            if not isinstance(t, dict):
                continue
            can = str(t.get("canonical_term") or t.get("term") or "").strip()
            start = int(t.get("start", -1))
            end = int(t.get("end", -1))
            if not can or start < 0 or end <= start:
                continue
            key = (can.lower(), start, end)
            row = dict(t)
            row["surface_term"] = str(t.get("surface_term") or t.get("term") or can)
            row["canonical_term"] = can
            row["term"] = can
            row["term_id"] = _term_id(page_num, can, start, end)
            if key not in dedup:
                dedup[key] = row

        page_terms = list(dedup.values())
        page_terms.sort(key=lambda x: (not bool(x.get("flagged", False)), int(x.get("start", 0))))

        if total_emitted >= max_terms_total:
            page_terms = []
        else:
            cap = max_terms_total - total_emitted
            page_terms = page_terms[:cap]
        total_emitted += len(page_terms)

        flagged_here = sum(1 for t in page_terms if bool(t.get("flagged", False)))
        flagged_total += flagged_here
        if flagged_here > 0:
            pages_with_flags += 1

        out_pages.append(
            {
                "page_num": page_num,
                "text": page_text,
                "blocks": [{"block_id": f"{document_id}-p{page_num}-b1", "text": page_text, "start": 0, "end": len(page_text)}],
                "terms": page_terms,
            }
        )

    return {
        "document_id": document_id,
        "filename": filename,
        "page_count": len(pages),
        "predicted_src": det.get("predicted_src"),
        "predicted_src_confidence": det.get("confidence"),
        "predicted_src_probs": det.get("probs", {}),
        "src_used": src_used,
        "src_warning": src_warning,
        "src_warning_reason": src_warning_reason,
        "is_ambiguous": bool(det.get("is_ambiguous", False)),
        "top2_gap": det.get("top2_gap"),
        "suggested_src": det.get("predicted_src"),
        "pages": out_pages,
        "summary": {
            "flagged_term_count": flagged_total,
            "pages_with_flags": pages_with_flags,
        },
    }


def explain_term_from_document(doc: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    resources = get_resources(load_explain=True)
    client = resources.explain_client
    if client is None:
        raise RuntimeError("Explain service unavailable")

    page_num = int(req.get("page_num", 1))
    term_id = str(req.get("term_id", "")).strip()
    detail = str(req.get("detail", "short"))
    audience = str(req.get("audience_level", "grad"))
    tgt = str(req.get("tgt") or doc.get("src_used") or "PM")
    src = str(req.get("src") or doc.get("src_used") or "CSM")

    page = next((p for p in doc.get("pages", []) if int(p.get("page_num", 0)) == page_num), None)
    if page is None:
        raise KeyError("page_not_found")

    terms = page.get("terms", []) if isinstance(page.get("terms", []), list) else []
    term_obj = None
    if term_id:
        term_obj = next((t for t in terms if str(t.get("term_id", "")) == term_id), None)
    if term_obj is None and req.get("term"):
        needle = str(req.get("term")).strip().lower()
        term_obj = next((t for t in terms if str(t.get("term", "")).lower() == needle), None)
    if term_obj is None:
        raise KeyError("term_not_found")

    analogs = [str(a.get("candidate", "")) for a in term_obj.get("analogs", []) if isinstance(a, dict)]
    text = str(req.get("text") or page.get("text") or "")

    ereq = ExplainRequest(
        text=_sanitize_output_text(text),
        term=_sanitize_output_text(str(term_obj.get("term", ""))),
        src=src,
        tgt=tgt,
        audience_level=audience,
        subtrack=str(req.get("subtrack", "")),
        analogs=analogs[:5],
        detail="long" if detail == "long" else "short",
    )
    out = client.explain(ereq)
    if isinstance(out, dict):
        out["page_num"] = page_num
        out["term_id"] = str(term_obj.get("term_id", ""))
    return out

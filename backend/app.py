from __future__ import annotations

import json
import asyncio
import os
import re
import sqlite3
import threading
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal
import hashlib
import uuid

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bandit import EpsilonGreedyBandit
from gemini_client import generate_with_gemini
from gpt_client import generate_with_gpt
from prompts import build_prompt, get_prompt_action_names
from reward import compute_reward
from semantic import get_semantic_mode, semantic_similarity
from llm.openai_client import ExplainRequest
from resources import ArtifactsMissingError, check_ready, get_resources, get_spacy
from pdf.extract import extract_pdf_pages, PdfExtractError, PdfEncryptedError, PdfEmptyTextError
from pdf.store import create_document_id, save_document, load_document
from pdf.annotate import annotate_pages, explain_term_from_document
from term_strategy import (
    TermStrategy,
    TermStrategyEngine,
    build_term_instruction_block,
    strategy_penalty,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

Domain = Literal["CSM", "PM", "CHEM", "CHEME", "CCE"]
SourceDomain = Literal["CSM", "PM", "CHEM", "CHEME", "CCE", "auto"]
AudienceLevel = Literal["undergrad", "grad", "expert"]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "domain_clf.joblib"
LEXICON_PATH = ROOT / "data" / "processed" / "domain_lexicon.json"
TERM_STATS_PATH = ROOT / "data" / "processed" / "term_stats.csv"
TERM_ALIASES_PATH = ROOT / "backend" / "term_aliases.json"
EXPLAIN_CACHE_DB = ROOT / "backend" / "explain_cache.sqlite3"
FEEDBACK_DB = ROOT / "backend" / "feedback.sqlite3"
ANALYTICS_LOG_PATH = Path(os.getenv("PRODUCT_ANALYTICS_LOG_PATH", str(ROOT / "logs" / "product_analytics.jsonl")))


def _get_cors_origins() -> list[str]:
    raw = os.getenv(
        "BACKEND_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["http://localhost:3000"]


class TranslateRequest(BaseModel):
    text: str = Field(min_length=3)
    src: Domain
    tgt: Domain
    k: int = Field(default=2, ge=2, le=8)


class AnnotateRequest(BaseModel):
    text: str = Field(min_length=3)
    src: SourceDomain = "auto"
    tgt: Domain
    audience_level: AudienceLevel = "grad"
    subtrack: str | None = None
    same_field_mode: Literal["normal", "study"] = "normal"
    max_terms: int = Field(default=8, ge=1, le=20)
    include_short_explanations: bool = False


class ExplainRequestBody(BaseModel):
    text: str = Field(min_length=3)
    term: str = Field(min_length=1)
    src: SourceDomain = "auto"
    tgt: Domain
    audience_level: AudienceLevel = "grad"
    subtrack: str | None = None
    analogs: list[str] = Field(default_factory=list)
    detail: Literal["short", "long"] = "short"


class PdfExplainRequestBody(BaseModel):
    document_id: str = Field(min_length=8)
    page_num: int = Field(ge=1)
    term_id: str | None = None
    term: str | None = None
    text: str | None = None
    src: SourceDomain = "auto"
    tgt: Domain
    audience_level: AudienceLevel = "grad"
    subtrack: str | None = None
    detail: Literal["short", "long"] = "short"


class FeedbackRequest(BaseModel):
    term: str = Field(min_length=1)
    src: SourceDomain = "auto"
    tgt: Domain
    selected_analog: str | None = None
    helpful: bool
    note: str | None = None


def _temperature_for_step(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.2
    # Sweep temperature from low to moderate across iterative calls.
    value = 0.2 + (0.7 * (step / (total_steps - 1)))
    return round(min(max(value, 0.2), 1.0), 2)


class CandidateScore(BaseModel):
    text: str
    total_score: float
    breakdown: dict[str, float]
    temperature: float
    action: str
    lex_terms_hit: list[str] = Field(default_factory=list)
    lex_terms_hit_style: list[str] = Field(default_factory=list)


class TermStrategyItem(BaseModel):
    term: str
    type: str
    native_score: float
    neighbor: str | None = None
    reason: str


class TranslateResponse(BaseModel):
    best_candidate: str
    best_score: float
    score_breakdown: dict[str, float]
    candidates: list[CandidateScore]
    prompt_action: str
    used_fallback: bool
    num_attempted: int
    num_returned: int
    cache_hit: bool = False
    term_strategies: list[TermStrategyItem] = Field(default_factory=list)
    src_warning: bool = False
    predicted_src: str | None = None
    predicted_src_confidence: float | None = None
    prompt_actions_used: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    semantic_mode: str = "overlap"
    lexicon_mode: str = "style"


app = FastAPI(title="SciBabel API", version="0.1.0")
cors_origins = _get_cors_origins()
allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None
lexicon_by_domain: dict[str, list[str]] = {}
term_log_odds: dict[tuple[str, str], dict[str, float]] = {}
term_strategy_engine: TermStrategyEngine | None = None
bandit = EpsilonGreedyBandit(actions=get_prompt_action_names(), epsilon=0.2)
_response_cache: dict[str, tuple[float, TranslateResponse]] = {}
semantic_enabled: bool = False
SCIBABEL_ENV = os.getenv("SCIBABEL_ENV", "dev").strip().lower()
if SCIBABEL_ENV not in {"dev", "production"}:
    SCIBABEL_ENV = "dev"
if os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"} and "SCIBABEL_ENV" not in os.environ:
    SCIBABEL_ENV = "production"
print(f"[startup] SCIBABEL_ENV={SCIBABEL_ENV}")

_ANNOTATE_DEFAULT_CONCURRENCY = "1" if SCIBABEL_ENV == "production" else "4"
ANNOTATE_MAX_CONCURRENCY = max(1, int(os.getenv("ANNOTATE_MAX_CONCURRENCY", _ANNOTATE_DEFAULT_CONCURRENCY)))
ANNOTATE_ACQUIRE_TIMEOUT_SEC = max(0.0, float(os.getenv("ANNOTATE_ACQUIRE_TIMEOUT_SEC", "0.1")))
ANNOTATE_TIMEOUT_SEC = max(0.1, float(os.getenv("ANNOTATE_TIMEOUT_SEC", "5")))
_ANNOTATE_SEMAPHORE = asyncio.Semaphore(ANNOTATE_MAX_CONCURRENCY)
print(f"[startup] ANNOTATE_MAX_CONCURRENCY={ANNOTATE_MAX_CONCURRENCY} ANNOTATE_TIMEOUT_SEC={ANNOTATE_TIMEOUT_SEC}")
_ANALYTICS_LOCK = threading.Lock()


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PRODUCT_ANALYTICS_ENABLED = _truthy_env("PRODUCT_ANALYTICS_ENABLED", default=True)


def _init_feedback_db() -> None:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS term_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                term TEXT NOT NULL,
                src TEXT NOT NULL,
                tgt TEXT NOT NULL,
                selected_analog TEXT,
                helpful INTEGER NOT NULL,
                note TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


ARTIFACT_HINT = (
    "Run make textmining-all locally and upload artifacts, "
    "or run scripts/textmining/build_artifacts.py"
)


def _artifacts_missing_response(missing: list[str]) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "artifacts_missing",
            "missing": missing,
            "hint": ARTIFACT_HINT,
        },
    )


def _annotate_log(*, request_id: str, started_at: float, status_code: int, error_reason: str | None = None) -> None:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    print(
        "[annotate_req]",
        {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "status_code": status_code,
            "error_reason": error_reason or "none",
        },
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _ANALYTICS_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _annotate_payload_meta(payload: AnnotateRequest) -> dict[str, object]:
    text = _sanitize_output_text(payload.text)
    return {
        "src": payload.src,
        "tgt": payload.tgt,
        "audience_level": payload.audience_level,
        "same_field_mode": payload.same_field_mode,
        "max_terms": int(payload.max_terms),
        "include_short_explanations": bool(payload.include_short_explanations),
        "text_chars": len(text),
        "text_words": len(text.split()),
    }


def _annotate_result_meta(result: dict[str, object]) -> dict[str, object]:
    terms = result.get("terms", []) if isinstance(result, dict) else []
    if not isinstance(terms, list):
        terms = []
    flagged = 0
    analog_total = 0
    evidence_total = 0
    short_explanations = 0
    for term in terms:
        if not isinstance(term, dict):
            continue
        if bool(term.get("flagged")):
            flagged += 1
        analogs = term.get("analogs", [])
        if isinstance(analogs, list):
            analog_total += len(analogs)
        evidence = term.get("evidence", [])
        if isinstance(evidence, list):
            evidence_total += len(evidence)
        if str(term.get("short_explanation") or "").strip():
            short_explanations += 1

    return {
        "src_used": result.get("src_used"),
        "predicted_src": result.get("predicted_src"),
        "src_warning": bool(result.get("src_warning", False)),
        "src_warning_reason": result.get("src_warning_reason"),
        "is_ambiguous": bool(result.get("is_ambiguous", False)),
        "total_terms": len(terms),
        "flagged_terms": flagged,
        "analog_suggestions": analog_total,
        "evidence_items": evidence_total,
        "short_explanations": short_explanations,
    }


def _explain_payload_meta(payload: ExplainRequestBody, src_effective: str) -> dict[str, object]:
    text = _sanitize_output_text(payload.text)
    term = _sanitize_output_text(payload.term)
    return {
        "src": payload.src,
        "src_effective": src_effective,
        "tgt": payload.tgt,
        "audience_level": payload.audience_level,
        "detail": payload.detail,
        "analogs_input": len(payload.analogs),
        "text_chars": len(text),
        "text_words": len(text.split()),
        "term_chars": len(term),
    }


def _explain_result_meta(result: dict[str, object]) -> dict[str, object]:
    short_text = str(result.get("short_explanation", "")) if isinstance(result, dict) else ""
    long_text = str(result.get("long_explanation", "")) if isinstance(result, dict) else ""
    return {
        "short_chars": len(short_text.strip()),
        "long_chars": len(long_text.strip()),
        "has_short": bool(short_text.strip()),
        "has_long": bool(long_text.strip()),
    }


def _log_product_event(
    *,
    event_type: str,
    request_id: str,
    started_at: float,
    status_code: int,
    payload_meta: dict[str, object],
    result_meta: dict[str, object] | None = None,
    error_reason: str | None = None,
) -> None:
    if not PRODUCT_ANALYTICS_ENABLED:
        return
    try:
        event = {
            "ts": int(time.time()),
            "env": SCIBABEL_ENV,
            "event_type": event_type,
            "request_id": request_id,
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "status_code": status_code,
            "error_reason": error_reason or "none",
            "payload": payload_meta,
            "result": result_meta or {},
        }
        _append_jsonl(ANALYTICS_LOG_PATH, event)
    except Exception:
        # Never fail the API request because analytics logging is unavailable.
        pass


async def _acquire_annotate_slot() -> bool:
    try:
        await asyncio.wait_for(_ANNOTATE_SEMAPHORE.acquire(), timeout=ANNOTATE_ACQUIRE_TIMEOUT_SEC)
        return True
    except asyncio.TimeoutError:
        return False


def _busy_response() -> JSONResponse:
    return JSONResponse(status_code=429, content={"error": "busy", "hint": "try again"})


def _timeout_budget_response(timeout_sec: float) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": "timeout_budget_exceeded", "timeout_sec": timeout_sec},
    )


def _generate_with_provider(prompt: str, temperature: float, top_p: float) -> str:
    provider = os.getenv("LLM_PROVIDER", "gpt").strip().lower()
    if provider in {"gpt", "openai"}:
        return generate_with_gpt(prompt=prompt, temperature=temperature, top_p=top_p)
    if provider == "gemini":
        return generate_with_gemini(prompt=prompt, temperature=temperature, top_p=top_p)
    raise ValueError("Unsupported LLM_PROVIDER. Use 'gpt' or 'gemini'.")


def _load_artifacts() -> None:
    global clf, lexicon_by_domain, term_log_odds, term_strategy_engine

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing classifier at {MODEL_PATH}. {ARTIFACT_HINT}."
        )
    if not LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"Missing lexicon at {LEXICON_PATH}. {ARTIFACT_HINT}."
        )
    if not TERM_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Missing term stats at {TERM_STATS_PATH}. {ARTIFACT_HINT}."
        )

    clf = joblib.load(MODEL_PATH)
    lexicon_raw = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    lexicon_by_domain = {}
    for d in ["CSM", "PM", "CHEM", "CHEME", "CCE"]:
        val = lexicon_raw.get(d, [])
        if isinstance(val, list):
            lexicon_by_domain[d] = [str(x) for x in val]
            continue
        if isinstance(val, dict):
            # Prefer bigrams/trigrams for steering and lex scoring.
            merged = (
                [str(x) for x in val.get("bigrams", [])]
                + [str(x) for x in val.get("trigrams", [])]
                + [str(x) for x in val.get("style", [])]
                + [str(x) for x in val.get("top_bigrams", [])]
                + [str(x) for x in val.get("top_trigrams", [])]
                + [str(x) for x in val.get("top_terms", [])]
            )
            seen = set()
            deduped = []
            for t in merged:
                tl = t.lower()
                if tl in seen:
                    continue
                deduped.append(t)
                seen.add(tl)
            lexicon_by_domain[d] = deduped
            continue
        lexicon_by_domain[d] = []

    term_log_odds = {}
    if TERM_STATS_PATH.exists():
        with TERM_STATS_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = row.get("domain", "")
                term = row.get("term", "")
                if not domain or not term:
                    continue
                try:
                    z = float(row.get("z", row.get("log_odds", "0")) or 0)
                except ValueError:
                    z = 0.0
                try:
                    delta = float(row.get("delta", row.get("log_odds", "0")) or 0)
                except ValueError:
                    delta = 0.0
                try:
                    ng = float(row.get("ngram_len", str(max(1, len(term.split())))) or 1)
                except ValueError:
                    ng = 1.0
                term_log_odds[(domain, term.lower())] = {
                    "z": z,
                    "delta": delta,
                    "ngram_len": ng,
                }

    term_strategy_engine = TermStrategyEngine(
        lexicon_by_domain=lexicon_by_domain,
        aliases_path=TERM_ALIASES_PATH,
        term_log_odds=term_log_odds,
    )


def _normalize_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _cache_key(text: str, src: str, tgt: str, k: int) -> str:
    raw = f"{src}|{tgt}|{k}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rule_based_fallback_translation(text: str, tgt: str, strategies: list[TermStrategy]) -> str:
    """Deterministic local fallback when LLM provider is unavailable.

    Keeps semantics mostly intact while applying lightweight domain-term reframing.
    """
    out = text

    for s in strategies:
        term_pat = re.compile(rf"\\b{re.escape(s.term)}\\b", re.IGNORECASE)
        if s.type == "analogous" and s.neighbor:
            out = term_pat.sub(s.neighbor, out)

    lead_map = {
        "CSM": "From a computational modeling perspective, ",
        "PM": "From a physics-oriented interpretation, ",
        "CHEM": "From a chemistry perspective, ",
        "CHEME": "From a chemical engineering standpoint, ",
        "CCE": "From a process-engineering standpoint, ",
    }
    lead = lead_map.get(tgt, f"In {tgt} terms, ")
    lowered = out[0].lower() + out[1:] if len(out) > 1 else out
    out = f"{lead}{lowered}"

    return out


def _pick_fallback_reason(reasons: set[str]) -> str | None:
    priority = ["no_key", "timeout", "api_error", "filter_empty", "other"]
    for key in priority:
        if key in reasons:
            return key
    return None


def _sanitize_output_text(text: str) -> str:
    out = text
    out = re.sub(r"\(\s*domain-specific concept\s*\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\(\s*native\s*=\s*[^\)]*\)", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\[\s*updatedgpt[^\]]*\]", "", out, flags=re.IGNORECASE)
    out = re.sub(r"updatedgpt[_\-a-z0-9]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "OK"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    t0 = time.perf_counter()
    out = check_ready()
    print(f"[timing] ready_total_sec={round(time.perf_counter() - t0, 4)} ready={out.get('ready')}")
    return out


@app.post("/annotate")
async def annotate(payload: AnnotateRequest):
    request_id = uuid.uuid4().hex[:10]
    started = time.perf_counter()
    payload_meta = _annotate_payload_meta(payload)
    acquired = await _acquire_annotate_slot()
    if not acquired:
        _annotate_log(request_id=request_id, started_at=started, status_code=429, error_reason="busy")
        _log_product_event(
            event_type="annotate",
            request_id=request_id,
            started_at=started,
            status_code=429,
            payload_meta=payload_meta,
            error_reason="busy",
        )
        return _busy_response()
    try:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_annotate_impl_sync, payload, False),
                timeout=ANNOTATE_TIMEOUT_SEC,
            )
            _annotate_log(request_id=request_id, started_at=started, status_code=200)
            _log_product_event(
                event_type="annotate",
                request_id=request_id,
                started_at=started,
                status_code=200,
                payload_meta=payload_meta,
                result_meta=_annotate_result_meta(result),
            )
            return result
        except asyncio.TimeoutError:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="timeout_budget_exceeded")
            _log_product_event(
                event_type="annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="timeout_budget_exceeded",
            )
            return _timeout_budget_response(ANNOTATE_TIMEOUT_SEC)
        except ArtifactsMissingError as exc:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="artifacts_missing")
            _log_product_event(
                event_type="annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="artifacts_missing",
            )
            return _artifacts_missing_response(exc.missing)
        except Exception as exc:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="annotate_failed")
            _log_product_event(
                event_type="annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="annotate_failed",
            )
            return JSONResponse(status_code=503, content={"error": "annotate_failed", "detail": str(exc)})
    finally:
        _ANNOTATE_SEMAPHORE.release()


@app.post("/pdf/annotate")
async def pdf_annotate(
    file: UploadFile = File(...),
    src: SourceDomain = Form("auto"),
    tgt: Domain = Form(...),
    audience_level: AudienceLevel = Form("grad"),
    subtrack: str | None = Form(None),
    same_field_mode: Literal["normal", "study"] = Form("normal"),
    max_terms: int = Form(8),
) -> dict[str, object]:
    if (file.filename or "").lower().endswith(".pdf") is False:
        raise HTTPException(status_code=422, detail="file must be a .pdf")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty file")

    max_file_mb = max(1, int(os.getenv("PDF_MAX_FILE_MB", "20")))
    if len(data) > max_file_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"file too large (> {max_file_mb}MB)")

    max_pages = max(1, int(os.getenv("PDF_MAX_PAGES", "40")))
    try:
        extracted = await asyncio.to_thread(extract_pdf_pages, data, max_pages=max_pages, timeout_sec=15.0)
    except PdfEncryptedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PdfEmptyTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PdfExtractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"pdf_extract_failed: {exc}") from exc

    document_id = create_document_id()
    pages = [{"page_num": p.page_num, "text": p.text} for p in extracted]
    payload = {
        "src": src,
        "tgt": tgt,
        "audience_level": audience_level,
        "subtrack": subtrack,
        "same_field_mode": same_field_mode,
    }
    max_terms_total = max(1, min(200, int(max_terms) * max(1, len(pages))))
    result = await asyncio.to_thread(
        annotate_pages,
        document_id=document_id,
        filename=file.filename or "upload.pdf",
        pages=pages,
        payload=payload,
        max_terms_per_page=max(1, min(20, int(max_terms))),
        max_terms_total=max_terms_total,
    )
    save_document(result)
    return result


@app.get("/pdf/document/{document_id}")
def pdf_get_document(document_id: str) -> dict[str, object]:
    doc = load_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")
    return doc


@app.post("/pdf/explain")
def pdf_explain(payload: PdfExplainRequestBody) -> dict[str, object]:
    doc = load_document(payload.document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document_not_found")

    req = {
        "page_num": payload.page_num,
        "term_id": payload.term_id,
        "term": payload.term,
        "text": payload.text,
        "src": payload.src,
        "tgt": payload.tgt,
        "audience_level": payload.audience_level,
        "subtrack": payload.subtrack,
        "detail": payload.detail,
    }
    try:
        out = explain_term_from_document(doc, req)
    except KeyError as exc:
        code = str(exc)
        if "page_not_found" in code:
            raise HTTPException(status_code=404, detail="page_not_found") from exc
        if "term_not_found" in code:
            raise HTTPException(status_code=404, detail="term_not_found") from exc
        raise HTTPException(status_code=422, detail="bad_request") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"explain_failed: {exc}") from exc
    return out


@app.post("/profile_annotate")
async def profile_annotate(payload: AnnotateRequest):
    request_id = uuid.uuid4().hex[:10]
    started = time.perf_counter()
    payload_meta = _annotate_payload_meta(payload)
    acquired = await _acquire_annotate_slot()
    if not acquired:
        _annotate_log(request_id=request_id, started_at=started, status_code=429, error_reason="busy")
        _log_product_event(
            event_type="profile_annotate",
            request_id=request_id,
            started_at=started,
            status_code=429,
            payload_meta=payload_meta,
            error_reason="busy",
        )
        return _busy_response()
    try:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_annotate_impl_sync, payload, True),
                timeout=ANNOTATE_TIMEOUT_SEC,
            )
            _annotate_log(request_id=request_id, started_at=started, status_code=200)
            _log_product_event(
                event_type="profile_annotate",
                request_id=request_id,
                started_at=started,
                status_code=200,
                payload_meta=payload_meta,
                result_meta=_annotate_result_meta(result),
            )
            return result
        except asyncio.TimeoutError:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="timeout_budget_exceeded")
            _log_product_event(
                event_type="profile_annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="timeout_budget_exceeded",
            )
            return _timeout_budget_response(ANNOTATE_TIMEOUT_SEC)
        except ArtifactsMissingError as exc:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="artifacts_missing")
            _log_product_event(
                event_type="profile_annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="artifacts_missing",
            )
            return _artifacts_missing_response(exc.missing)
        except Exception as exc:
            _annotate_log(request_id=request_id, started_at=started, status_code=503, error_reason="annotate_failed")
            _log_product_event(
                event_type="profile_annotate",
                request_id=request_id,
                started_at=started,
                status_code=503,
                payload_meta=payload_meta,
                error_reason="annotate_failed",
            )
            return JSONResponse(status_code=503, content={"error": "annotate_failed", "detail": str(exc)})
    finally:
        _ANNOTATE_SEMAPHORE.release()


def _annotate_impl_sync(payload: AnnotateRequest, include_profile: bool = False) -> dict[str, object]:
    t_all = time.perf_counter()
    effective_max_terms = int(payload.max_terms)
    if SCIBABEL_ENV == "production":
        cap = int(os.getenv("PRODUCTION_MAX_TERMS", "6"))
        effective_max_terms = max(1, min(effective_max_terms, cap))

    t0 = time.perf_counter()
    resources = get_resources(load_explain=payload.include_short_explanations)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = get_spacy()
    t_spacy = time.perf_counter() - t0

    engine = resources.annotation_engine
    detector = resources.source_detector
    det = detector.detect_source(_sanitize_output_text(payload.text))

    src_warning = False
    src_warning_reason = "none"
    src_used = str(payload.src)

    if payload.src == "auto":
        src_used = str(det.get("predicted_src") or "CSM")
        if bool(det.get("is_ambiguous", False)):
            src_warning = True
            src_warning_reason = str(det.get("reason") or "ambiguous")
        print(
            "domain_detect:",
            {
                "predicted": det.get("predicted_src"),
                "conf": det.get("confidence"),
                "top2_gap": det.get("top2_gap"),
                "ambiguous": det.get("is_ambiguous"),
                "reason": det.get("reason"),
            },
        )
    else:
        src_used = str(payload.src)
        pred = str(det.get("predicted_src") or "")
        conf = float(det.get("confidence") or 0.0)
        if pred and pred != src_used and conf >= 0.65:
            src_warning = True
            src_warning_reason = "mismatch"

    text_clean = _sanitize_output_text(payload.text)
    try:
        out = engine.annotate(
            text=text_clean,
            src=src_used,
            tgt=payload.tgt,
            max_terms=effective_max_terms,
            same_field_mode=payload.same_field_mode,
        )
    except Exception as exc:
        raise RuntimeError(f"Annotate failed: {exc}") from exc

    if payload.include_short_explanations:
        client = resources.explain_client
        if client is None:
            raise HTTPException(status_code=503, detail="Explain service unavailable")
        for term in out.get("terms", []):
            if not isinstance(term, dict) or not term.get("flagged"):
                continue
            analogs = [str(x.get("candidate", "")) for x in term.get("analogs", []) if isinstance(x, dict)]
            req = ExplainRequest(
                text=payload.text,
                term=str(term.get("term", "")),
                src=src_used,
                tgt=payload.tgt,
                audience_level=payload.audience_level,
                subtrack=payload.subtrack or "",
                analogs=analogs,
                detail="short",
            )
            try:
                explained = client.explain(req)
                term["short_explanation"] = explained.get("short_explanation", "")
            except Exception:
                term["short_explanation"] = ""

    response = {
        "predicted_src": det.get("predicted_src"),
        "predicted_src_confidence": det.get("confidence"),
        "predicted_src_probs": det.get("probs", {}),
        "src_used": src_used,
        "src_warning": src_warning,
        "src_warning_reason": src_warning_reason,
        "is_ambiguous": bool(det.get("is_ambiguous", False)),
        "top2_gap": det.get("top2_gap"),
        "suggested_src": det.get("predicted_src"),
        "terms": out.get("terms", []),
    }

    step_t = out.get("_timings", {}) if isinstance(out, dict) else {}
    timings = {
        "load_light_resources": round(t_load, 4),
        "get_spacy": round(t_spacy, 4),
        "spacy_extract": float(step_t.get("spacy_extract_sec", 0.0)),
        "yake_extract": float(step_t.get("yake_extract_sec", 0.0)),
        "merge_dedupe": float(step_t.get("merge_dedupe_sec", 0.0)),
        "score_terms": float(step_t.get("score_terms_sec", 0.0)),
        "analog_suggest": float(step_t.get("analog_search_sec", 0.0)),
        "evidence": float(step_t.get("evidence_sec", 0.0)),
        "total": round(time.perf_counter() - t_all, 4),
    }
    print(
        "[timing] annotate",
        {
            "env": SCIBABEL_ENV,
            "load_light_resources": timings["load_light_resources"],
            "get_spacy": timings["get_spacy"],
            "spacy_extract": timings["spacy_extract"],
            "yake_extract": timings["yake_extract"],
            "merge_dedupe": timings["merge_dedupe"],
            "score_terms": timings["score_terms"],
            "analog_suggest": timings["analog_suggest"],
            "evidence": timings["evidence"],
            "total": timings["total"],
            "max_terms": effective_max_terms,
        },
    )
    if include_profile:
        response["timings"] = timings
    return response


@app.post("/detect_source")
def detect_source(payload: dict[str, str]) -> dict[str, object]:
    text = str(payload.get("text", "")).strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="text is required")
    try:
        detector = get_resources(load_explain=False).source_detector
    except ArtifactsMissingError as exc:
        return _artifacts_missing_response(exc.missing)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Source detector unavailable: {exc}") from exc
    return detector.detect_source(_sanitize_output_text(text))


@app.post("/explain")
def explain(payload: ExplainRequestBody) -> dict[str, object]:
    request_id = uuid.uuid4().hex[:10]
    started = time.perf_counter()
    try:
        resources = get_resources(load_explain=True)
    except ArtifactsMissingError as exc:
        _log_product_event(
            event_type="explain",
            request_id=request_id,
            started_at=started,
            status_code=503,
            payload_meta={"src": payload.src, "tgt": payload.tgt, "detail": payload.detail},
            error_reason="artifacts_missing",
        )
        return _artifacts_missing_response(exc.missing)
    except Exception as exc:
        _log_product_event(
            event_type="explain",
            request_id=request_id,
            started_at=started,
            status_code=503,
            payload_meta={"src": payload.src, "tgt": payload.tgt, "detail": payload.detail},
            error_reason="explain_unavailable",
        )
        raise HTTPException(status_code=503, detail=f"Explain service unavailable: {exc}") from exc

    client = resources.explain_client
    detector = resources.source_detector
    if client is None:
        _log_product_event(
            event_type="explain",
            request_id=request_id,
            started_at=started,
            status_code=503,
            payload_meta={"src": payload.src, "tgt": payload.tgt, "detail": payload.detail},
            error_reason="explain_unavailable",
        )
        raise HTTPException(status_code=503, detail="Explain service unavailable")

    src_effective = payload.src
    if payload.src == "auto":
        det = detector.detect_source(_sanitize_output_text(payload.text))
        pred = str(det.get("predicted_src") or "")
        src_effective = pred if pred in {"CSM", "PM", "CHEM", "CHEME", "CCE"} else "CSM"

    req = ExplainRequest(
        text=_sanitize_output_text(payload.text),
        term=_sanitize_output_text(payload.term),
        src=str(src_effective),
        tgt=payload.tgt,
        audience_level=payload.audience_level,
        subtrack=payload.subtrack or "",
        analogs=[_sanitize_output_text(a) for a in payload.analogs[:5]],
        detail=payload.detail,
    )
    payload_meta = _explain_payload_meta(payload, str(src_effective))

    try:
        out = client.explain(req)
    except Exception as exc:
        _log_product_event(
            event_type="explain",
            request_id=request_id,
            started_at=started,
            status_code=503,
            payload_meta=payload_meta,
            error_reason="explain_failed",
        )
        raise HTTPException(status_code=503, detail=f"Explain request failed: {exc}") from exc

    _log_product_event(
        event_type="explain",
        request_id=request_id,
        started_at=started,
        status_code=200,
        payload_meta=payload_meta,
        result_meta=_explain_result_meta(out),
    )

    return out


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, object]:
    try:
        _init_feedback_db()
        conn = sqlite3.connect(FEEDBACK_DB)
        try:
            conn.execute(
                """
                INSERT INTO term_feedback(created_at, term, src, tgt, selected_analog, helpful, note)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    time.time(),
                    _sanitize_output_text(payload.term),
                    payload.src,
                    payload.tgt,
                    _sanitize_output_text(payload.selected_analog or "") or None,
                    1 if payload.helpful else 0,
                    _sanitize_output_text(payload.note or "") or None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {exc}") from exc

    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    global clf, lexicon_by_domain, term_strategy_engine, semantic_enabled

    if clf is None or not lexicon_by_domain:
        try:
            _load_artifacts()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    if term_strategy_engine is None:
        try:
            _load_artifacts()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    max_input_chars = max(128, int(os.getenv("GEMINI_MAX_INPUT_CHARS", "1200")))
    normalized_text = _normalize_text(payload.text, max_input_chars)

    # Fast path: same-domain translation needs no expensive generation.
    if payload.src == payload.tgt:
        reward = compute_reward(
            source_text=normalized_text,
            candidate=normalized_text,
            tgt=payload.tgt,
            clf=clf,
            lexicon_by_domain=lexicon_by_domain,
            semantic_similarity_fn=semantic_similarity,
            term_log_odds=term_log_odds,
            min_semantic_sim=0.0,
        )
        candidate = CandidateScore(
            text=normalized_text,
            total_score=reward.total,
            breakdown={
                "domain": reward.breakdown.domain,
                "meaning": reward.breakdown.meaning,
                "lex": reward.breakdown.lex,
                "semantic_sim": reward.breakdown.semantic_sim,
                "copy_score": reward.breakdown.copy_score,
                "copy_penalty": reward.breakdown.copy_penalty,
            },
            temperature=0.0,
            action="identity",
            lex_terms_hit=reward.breakdown.lex_terms_hit,
            lex_terms_hit_style=reward.breakdown.lex_terms_hit,
        )
        return TranslateResponse(
            best_candidate=candidate.text,
            best_score=candidate.total_score,
            score_breakdown=candidate.breakdown,
            candidates=[candidate],
            prompt_action="identity",
            used_fallback=False,
            num_attempted=0,
            num_returned=1,
            cache_hit=True,
            prompt_actions_used=["identity"],
            fallback_reason=None,
            semantic_mode=get_semantic_mode(),
        )

    cache_ttl_sec = max(0, int(os.getenv("CACHE_TTL_SEC", "3600")))
    request_key = _cache_key(normalized_text, payload.src, payload.tgt, payload.k)
    if cache_ttl_sec > 0 and request_key in _response_cache:
        cached_ts, cached_resp = _response_cache[request_key]
        if (time.time() - cached_ts) <= cache_ttl_sec:
            return cached_resp.model_copy(update={"cache_hit": True})

    key_terms = term_strategy_engine.extract_key_terms(normalized_text, max_terms=10)
    strategies: list[TermStrategy] = [
        term_strategy_engine.classify_term(kt.term, payload.tgt) for kt in key_terms
    ]
    term_instruction_block = build_term_instruction_block(strategies, max_terms=8)

    route_key = f"{payload.src}->{payload.tgt}"
    all_actions = get_prompt_action_names()
    if payload.k <= len(all_actions):
        actions_for_slots = all_actions[: payload.k]
    else:
        actions_for_slots = list(all_actions)
        while len(actions_for_slots) < payload.k:
            actions_for_slots.append(bandit.choose(route_key))

    num_generations = payload.k
    max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "0")))
    retry_sleep = float(os.getenv("GEMINI_RETRY_SLEEP_SEC", "1.5"))
    strategy_penalty_weight = float(os.getenv("STRATEGY_PENALTY_WEIGHT", "0.15"))
    min_semantic_sim = float(os.getenv("MIN_SEMANTIC_SIM", "0.78"))
    if not semantic_enabled:
        min_semantic_sim = min(
            min_semantic_sim,
            float(os.getenv("MIN_SEMANTIC_SIM_FALLBACK", "0.55")),
        )
    alpha_lex = float(os.getenv("ALPHA_LEX", "0.35"))
    beta_copy = float(os.getenv("BETA_COPY", "0.5"))
    copy_threshold = float(os.getenv("COPY_THRESHOLD", "0.86"))
    lex_score_clamp = float(os.getenv("LEX_SCORE_CLAMP", "2.0"))
    src_warning_threshold = float(os.getenv("SRC_WARNING_CONF", "0.55"))
    target_lex_hints = lexicon_by_domain.get(payload.tgt, [])[:20]

    candidate_pool: dict[str, CandidateScore] = {}
    used_fallback = False
    num_attempted = 0
    fallback_reasons: set[str] = set()

    src_labels = list(getattr(clf, "classes_", []))
    src_probs = clf.predict_proba([normalized_text])[0]
    src_pred_idx = int(src_probs.argmax())
    predicted_src = str(src_labels[src_pred_idx]) if src_labels else None
    predicted_src_conf = float(src_probs[src_pred_idx]) if src_labels else None
    src_warning = bool(
        predicted_src
        and predicted_src != payload.src
        and (predicted_src_conf is not None and predicted_src_conf >= src_warning_threshold)
    )

    def _generate_one(action_name: str, temp: float) -> tuple[str, str, int, bool, str | None]:
        local_attempts = 0
        prompt = build_prompt(
            action=action_name,
            text=normalized_text,
            src=payload.src,
            tgt=payload.tgt,
            term_instructions=term_instruction_block,
            target_lexicon_hints=target_lex_hints,
        )
        for retry_idx in range(max_retries + 1):
            local_attempts += 1
            try:
                text = _generate_with_provider(prompt=prompt, temperature=temp, top_p=0.95)
                return text, action_name, local_attempts, False, None
            except NotImplementedError as exc:
                fallback_text = _rule_based_fallback_translation(
                    text=normalized_text,
                    tgt=payload.tgt,
                    strategies=strategies,
                )
                msg = str(exc).lower()
                reason = "no_key" if ("api_key" in msg or "not configured" in msg) else "api_error"
                return fallback_text, action_name, local_attempts, True, reason
            except Exception as exc:
                message = str(exc)
                quota_like = ("429" in message) or ("RESOURCE_EXHAUSTED" in message)
                timeout_like = ("timeout" in message.lower()) or ("timed out" in message.lower())
                if quota_like and retry_idx < max_retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
                    continue
                if quota_like:
                    return normalized_text, action_name, local_attempts, True, "api_error"
                # Non-quota generation error: degrade gracefully for this slot.
                if timeout_like:
                    return normalized_text, action_name, local_attempts, True, "timeout"
                return normalized_text, action_name, local_attempts, True, "other"

    temperatures = [_temperature_for_step(i, max(2, num_generations)) for i in range(num_generations)]

    with ThreadPoolExecutor(max_workers=max(1, num_generations)) as executor:
        future_map = {
            executor.submit(_generate_one, a, t): (a, t)
            for a, t in zip(actions_for_slots, temperatures)
        }
        for future in as_completed(future_map):
            action_name, temp = future_map[future]
            candidate, action_name, attempts, slot_fallback, slot_reason = future.result()
            num_attempted += attempts
            used_fallback = used_fallback or slot_fallback
            if slot_fallback and slot_reason:
                fallback_reasons.add(slot_reason)

            if not candidate.strip():
                candidate = normalized_text
                used_fallback = True
                fallback_reasons.add("other")

            candidate = _sanitize_output_text(candidate)

            reward = compute_reward(
                source_text=normalized_text,
                candidate=candidate,
                tgt=payload.tgt,
                clf=clf,
                lexicon_by_domain=lexicon_by_domain,
                semantic_similarity_fn=semantic_similarity,
                term_log_odds=term_log_odds,
                min_semantic_sim=min_semantic_sim,
                alpha_lex=alpha_lex,
                beta_copy=beta_copy,
                copy_threshold=copy_threshold,
                lex_score_clamp=lex_score_clamp,
            )
            if not reward.eligible:
                continue

            strat_pen = strategy_penalty(candidate, strategies)
            final_score = reward.total - (strategy_penalty_weight * strat_pen)

            scored_candidate = CandidateScore(
                text=candidate,
                total_score=final_score,
                breakdown={
                    "domain": reward.breakdown.domain,
                    "meaning": reward.breakdown.meaning,
                    "lex": reward.breakdown.lex,
                    "semantic_sim": reward.breakdown.semantic_sim,
                    "copy_score": reward.breakdown.copy_score,
                    "copy_penalty": reward.breakdown.copy_penalty,
                    "strategy_penalty": strat_pen,
                },
                temperature=temp,
                action=action_name,
                lex_terms_hit=reward.breakdown.lex_terms_hit,
                lex_terms_hit_style=reward.breakdown.lex_terms_hit,
            )

            existing = candidate_pool.get(scored_candidate.text)
            if existing is None or scored_candidate.total_score > existing.total_score:
                candidate_pool[scored_candidate.text] = scored_candidate

    # Keep only top-k after parallel generation.
    top_k = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)[: payload.k]
    candidate_pool = {c.text: c for c in top_k}

    scored = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)

    if not scored:
        fallback_reward = compute_reward(
            source_text=normalized_text,
            candidate=normalized_text,
            tgt=payload.tgt,
            clf=clf,
            lexicon_by_domain=lexicon_by_domain,
            semantic_similarity_fn=semantic_similarity,
            term_log_odds=term_log_odds,
            min_semantic_sim=0.0,
        )
        scored = [
            CandidateScore(
                text=normalized_text,
                total_score=fallback_reward.total,
                breakdown={
                    "domain": fallback_reward.breakdown.domain,
                    "meaning": fallback_reward.breakdown.meaning,
                    "lex": fallback_reward.breakdown.lex,
                    "semantic_sim": fallback_reward.breakdown.semantic_sim,
                    "copy_score": fallback_reward.breakdown.copy_score,
                    "copy_penalty": fallback_reward.breakdown.copy_penalty,
                    "strategy_penalty": 0.0,
                },
                temperature=0.0,
                action="fallback_identity",
                lex_terms_hit=fallback_reward.breakdown.lex_terms_hit,
                lex_terms_hit_style=fallback_reward.breakdown.lex_terms_hit,
            )
        ]
        used_fallback = True
        fallback_reasons.add("filter_empty")

    best = scored[0]
    if best.action in get_prompt_action_names():
        bandit.update(route_key, best.action, best.total_score)

    response = TranslateResponse(
        best_candidate=_sanitize_output_text(best.text),
        best_score=best.total_score,
        score_breakdown=best.breakdown,
        candidates=scored,
        prompt_action=best.action,
        used_fallback=used_fallback,
        num_attempted=num_attempted,
        num_returned=len(scored),
        cache_hit=False,
        src_warning=src_warning,
        predicted_src=predicted_src,
        predicted_src_confidence=predicted_src_conf,
        prompt_actions_used=actions_for_slots,
        fallback_reason=_pick_fallback_reason(fallback_reasons),
        semantic_mode=get_semantic_mode(),
        lexicon_mode="style",
        term_strategies=[
            TermStrategyItem(
                term=s.term,
                type=s.type,
                native_score=s.native_score,
                neighbor=s.neighbor,
                reason=s.reason,
            )
            for s in strategies
        ],
    )
    if cache_ttl_sec > 0:
        _response_cache[request_key] = (time.time(), response)
    return response


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)

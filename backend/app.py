from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal
import hashlib

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bandit import EpsilonGreedyBandit
from gemini_client import generate_with_gemini
from gpt_client import generate_with_gpt
from prompts import build_prompt, get_prompt_action_names
from reward import compute_reward
from semantic import get_semantic_mode, init_semantic_model, semantic_similarity
from llm.fake_client import FakeExplainClient
from llm.openai_client import ExplainRequest, OpenAIExplainClient
from domain_detect import SourceDetector
from terms.engine import AnnotationArtifactsMissing, TermAnnotationEngine
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
annotation_engine: TermAnnotationEngine | None = None
explain_client: OpenAIExplainClient | FakeExplainClient | None = None
source_detector: SourceDetector | None = None


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


def _load_annotation_engine() -> None:
    global annotation_engine
    src_th = float(os.getenv("UNFAMILIAR_SRC_THRESHOLD", "0.35"))
    tgt_th = float(os.getenv("UNFAMILIAR_TGT_THRESHOLD", "0.45"))
    analog_th = float(os.getenv("ANALOG_SIM_THRESHOLD", "0.20"))
    annotation_engine = TermAnnotationEngine(
        root=ROOT,
        src_threshold=src_th,
        tgt_threshold=tgt_th,
        analog_threshold=analog_th,
    )


def _load_explain_client() -> None:
    global explain_client
    if os.getenv("SCIBABEL_FAKE_LLM", "0") == "1":
        explain_client = FakeExplainClient()
        return
    explain_client = OpenAIExplainClient(db_path=EXPLAIN_CACHE_DB)


def _load_source_detector() -> None:
    global source_detector
    source_detector = SourceDetector(model_path=MODEL_PATH)


def _ensure_source_detector() -> SourceDetector:
    global source_detector
    if source_detector is None:
        try:
            _load_source_detector()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Source detector unavailable: {exc}") from exc
    if source_detector is None:
        raise HTTPException(status_code=503, detail="Source detector unavailable")
    return source_detector


def _generate_with_provider(prompt: str, temperature: float, top_p: float) -> str:
    provider = os.getenv("LLM_PROVIDER", "gpt").strip().lower()
    if provider in {"gpt", "openai"}:
        return generate_with_gpt(prompt=prompt, temperature=temperature, top_p=top_p)
    if provider == "gemini":
        return generate_with_gemini(prompt=prompt, temperature=temperature, top_p=top_p)
    raise ValueError("Unsupported LLM_PROVIDER. Use 'gpt' or 'gemini'.")


def _load_artifacts() -> None:
    global clf, lexicon_by_domain, term_log_odds, term_strategy_engine

    if not MODEL_PATH.exists() or not LEXICON_PATH.exists():
        sample_corpus = ROOT / "data" / "processed" / "sample_corpus.jsonl"
        if not sample_corpus.exists():
            raise FileNotFoundError(
                f"Missing sample corpus at {sample_corpus}; cannot bootstrap artifacts."
            )

        # Bootstrap lightweight artifacts at runtime for first deploy.
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "03_mine_terms.py"),
                "--corpus",
                str(sample_corpus),
                "--lexicon-out",
                str(LEXICON_PATH),
                "--term-stats-out",
                str(ROOT / "data" / "processed" / "term_stats.csv"),
                "--top-n",
                "120",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "04_train_domain_classifier.py"),
                "--corpus",
                str(sample_corpus),
                "--model-out",
                str(MODEL_PATH),
            ],
            check=True,
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing classifier at {MODEL_PATH}. Run scripts/04_train_domain_classifier.py first."
        )
    if not LEXICON_PATH.exists():
        raise FileNotFoundError(
            f"Missing lexicon at {LEXICON_PATH}. Run scripts/03_mine_terms.py first."
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


@app.on_event("startup")
def startup_event() -> None:
    global semantic_enabled
    try:
        _load_artifacts()
    except Exception as exc:
        # Delay hard failure; endpoint returns actionable message.
        print(f"[startup] Artifact load warning: {exc}")

    semantic_model_name = os.getenv("SEMANTIC_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    semantic_enabled = init_semantic_model(semantic_model_name)

    try:
        _load_annotation_engine()
    except Exception as exc:
        print(f"[startup] Annotation engine warning: {exc}")

    try:
        _load_explain_client()
    except Exception as exc:
        print(f"[startup] Explain client warning: {exc}")

    try:
        _load_source_detector()
    except Exception as exc:
        print(f"[startup] Source detector warning: {exc}")

    try:
        _init_feedback_db()
    except Exception as exc:
        print(f"[startup] Feedback DB warning: {exc}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "OK"}


def _ensure_annotation_ready() -> TermAnnotationEngine:
    global annotation_engine
    if annotation_engine is None:
        try:
            _load_annotation_engine()
        except AnnotationArtifactsMissing as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Annotation engine unavailable: {exc}") from exc
    if annotation_engine is None:
        raise HTTPException(status_code=503, detail="Annotation engine unavailable. Run make textmining-all.")
    return annotation_engine


def _ensure_explain_ready() -> OpenAIExplainClient | FakeExplainClient:
    global explain_client
    if explain_client is None:
        try:
            _load_explain_client()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Explain service unavailable: {exc}") from exc
    if explain_client is None:
        raise HTTPException(status_code=503, detail="Explain service unavailable.")
    return explain_client


@app.post("/annotate")
def annotate(payload: AnnotateRequest) -> dict[str, object]:
    engine = _ensure_annotation_ready()
    detector = _ensure_source_detector()
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

    out = engine.annotate(
        text=_sanitize_output_text(payload.text),
        src=src_used,
        tgt=payload.tgt,
        max_terms=payload.max_terms,
    )

    if payload.include_short_explanations:
        client = _ensure_explain_ready()
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

    return {
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


@app.post("/detect_source")
def detect_source(payload: dict[str, str]) -> dict[str, object]:
    text = str(payload.get("text", "")).strip()
    if len(text) < 3:
        raise HTTPException(status_code=422, detail="text is required")
    detector = _ensure_source_detector()
    return detector.detect_source(_sanitize_output_text(text))


@app.post("/explain")
def explain(payload: ExplainRequestBody) -> dict[str, object]:
    _ensure_annotation_ready()
    client = _ensure_explain_ready()
    detector = _ensure_source_detector()

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

    try:
        out = client.explain(req)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Explain request failed: {exc}") from exc

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

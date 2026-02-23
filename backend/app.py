from __future__ import annotations

import json
import os
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
from term_strategy import (
    TermStrategy,
    TermStrategyEngine,
    build_term_instruction_block,
    strategy_penalty,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

Domain = Literal["CSM", "PM", "CCE"]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "domain_clf.joblib"
LEXICON_PATH = ROOT / "data" / "processed" / "domain_lexicon.json"
TERM_STATS_PATH = ROOT / "data" / "processed" / "term_stats.csv"
TERM_ALIASES_PATH = ROOT / "backend" / "term_aliases.json"


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
term_log_odds: dict[tuple[str, str], float] = {}
term_strategy_engine: TermStrategyEngine | None = None
bandit = EpsilonGreedyBandit(actions=get_prompt_action_names(), epsilon=0.2)
_response_cache: dict[str, tuple[float, TranslateResponse]] = {}


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
    lexicon_by_domain = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))

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
                    score = float(row.get("log_odds", "0") or 0)
                except ValueError:
                    score = 0.0
                term_log_odds[(domain, term.lower())] = score

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


@app.on_event("startup")
def startup_event() -> None:
    try:
        _load_artifacts()
    except Exception as exc:
        # Delay hard failure; endpoint returns actionable message.
        print(f"[startup] Artifact load warning: {exc}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "OK"}


@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    global clf, lexicon_by_domain, term_strategy_engine

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
            w_domain=0.5,
            w_meaning=0.3,
            w_lex=0.2,
        )
        candidate = CandidateScore(
            text=normalized_text,
            total_score=reward.total,
            breakdown={
                "domain": reward.breakdown.domain,
                "meaning": reward.breakdown.meaning,
                "lex": reward.breakdown.lex,
            },
            temperature=0.0,
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
    action = bandit.choose(route_key)
    prompt = build_prompt(
        action=action,
        text=normalized_text,
        src=payload.src,
        tgt=payload.tgt,
        term_instructions=term_instruction_block,
    )

    num_generations = payload.k
    max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "0")))
    retry_sleep = float(os.getenv("GEMINI_RETRY_SLEEP_SEC", "1.5"))
    strategy_penalty_weight = float(os.getenv("STRATEGY_PENALTY_WEIGHT", "0.15"))

    candidate_pool: dict[str, CandidateScore] = {}
    used_fallback = False
    num_attempted = 0

    def _generate_one(temp: float) -> tuple[str, int, bool]:
        local_attempts = 0
        for retry_idx in range(max_retries + 1):
            local_attempts += 1
            try:
                text = _generate_with_provider(prompt=prompt, temperature=temp, top_p=0.95)
                return text, local_attempts, False
            except NotImplementedError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                message = str(exc)
                quota_like = ("429" in message) or ("RESOURCE_EXHAUSTED" in message)
                if quota_like and retry_idx < max_retries and retry_sleep > 0:
                    time.sleep(retry_sleep)
                    continue
                if quota_like:
                    return normalized_text, local_attempts, True
                # Non-quota generation error: degrade gracefully for this slot.
                return normalized_text, local_attempts, True

    temperatures = [_temperature_for_step(i, max(2, num_generations)) for i in range(num_generations)]

    with ThreadPoolExecutor(max_workers=max(1, num_generations)) as executor:
        future_map = {executor.submit(_generate_one, t): t for t in temperatures}
        for future in as_completed(future_map):
            temp = future_map[future]
            candidate, attempts, slot_fallback = future.result()
            num_attempted += attempts
            used_fallback = used_fallback or slot_fallback

            if not candidate.strip():
                candidate = normalized_text
                used_fallback = True

            reward = compute_reward(
                source_text=normalized_text,
                candidate=candidate,
                tgt=payload.tgt,
                clf=clf,
                lexicon_by_domain=lexicon_by_domain,
                w_domain=0.5,
                w_meaning=0.3,
                w_lex=0.2,
            )
            strat_pen = strategy_penalty(candidate, strategies)
            final_score = reward.total - (strategy_penalty_weight * strat_pen)

            scored_candidate = CandidateScore(
                text=candidate,
                total_score=final_score,
                breakdown={
                    "domain": reward.breakdown.domain,
                    "meaning": reward.breakdown.meaning,
                    "lex": reward.breakdown.lex,
                    "strategy_penalty": strat_pen,
                },
                temperature=temp,
            )

            existing = candidate_pool.get(scored_candidate.text)
            if existing is None or scored_candidate.total_score > existing.total_score:
                candidate_pool[scored_candidate.text] = scored_candidate

    # Keep only top-k after parallel generation.
    top_k = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)[: payload.k]
    candidate_pool = {c.text: c for c in top_k}

    scored = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)

    if not scored:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return any candidate. Try again later or reduce GEMINI_TOTAL_RUNS.",
        )

    best = scored[0]
    bandit.update(route_key, action, best.total_score)

    response = TranslateResponse(
        best_candidate=best.text,
        best_score=best.total_score,
        score_breakdown=best.breakdown,
        candidates=scored,
        prompt_action=action,
        used_fallback=used_fallback,
        num_attempted=num_attempted,
        num_returned=len(scored),
        cache_hit=False,
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

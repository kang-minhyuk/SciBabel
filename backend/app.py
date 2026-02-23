from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bandit import EpsilonGreedyBandit
from gemini_client import generate_with_gemini
from prompts import build_prompt, get_prompt_action_names
from reward import compute_reward

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

Domain = Literal["CSM", "PM", "CCE"]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "domain_clf.joblib"
LEXICON_PATH = ROOT / "data" / "processed" / "domain_lexicon.json"


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


class TranslateResponse(BaseModel):
    best_candidate: str
    best_score: float
    score_breakdown: dict[str, float]
    candidates: list[CandidateScore]
    prompt_action: str
    used_fallback: bool
    num_attempted: int
    num_returned: int


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
bandit = EpsilonGreedyBandit(actions=get_prompt_action_names(), epsilon=0.2)


def _load_artifacts() -> None:
    global clf, lexicon_by_domain

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
    global clf, lexicon_by_domain

    if clf is None or not lexicon_by_domain:
        try:
            _load_artifacts()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    route_key = f"{payload.src}->{payload.tgt}"
    action = bandit.choose(route_key)
    prompt = build_prompt(action=action, text=payload.text, src=payload.src, tgt=payload.tgt)

    total_runs_env = int(os.getenv("GEMINI_TOTAL_RUNS", "16"))
    total_runs = max(payload.k, total_runs_env)
    inter_call_sleep = float(os.getenv("GEMINI_INTER_CALL_SLEEP_SEC", "0.6"))

    candidate_pool: dict[str, CandidateScore] = {}
    used_fallback = False
    num_attempted = 0
    for i in range(total_runs):
        temp = _temperature_for_step(i, total_runs)
        num_attempted += 1
        try:
            candidate = generate_with_gemini(prompt=prompt, temperature=temp, top_p=0.95)
        except NotImplementedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            message = str(exc)
            quota_like = ("429" in message) or ("RESOURCE_EXHAUSTED" in message)
            # Free-tier API keys can throttle quickly. If we already have
            # candidates, return best effort instead of failing the whole request.
            if quota_like:
                if not candidate_pool:
                    fallback_text = payload.text
                    fallback_reward = compute_reward(
                        source_text=payload.text,
                        candidate=fallback_text,
                        tgt=payload.tgt,
                        clf=clf,
                        lexicon_by_domain=lexicon_by_domain,
                        w_domain=0.5,
                        w_meaning=0.3,
                        w_lex=0.2,
                    )
                    fallback = CandidateScore(
                        text=fallback_text,
                        total_score=fallback_reward.total,
                        breakdown={
                            "domain": fallback_reward.breakdown.domain,
                            "meaning": fallback_reward.breakdown.meaning,
                            "lex": fallback_reward.breakdown.lex,
                        },
                        temperature=temp,
                    )
                    candidate_pool[fallback.text] = fallback
                    used_fallback = True
                break
            raise HTTPException(status_code=502, detail=f"Gemini call failed: {exc}") from exc

        if not candidate.strip():
            candidate = payload.text
            used_fallback = True

        reward = compute_reward(
            source_text=payload.text,
            candidate=candidate,
            tgt=payload.tgt,
            clf=clf,
            lexicon_by_domain=lexicon_by_domain,
            w_domain=0.5,
            w_meaning=0.3,
            w_lex=0.2,
        )

        scored_candidate = CandidateScore(
            text=candidate,
            total_score=reward.total,
            breakdown={
                "domain": reward.breakdown.domain,
                "meaning": reward.breakdown.meaning,
                "lex": reward.breakdown.lex,
            },
            temperature=temp,
        )

        existing = candidate_pool.get(scored_candidate.text)
        if existing is None or scored_candidate.total_score > existing.total_score:
            candidate_pool[scored_candidate.text] = scored_candidate

        # Keep only top-k candidates while iterating many calls.
        top_k = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)[: payload.k]
        candidate_pool = {c.text: c for c in top_k}

        # Avoid bursty requests on free-tier quotas.
        if i < total_runs - 1 and inter_call_sleep > 0:
            time.sleep(inter_call_sleep)

    scored = sorted(candidate_pool.values(), key=lambda c: c.total_score, reverse=True)

    if not scored:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return any candidate. Try again later or reduce GEMINI_TOTAL_RUNS.",
        )

    best = scored[0]
    bandit.update(route_key, action, best.total_score)

    return TranslateResponse(
        best_candidate=best.text,
        best_score=best.total_score,
        score_breakdown=best.breakdown,
        candidates=scored,
        prompt_action=action,
        used_fallback=used_fallback,
        num_attempted=num_attempted,
        num_returned=len(scored),
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)

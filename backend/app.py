from __future__ import annotations

import json
import os
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


class TranslateRequest(BaseModel):
    text: str = Field(min_length=3)
    src: Domain
    tgt: Domain
    k: int = Field(default=2, ge=2, le=8)


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


app = FastAPI(title="SciBabel API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf = None
lexicon_by_domain: dict[str, list[str]] = {}
bandit = EpsilonGreedyBandit(actions=get_prompt_action_names(), epsilon=0.2)


def _load_artifacts() -> None:
    global clf, lexicon_by_domain

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

    temperatures = [0.2 + (i * 0.2) for i in range(payload.k)]
    inter_call_sleep = float(os.getenv("GEMINI_INTER_CALL_SLEEP_SEC", "0.6"))

    scored: list[CandidateScore] = []
    for i, temp in enumerate(temperatures):
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
                if not scored:
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
                    scored.append(
                        CandidateScore(
                            text=fallback_text,
                            total_score=fallback_reward.total,
                            breakdown={
                                "domain": fallback_reward.breakdown.domain,
                                "meaning": fallback_reward.breakdown.meaning,
                                "lex": fallback_reward.breakdown.lex,
                            },
                            temperature=temp,
                        )
                    )
                break
            raise HTTPException(status_code=502, detail=f"Gemini call failed: {exc}") from exc

        if not candidate.strip():
            candidate = payload.text

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

        scored.append(
            CandidateScore(
                text=candidate,
                total_score=reward.total,
                breakdown={
                    "domain": reward.breakdown.domain,
                    "meaning": reward.breakdown.meaning,
                    "lex": reward.breakdown.lex,
                },
                temperature=temp,
            )
        )

        # Avoid bursty requests on free-tier quotas.
        if i < len(temperatures) - 1 and inter_call_sleep > 0:
            time.sleep(inter_call_sleep)

    if not scored:
        raise HTTPException(
            status_code=502,
            detail="Gemini did not return any candidate. Try again with k=2 or wait for quota reset.",
        )

    scored.sort(key=lambda c: c.total_score, reverse=True)
    best = scored[0]
    bandit.update(route_key, action, best.total_score)

    return TranslateResponse(
        best_candidate=best.text,
        best_score=best.total_score,
        score_breakdown=best.breakdown,
        candidates=scored,
        prompt_action=action,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=True)

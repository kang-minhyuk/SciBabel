# Backend (FastAPI)

## Endpoints

- `GET /health`
- `POST /translate`

## `/translate` request shape

```json
{
  "text": "Graph neural networks...",
  "src": "CSM",
  "tgt": "PM",
  "k": 4
}
```

## Runtime dependencies

The backend expects:

- `models/domain_clf.joblib`
- `data/processed/domain_lexicon.json`

Generate these via `make train-sample` or scripts in `../scripts/`.

## Gemini config

Set `GEMINI_API_KEY` in `.env` at repo root.

If not set, translation call returns a descriptive runtime error.

## Deployment notes

### Required env vars

- `GEMINI_API_KEY`
- `BACKEND_CORS_ORIGINS` (comma-separated; example: `https://your-frontend.vercel.app`)

### Optional env vars

- `GEMINI_TOTAL_RUNS` (default `16`)
- `GEMINI_INTER_CALL_SLEEP_SEC` (default `0.6`)
- `GEMINI_MODEL` (optional model pin)

### Render/Fly/Railway start command

- `uvicorn app:app --host 0.0.0.0 --port $PORT`

If model artifacts are missing in deployment, run sample training in build:

- `python scripts/03_mine_terms.py --corpus data/processed/sample_corpus.jsonl --lexicon-out data/processed/domain_lexicon.json --term-stats-out data/processed/term_stats.csv --top-n 120`
- `python scripts/04_train_domain_classifier.py --corpus data/processed/sample_corpus.jsonl --model-out models/domain_clf.joblib`

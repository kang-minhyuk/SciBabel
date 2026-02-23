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

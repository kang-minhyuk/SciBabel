# SciBabel (working title)

SciBabel is an MVP cross-domain scientific language translator with:

- FastAPI backend (`/health`, `/translate`)
- Offline corpus + term mining + classifier training scripts
- Next.js + Tailwind frontend for interactive translation

## Repository layout

- `backend/` API service and reranking logic
- `scripts/` data pipeline scripts
- `data/` sample and generated data artifacts
- `models/` trained model artifacts
- `frontend/` Next.js app
- `docs/` design docs

## Quickstart (macOS)

1. Copy env template:
   - `cp .env.example .env`
2. Install dependencies:
   - `make setup`
3. Build sample artifacts:
   - `make fetch-sample`
   - `make train-sample`
4. Start both services:
   - `make dev`

Frontend: http://localhost:3000  
Backend: http://localhost:8000  
Health check: http://localhost:8000/health

## Run backend/frontend separately

- Backend: `make dev-backend`
- Frontend: `make dev-frontend` or `npm run dev` (from repo root)

## Run scripts manually

Fetch arXiv abstracts:

- `python3 scripts/01_fetch_arxiv.py --query "cat:cs.LG" --max-results 50 --domain CSM --out data/raw/csm_arxiv.jsonl`

Merge domain files into corpus:

- `python3 scripts/02_build_corpus.py --inputs data/raw/csm_arxiv.jsonl=CSM data/raw/pm_arxiv.jsonl=PM data/raw/cce_arxiv.jsonl=CCE --out data/processed/corpus.parquet`

Mine terms:

- `python3 scripts/03_mine_terms.py --corpus data/processed/corpus.parquet --lexicon-out data/processed/domain_lexicon.json --term-stats-out data/processed/term_stats.csv`

Train domain classifier:

- `python3 scripts/04_train_domain_classifier.py --corpus data/processed/corpus.parquet --model-out models/domain_clf.joblib`

## Notes

- Set `GEMINI_API_KEY` in `.env` to enable live generation.
- If Gemini is not configured, `/translate` returns a clear error.
- Bandit prompt selection is in-memory for MVP.

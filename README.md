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

- LLM provider is configurable via `LLM_PROVIDER` (`gpt` default, `gemini` optional).
- Set `GEMINI_API_KEY` in `.env` to enable live generation.
- If Gemini is not configured, `/translate` returns a clear error.
- Bandit prompt selection is in-memory for MVP.
- GitHub Actions CI runs backend checks + sample artifact build + frontend build on push/PR to `main`.

## Deploy (website-first)

### Frontend (Vercel)

- Import this GitHub repo into Vercel.
- Set project root to `frontend/`.
- Add env var: `NEXT_PUBLIC_API_BASE_URL=https://scibabel.onrender.com`.

### Backend (Render)

- Create a new Web Service from this repo.
- Root directory: leave empty (repo root)
- Build command: `pip install -r backend/requirements.txt`
- Start command: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

Backend env vars:

- `GEMINI_API_KEY=<your-key>`
- `BACKEND_CORS_ORIGINS=https://sci-babel.vercel.app`
- Optional: `GEMINI_MAX_RETRIES=0`, `GEMINI_RETRY_SLEEP_SEC=1.5`

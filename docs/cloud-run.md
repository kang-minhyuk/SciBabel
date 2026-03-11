# Deploy SciBabel backend to Google Cloud Run

## 1) Prerequisites

- Install Google Cloud CLI (`gcloud`)
- Authenticate:
  - `gcloud auth login`
  - `gcloud auth application-default login`
- Set project:
  - `gcloud config set project <PROJECT_ID>`
- Enable APIs:
  - `run.googleapis.com`
  - `cloudbuild.googleapis.com`
  - `artifactregistry.googleapis.com`

## 2) Deploy

From repo root:

- `PROJECT_ID=<your-project> REGION=us-central1 SERVICE_NAME=scibabel-backend ./scripts/deploy_cloud_run.sh`

Or via Make:

- `PROJECT_ID=<your-project> REGION=us-central1 SERVICE_NAME=scibabel-backend make cloudrun-deploy`

## 3) Configure secrets/env vars

After first deploy, set required runtime variables if needed:

- `GEMINI_API_KEY` (if used)
- `BACKEND_CORS_ORIGINS` (frontend URL)

Example:

- `gcloud run services update scibabel-backend --region us-central1 --set-env-vars BACKEND_CORS_ORIGINS=https://your-frontend-domain`
- `gcloud run services update scibabel-backend --region us-central1 --set-env-vars GEMINI_API_KEY=<your-key>`

## 4) Validate

- `curl -sS https://<cloud-run-url>/health`
- `curl -sS https://<cloud-run-url>/ready`

## Notes

- Image is built from `Dockerfile` using `backend/requirements-prod.txt`.
- Cloud Build upload filtering is controlled by `.gcloudignore` and includes required runtime artifacts:
  - `data/processed/domain_lexicon.json`
  - `data/processed/term_stats.csv`
  - `models/domain_clf.joblib`
- Heavy libraries (`torch`, `transformers`, `sentence-transformers`) are not included in production dependencies.
- Current stability defaults are baked in for production-safe annotate behavior:
  - `ANNOTATE_MAX_CONCURRENCY=1`
  - `ANNOTATE_TIMEOUT_SEC=20`
  - `EVIDENCE_ENABLED=false`
  - `YAKE_ENABLED=false`

## 5) Recommended production knobs

- Keep one warm instance to reduce cold starts:
  - `gcloud run services update scibabel-backend --region us-central1 --min-instances 1`

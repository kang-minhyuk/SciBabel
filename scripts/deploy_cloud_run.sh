#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID=${PROJECT_ID:-}
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-scibabel-backend}
IMAGE_NAME=${IMAGE_NAME:-scibabel-backend}

if [[ -z "$PROJECT_ID" ]]; then
  echo "PROJECT_ID is required"
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${IMAGE_NAME}:$(date +%Y%m%d-%H%M%S)"

echo "[deploy] building image: ${IMAGE_URI}"
gcloud builds submit --config cloudbuild.yaml --substitutions _IMAGE_URI="${IMAGE_URI}" .

echo "[deploy] deploying service: ${SERVICE_NAME}"
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 4 \
  --timeout 30 \
  --set-env-vars "SCIBABEL_ENV=production,EVIDENCE_ENABLED=false,YAKE_ENABLED=false,ANALOG_USE_EMBEDDINGS=false,ANALOG_MAX_CANDIDATES=300,ANALOG_MAX_TERMS=6,SPACY_LOAD_MODEL_IN_PROD=false,PRODUCTION_MAX_TERMS=6,ANNOTATE_MAX_CONCURRENCY=1,ANNOTATE_TIMEOUT_SEC=20,ANNOTATE_ACQUIRE_TIMEOUT_SEC=0.1"

echo "[deploy] done"

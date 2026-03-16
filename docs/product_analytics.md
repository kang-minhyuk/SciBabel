# Product Analytics

This project supports lightweight, privacy-safe product analytics for the key API endpoints.

## Scope

The backend logs structured JSONL events for:

- `POST /annotate`
- `POST /profile_annotate`
- `POST /explain`

Raw user text is not logged. Instead, analytics capture metadata such as text length, request mode, and aggregate output counts.

## Configuration

Environment variables:

- `PRODUCT_ANALYTICS_ENABLED` (default: `true`)
- `PRODUCT_ANALYTICS_LOG_PATH` (default: `logs/product_analytics.jsonl` under repo root)

Examples:

```bash
export PRODUCT_ANALYTICS_ENABLED=true
export PRODUCT_ANALYTICS_LOG_PATH=/tmp/scibabel_product_analytics.jsonl
```

## Event Schema

Each line in the JSONL log contains:

- `ts`: unix timestamp (seconds)
- `env`: runtime env (`dev` or `production`)
- `event_type`: `annotate`, `profile_annotate`, or `explain`
- `request_id`: short request identifier
- `latency_ms`: end-to-end request latency
- `status_code`: HTTP status
- `error_reason`: normalized reason (`none` on success)
- `payload`: request metadata (domains, options, length stats)
- `result`: aggregate output metadata (term counts, ambiguity flags, explanation presence)

## Summarization

Use the evaluator script to generate operator-facing reports:

```bash
python3 scripts/eval/summarize_product_analytics.py \
  --input logs/product_analytics.jsonl \
  --out-md reports/product_analytics/product_analytics_summary.md \
  --out-json reports/product_analytics/product_analytics_summary.json
```

The summary includes:

- event counts by endpoint
- status/error distributions
- latency aggregates (`avg`, `p50`, `p95`, `p99`)
- quality signals (avg terms, ambiguity rate, explain non-empty rate)

## Operational Notes

- Logging failures are intentionally non-fatal and never fail API requests.
- Keep log retention bounded in production (for example via log shipping or periodic rotation).
- Review analytics for regressions after each deployment and before baseline updates.

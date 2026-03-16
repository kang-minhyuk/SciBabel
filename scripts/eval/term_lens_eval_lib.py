from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BAD_FRAGMENTS = {
    "optimize a graph",
    "memory cost on",
    "on long sequences",
    "by an",
    "of the",
    "the thin",
    "we derive the",
}

GENERIC_ANALOGS = {
    "at low temperature",
    "pattern of the",
    "of the",
    "by an",
    "the model",
}

ENV_API_KEYS = ["SCIBABEL_API_BASE_URL", "NEXT_PUBLIC_API_BASE_URL", "CLOUD_RUN_BASE_URL"]


def _is_local_url(url: str) -> bool:
    s = url.strip().lower()
    return (
        "localhost" in s
        or "127.0.0.1" in s
        or s.startswith("http://0.0.0.0")
    )


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _from_env_file(repo_root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for env_path in [repo_root / ".env", repo_root / ".env.local", repo_root / "frontend" / ".env.local"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _from_frontend_config(repo_root: Path) -> str | None:
    page = repo_root / "frontend" / "app" / "page.tsx"
    if not page.exists():
        return None
    text = page.read_text(encoding="utf-8")
    marker = "const API_BASE = "
    idx = text.find(marker)
    if idx < 0:
        return None
    tail = text[idx + len(marker):]
    quote = '"' if '"' in tail[:5] else "'"
    q1 = tail.find(quote)
    if q1 < 0:
        return None
    q2 = tail.find(quote, q1 + 1)
    if q2 < 0:
        return None
    val = tail[q1 + 1:q2].strip()
    if val.startswith("http"):
        return val
    return None


def resolve_live_api_base(repo_root: Path) -> tuple[str | None, str]:
    for key in ENV_API_KEYS:
        val = os.getenv(key, "").strip()
        if val and not _is_local_url(val):
            return val.rstrip("/"), f"env:{key}"

    env_vals = _from_env_file(repo_root)
    for key in ENV_API_KEYS:
        val = env_vals.get(key, "").strip()
        if val and not _is_local_url(val):
            return val.rstrip("/"), f"env-file:{key}"

    fe = _from_frontend_config(repo_root)
    if fe and not _is_local_url(fe):
        return fe.rstrip("/"), "frontend:app/page.tsx"

    return None, "missing"


def load_cases(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "id": str(row["id"]),
                "src": str(row["src"]),
                "tgt": str(row["tgt"]),
                "text": str(row["text"]),
            }
        )
    return rows


def _safe_json(resp: requests.Response) -> dict[str, Any]:
    try:
        val = resp.json()
        if isinstance(val, dict):
            return val
        return {"value": val}
    except Exception:
        return {"raw": resp.text[:4000]}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((q / 100.0) * (len(s) - 1)))
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def run_regression(api_base: str, cases: list[dict[str, str]], timeout: float = 25.0) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    bad_phrase_count = 0
    bad_analog_count = 0
    same_domain_over_flagging = 0
    ambiguous_high_conf = 0
    error_count = 0

    for case in cases:
        for same_field_mode in ["normal", "study"]:
            payload = {
                "text": case["text"],
                "src": case["src"],
                "tgt": case["tgt"],
                "same_field_mode": same_field_mode,
                "max_terms": 8,
            }

            t0 = time.perf_counter()
            try:
                resp = requests.post(f"{api_base}/annotate", json=payload, timeout=timeout)
                latency_sec = float(time.perf_counter() - t0)
                body = _safe_json(resp)
                status = int(resp.status_code)
            except Exception as exc:
                latency_sec = float(time.perf_counter() - t0)
                body = {"error": str(exc)}
                status = 0
                error_count += 1

            terms = body.get("terms", []) if isinstance(body, dict) else []
            if not isinstance(terms, list):
                terms = []
            flagged = [t for t in terms if isinstance(t, dict) and bool(t.get("flagged"))]
            analog_total = 0
            evidence_total = 0
            for t in terms:
                if isinstance(t, dict):
                    analog_total += len(t.get("analogs", []) if isinstance(t.get("analogs", []), list) else [])
                    evidence_total += len(t.get("evidence", []) if isinstance(t.get("evidence", []), list) else [])

            bad_phrases = [
                str(t.get("term", "")).lower()
                for t in terms
                if isinstance(t, dict) and str(t.get("term", "")).lower() in BAD_FRAGMENTS
            ]
            bad_phrase_count += len(bad_phrases)

            bad_analogs: list[str] = []
            for t in terms:
                if not isinstance(t, dict):
                    continue
                analogs = t.get("analogs", []) if isinstance(t.get("analogs", []), list) else []
                for a in analogs:
                    if not isinstance(a, dict):
                        continue
                    cand = str(a.get("candidate", "")).lower()
                    if cand in GENERIC_ANALOGS:
                        bad_analogs.append(cand)
            bad_analog_count += len(bad_analogs)

            conf = float(body.get("predicted_src_confidence", 0.0)) if isinstance(body, dict) and body.get("predicted_src_confidence") is not None else 0.0
            gap = float(body.get("top2_gap", 0.0)) if isinstance(body, dict) and body.get("top2_gap") is not None else 0.0
            amb = bool(body.get("is_ambiguous", False)) if isinstance(body, dict) else False
            if conf >= 0.75 and gap >= 0.3 and amb:
                ambiguous_high_conf += 1

            rec = {
                "id": case["id"],
                "src": case["src"],
                "tgt": case["tgt"],
                "same_field_mode": same_field_mode,
                "status": status,
                "latency_sec": round(latency_sec, 4),
                "flagged_count": len(flagged),
                "analog_count": analog_total,
                "evidence_count": evidence_total,
                "bad_phrases": bad_phrases,
                "bad_analogs": bad_analogs,
                "is_ambiguous": amb,
                "predicted_src_confidence": conf,
                "top2_gap": gap,
                "error": body.get("error", "") if isinstance(body, dict) else "",
            }
            records.append(rec)

            if status != 200:
                error_count += 1

        same_payload = {
            "text": case["text"],
            "src": case["src"],
            "tgt": case["src"],
            "same_field_mode": "normal",
            "max_terms": 8,
        }
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{api_base}/annotate", json=same_payload, timeout=timeout)
            latency_sec = float(time.perf_counter() - t0)
            body = _safe_json(resp)
            status = int(resp.status_code)
        except Exception as exc:
            latency_sec = float(time.perf_counter() - t0)
            body = {"error": str(exc)}
            status = 0
            error_count += 1

        terms = body.get("terms", []) if isinstance(body, dict) else []
        if not isinstance(terms, list):
            terms = []
        flagged = [t for t in terms if isinstance(t, dict) and bool(t.get("flagged"))]
        analog_total = sum(len(t.get("analogs", []) if isinstance(t.get("analogs", []), list) else []) for t in terms if isinstance(t, dict))
        evidence_total = sum(len(t.get("evidence", []) if isinstance(t.get("evidence", []), list) else []) for t in terms if isinstance(t, dict))
        if len(flagged) > 2:
            same_domain_over_flagging += 1

        records.append(
            {
                "id": f"{case['id']}_same",
                "src": case["src"],
                "tgt": case["src"],
                "same_field_mode": "normal",
                "status": status,
                "latency_sec": round(latency_sec, 4),
                "flagged_count": len(flagged),
                "analog_count": analog_total,
                "evidence_count": evidence_total,
                "bad_phrases": [],
                "bad_analogs": [],
                "is_ambiguous": bool(body.get("is_ambiguous", False)) if isinstance(body, dict) else False,
                "predicted_src_confidence": float(body.get("predicted_src_confidence", 0.0)) if isinstance(body, dict) and body.get("predicted_src_confidence") is not None else 0.0,
                "top2_gap": float(body.get("top2_gap", 0.0)) if isinstance(body, dict) and body.get("top2_gap") is not None else 0.0,
                "error": body.get("error", "") if isinstance(body, dict) else "",
            }
        )
        if status != 200:
            error_count += 1

    latencies = [float(r["latency_sec"]) for r in records]
    statuses = [int(r["status"]) for r in records]

    return {
        "endpoint": api_base,
        "total_cases": len(cases),
        "total_requests": len(records),
        "success_count": sum(1 for s in statuses if s == 200),
        "failure_count": sum(1 for s in statuses if s != 200),
        "error_count": error_count,
        "p50_latency_sec": round(_percentile(latencies, 50), 4),
        "p95_latency_sec": round(_percentile(latencies, 95), 4),
        "avg_latency_sec": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "bad_phrase_fragments": bad_phrase_count,
        "bad_analog_suggestions": bad_analog_count,
        "same_domain_over_flagging": same_domain_over_flagging,
        "ambiguous_high_confidence": ambiguous_high_conf,
        "records": records,
    }


def write_regression_reports(result: dict[str, Any], out_dir: Path, label: str, ts: str | None = None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = ts or now_ts()
    json_path = out_dir / f"term_lens_regression_{label}_{stamp}.json"
    md_path = out_dir / f"term_lens_regression_{label}_{stamp}.md"

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# Term Lens Regression Report ({label})",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Endpoint: {result['endpoint']}",
        f"- Total cases: {result['total_cases']}",
        f"- Total requests: {result['total_requests']}",
        "",
        "## Summary",
        "",
        f"- success_count: {result['success_count']}",
        f"- failure_count: {result['failure_count']}",
        f"- error_count: {result['error_count']}",
        f"- avg_latency_sec: {result['avg_latency_sec']}",
        f"- p50_latency_sec: {result['p50_latency_sec']}",
        f"- p95_latency_sec: {result['p95_latency_sec']}",
        f"- bad phrase fragments: {result['bad_phrase_fragments']}",
        f"- bad analog suggestions: {result['bad_analog_suggestions']}",
        f"- same-domain over-flagging events: {result['same_domain_over_flagging']}",
        f"- ambiguous while high-confidence events: {result['ambiguous_high_confidence']}",
        "",
        "## Case Table",
        "",
        "| Case | src->tgt | mode | status | latency(s) | flagged | analogs | evidence | ambiguous | conf | gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]

    for r in result["records"]:
        lines.append(
            f"| {r['id']} | {r['src']}->{r['tgt']} | {r['same_field_mode']} | {r['status']} | {r['latency_sec']:.4f} | {r['flagged_count']} | {r['analog_count']} | {r['evidence_count']} | {r['is_ambiguous']} | {r['predicted_src_confidence']:.3f} | {r['top2_gap']:.3f} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, json_path

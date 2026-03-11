from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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


def _load_cases(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append({
            "id": str(row["id"]),
            "src": str(row["src"]),
            "tgt": str(row["tgt"]),
            "text": str(row["text"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run term-lens regression cases against /annotate.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default="scripts/eval/regression_cases_term_lens.jsonl")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    base = args.api_base.rstrip("/")
    cases = _load_cases(Path(args.cases))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    records: list[dict[str, object]] = []
    bad_phrase_count = 0
    bad_analog_count = 0
    same_domain_over_flagging = 0
    ambiguous_high_conf = 0

    for case in cases:
        for same_field_mode in ["normal", "study"]:
            payload = {
                "text": case["text"],
                "src": case["src"],
                "tgt": case["tgt"],
                "same_field_mode": same_field_mode,
                "max_terms": 8,
            }
            r = requests.post(f"{base}/annotate", json=payload, timeout=args.timeout)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}

            terms = body.get("terms", []) if isinstance(body, dict) else []
            flagged = [t for t in terms if isinstance(t, dict) and bool(t.get("flagged"))]

            bad_phrases = [
                str(t.get("term", "")).lower()
                for t in terms
                if isinstance(t, dict) and str(t.get("term", "")).lower() in BAD_FRAGMENTS
            ]
            bad_phrase_count += len(bad_phrases)

            bad_analogs = []
            for t in terms:
                if not isinstance(t, dict):
                    continue
                for a in t.get("analogs", []):
                    cand = str(a.get("candidate", "")).lower()
                    if cand in GENERIC_ANALOGS:
                        bad_analogs.append(cand)
            bad_analog_count += len(bad_analogs)

            same_domain = case["src"] == case["tgt"]
            if same_domain and same_field_mode == "normal" and len(flagged) > 2:
                same_domain_over_flagging += 1

            conf = float(body.get("predicted_src_confidence", 0.0)) if isinstance(body, dict) else 0.0
            gap = float(body.get("top2_gap", 0.0)) if isinstance(body, dict) else 0.0
            amb = bool(body.get("is_ambiguous", False)) if isinstance(body, dict) else False
            if conf >= 0.75 and gap >= 0.3 and amb:
                ambiguous_high_conf += 1

            records.append(
                {
                    "id": case["id"],
                    "src": case["src"],
                    "tgt": case["tgt"],
                    "same_field_mode": same_field_mode,
                    "status": r.status_code,
                    "flagged_count": len(flagged),
                    "bad_phrases": bad_phrases,
                    "bad_analogs": bad_analogs,
                    "is_ambiguous": amb,
                    "predicted_src_confidence": conf,
                    "top2_gap": gap,
                }
            )

        # Explicit same-domain pass for clutter measurement.
        payload_same = {
            "text": case["text"],
            "src": case["src"],
            "tgt": case["src"],
            "same_field_mode": "normal",
            "max_terms": 8,
        }
        r_same = requests.post(f"{base}/annotate", json=payload_same, timeout=args.timeout)
        body_same = r_same.json() if r_same.headers.get("content-type", "").startswith("application/json") else {"raw": r_same.text}
        terms_same = body_same.get("terms", []) if isinstance(body_same, dict) else []
        flagged_same = [t for t in terms_same if isinstance(t, dict) and bool(t.get("flagged"))]
        if len(flagged_same) > 2:
            same_domain_over_flagging += 1

        records.append(
            {
                "id": f"{case['id']}_same",
                "src": case["src"],
                "tgt": case["src"],
                "same_field_mode": "normal",
                "status": r_same.status_code,
                "flagged_count": len(flagged_same),
                "bad_phrases": [],
                "bad_analogs": [],
                "is_ambiguous": bool(body_same.get("is_ambiguous", False)) if isinstance(body_same, dict) else False,
                "predicted_src_confidence": float(body_same.get("predicted_src_confidence", 0.0)) if isinstance(body_same, dict) else 0.0,
                "top2_gap": float(body_same.get("top2_gap", 0.0)) if isinstance(body_same, dict) else 0.0,
            }
        )

    json_path = out_dir / f"term_lens_regression_{ts}.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / f"term_lens_regression_{ts}.md"
    lines = [
        "# Term Lens Regression Report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- API Base: {base}",
        f"- Cases: {len(cases)}",
        "",
        "## Aggregates",
        "",
        f"- bad phrase fragments: {bad_phrase_count}",
        f"- bad analog suggestions: {bad_analog_count}",
        f"- same-domain over-flagging events: {same_domain_over_flagging}",
        f"- ambiguous while high-confidence events: {ambiguous_high_conf}",
        "",
        "## Case Table",
        "",
        "| Case | src->tgt | mode | status | flagged | ambiguous | conf | gap | bad_phrases | bad_analogs |",
        "|---|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['id']} | {r['src']}->{r['tgt']} | {r['same_field_mode']} | {r['status']} | {r['flagged_count']} | {r['is_ambiguous']} | {r['predicted_src_confidence']:.3f} | {r['top2_gap']:.3f} | {', '.join(r['bad_phrases'])} | {', '.join(r['bad_analogs'])} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report_md": str(md_path.resolve()), "report_json": str(json_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

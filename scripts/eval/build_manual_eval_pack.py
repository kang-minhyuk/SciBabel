from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import requests

from term_lens_eval_lib import resolve_live_api_base


def _examples() -> list[dict[str, str]]:
    # 24 balanced examples: 6 each domain with mixed same/cross/easy/ambiguous.
    return [
        {"id": "csm_01", "src": "CSM", "tgt": "PM", "difficulty": "easy", "text": "We optimize a graph neural network with sparse regularization under distribution shift."},
        {"id": "csm_02", "src": "CSM", "tgt": "CHEM", "difficulty": "cross", "text": "The transformer uses low-rank attention to reduce memory cost on long sequences."},
        {"id": "csm_03", "src": "CSM", "tgt": "CHEME", "difficulty": "cross", "text": "A reinforcement learning policy improves sample efficiency using prioritized replay."},
        {"id": "csm_04", "src": "CSM", "tgt": "CSM", "difficulty": "same", "text": "Bayesian optimization tunes hyperparameters for noisy objective landscapes."},
        {"id": "csm_05", "src": "CSM", "tgt": "PM", "difficulty": "ambiguous", "text": "Uncertainty calibration and diffusion priors stabilize inverse modeling."},
        {"id": "csm_06", "src": "CSM", "tgt": "CSM", "difficulty": "same", "text": "Contrastive pretraining improves retrieval under class imbalance."},
        {"id": "pm_01", "src": "PM", "tgt": "CSM", "difficulty": "easy", "text": "The phase transition is characterized by an order parameter near criticality."},
        {"id": "pm_02", "src": "PM", "tgt": "CHEM", "difficulty": "cross", "text": "A Monte Carlo simulation estimates magnetization under an external field."},
        {"id": "pm_03", "src": "PM", "tgt": "CHEME", "difficulty": "cross", "text": "Lattice defects alter phonon scattering and thermal conductivity."},
        {"id": "pm_04", "src": "PM", "tgt": "PM", "difficulty": "same", "text": "Wave propagation in anisotropic media follows modified dispersion relations."},
        {"id": "pm_05", "src": "PM", "tgt": "CSM", "difficulty": "ambiguous", "text": "Stochastic fluctuations perturb coupled dynamics in nonlinear systems."},
        {"id": "pm_06", "src": "PM", "tgt": "PM", "difficulty": "same", "text": "Ramsey interferometry measures spin coherence at finite temperature."},
        {"id": "chem_01", "src": "CHEM", "tgt": "CHEME", "difficulty": "easy", "text": "NMR spectroscopy confirms the aromatic substitution pattern of the product."},
        {"id": "chem_02", "src": "CHEM", "tgt": "CSM", "difficulty": "cross", "text": "The catalytic cycle proceeds through oxidative addition and reductive elimination."},
        {"id": "chem_03", "src": "CHEM", "tgt": "PM", "difficulty": "cross", "text": "Ligand field effects tune the redox potential of the metal complex."},
        {"id": "chem_04", "src": "CHEM", "tgt": "CHEM", "difficulty": "same", "text": "Chromatography isolates intermediates before final purification."},
        {"id": "chem_05", "src": "CHEM", "tgt": "CHEME", "difficulty": "ambiguous", "text": "Reaction selectivity depends on solvent polarity and proton transfer kinetics."},
        {"id": "chem_06", "src": "CHEM", "tgt": "CHEM", "difficulty": "same", "text": "Molecular orbital analysis explains frontier electron interactions."},
        {"id": "cheme_01", "src": "CHEME", "tgt": "CHEM", "difficulty": "easy", "text": "Process control uses model predictive control to stabilize reactor temperature."},
        {"id": "cheme_02", "src": "CHEME", "tgt": "PM", "difficulty": "cross", "text": "Adsorption isotherms guide design of cyclic gas separation processes."},
        {"id": "cheme_03", "src": "CHEME", "tgt": "CSM", "difficulty": "cross", "text": "A packed-bed reactor improves conversion under controlled residence time."},
        {"id": "cheme_04", "src": "CHEME", "tgt": "CHEME", "difficulty": "same", "text": "Distillation reflux optimization reduces utility consumption."},
        {"id": "cheme_05", "src": "CHEME", "tgt": "CHEM", "difficulty": "ambiguous", "text": "Transport and diffusion constraints shape membrane process performance."},
        {"id": "cheme_06", "src": "CHEME", "tgt": "CHEME", "difficulty": "same", "text": "Heat exchanger network synthesis supports integrated thermal management."},
    ]


def _resolve_api(repo_root: Path, requested: str) -> tuple[str, str]:
    if requested:
        return requested.rstrip("/"), "arg"
    live, source = resolve_live_api_base(repo_root)
    if live:
        return live.rstrip("/"), source
    return "http://127.0.0.1:8000", "default-local"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build polished manual eval package for term-lens.")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--out-dir", default="reports/manual_eval")
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    api_base, source = _resolve_api(repo_root, args.api_base)
    rows = _examples()

    enriched: list[dict[str, object]] = []
    for ex in rows:
        payload = {
            "text": ex["text"],
            "src": ex["src"],
            "tgt": ex["tgt"],
            "same_field_mode": "normal",
            "max_terms": 8,
        }
        try:
            resp = requests.post(f"{api_base}/annotate", json=payload, timeout=args.timeout)
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
        except Exception as exc:
            body = {"error": str(exc)}
            resp = type("R", (), {"status_code": 0})()  # tiny stand-in

        terms = body.get("terms", []) if isinstance(body, dict) else []
        if not isinstance(terms, list):
            terms = []
        flagged = [t for t in terms if isinstance(t, dict) and bool(t.get("flagged"))]

        enriched.append(
            {
                **ex,
                "status": int(getattr(resp, "status_code", 0)),
                "predicted_src": body.get("predicted_src", "") if isinstance(body, dict) else "",
                "predicted_src_confidence": body.get("predicted_src_confidence", 0.0) if isinstance(body, dict) else 0.0,
                "is_ambiguous": body.get("is_ambiguous", False) if isinstance(body, dict) else False,
                "flagged_terms": [str(t.get("term", "")) for t in flagged],
                "analogs": [str(a.get("candidate", "")) for t in flagged if isinstance(t, dict) for a in (t.get("analogs", []) if isinstance(t.get("analogs", []), list) else []) if isinstance(a, dict)],
                "evidence_present": any(bool(t.get("evidence")) for t in flagged if isinstance(t, dict)),
                "raw": body,
            }
        )

    json_path = out_dir / f"manual_eval_examples_{ts}.json"
    md_path = out_dir / f"manual_eval_pack_{ts}.md"
    csv_path = out_dir / f"manual_eval_sheet_{ts}.csv"
    json_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "src",
            "tgt",
            "difficulty",
            "source detection reasonable? (yes/no)",
            "flagged terms useful? (1-5)",
            "analogs useful? (1-5)",
            "explanation likely helpful? (1-5)",
            "evidence helpful? (1-5)",
            "overall usefulness (1-5)",
            "comments",
        ])
        for ex in enriched:
            writer.writerow([ex["id"], ex["src"], ex["tgt"], ex["difficulty"], "", "", "", "", "", "", ""])

    lines = [
        "# Manual Evaluation Pack (Term Lens)",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Endpoint: {api_base}",
        f"- Endpoint source: {source}",
        f"- Total examples: {len(enriched)}",
        "",
        "## How to Review",
        "",
        "1. Read the input text and intended src->tgt mapping.",
        "2. Check whether source detection is reasonable.",
        "3. Judge whether flagged terms are truly unfamiliar in target context.",
        "4. Inspect analog suggestions and evidence presence.",
        "5. Fill the manual CSV sheet.",
        "",
        "## Rating Rubric",
        "",
        "- source detection reasonable? (yes/no)",
        "- flagged terms useful? (1-5)",
        "- analogs useful? (1-5)",
        "- explanation likely helpful? (1-5)",
        "- evidence helpful? (1-5)",
        "- overall usefulness (1-5)",
        "- comments",
        "",
        "## Example Cards",
        "",
    ]

    random.seed(7)
    for ex in enriched:
        terms_preview = ", ".join(ex["flagged_terms"][:5]) if ex["flagged_terms"] else "(none)"
        analog_preview = ", ".join(ex["analogs"][:5]) if ex["analogs"] else "(none)"
        lines.extend(
            [
                f"### {ex['id']} ({ex['src']} -> {ex['tgt']}, {ex['difficulty']})",
                "",
                f"- Input: {ex['text']}",
                f"- Predicted source: {ex['predicted_src']} (conf={float(ex['predicted_src_confidence'] or 0.0):.3f})",
                f"- Ambiguous: {ex['is_ambiguous']}",
                f"- Flagged terms: {terms_preview}",
                f"- Analogs: {analog_preview}",
                f"- Evidence present: {ex['evidence_present']}",
                "",
                "Manual scores:",
                "- source detection reasonable?",
                "- flagged terms useful?",
                "- analogs useful?",
                "- explanation likely helpful?",
                "- evidence helpful?",
                "- overall usefulness?",
                "- comments:",
                "",
            ]
        )

    lines.extend(
        [
            "## Summary Sheet",
            "",
            f"- CSV template: `{csv_path}`",
            f"- JSON examples: `{json_path}`",
        ]
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"manual_eval_md": str(md_path.resolve()), "manual_eval_csv": str(csv_path.resolve()), "manual_eval_json": str(json_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

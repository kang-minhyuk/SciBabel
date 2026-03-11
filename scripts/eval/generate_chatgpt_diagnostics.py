from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://scibabel-backend-523773192713.us-central1.run.app/annotate"
DOMAINS = ["CSM", "PM", "CHEM", "CHEME"]
SAMPLES = {
    "CSM": [
        "We optimize a graph neural network with sparse regularization under distribution shift.",
        "The transformer uses low-rank attention to reduce memory cost on long sequences.",
        "We train a diffusion model with classifier-free guidance for molecular generation.",
        "A reinforcement learning policy improves sample efficiency using prioritized replay.",
        "The encoder-decoder architecture applies contrastive learning for robust representations.",
        "We benchmark domain adaptation with pseudo-labeling and entropy minimization.",
        "Our Bayesian optimization routine tunes hyperparameters for noisy objectives.",
        "The model calibrates uncertainty with temperature scaling after fine-tuning.",
        "We compress the network via structured pruning and knowledge distillation.",
        "A multi-agent planner coordinates exploration with hierarchical value functions.",
    ],
    "PM": [
        "The Hamiltonian formalism describes coupled oscillators with weak damping.",
        "We analyze wave propagation in anisotropic media using dispersion relations.",
        "The phase transition is characterized by an order parameter near criticality.",
        "A Monte Carlo simulation estimates magnetization under an external field.",
        "Quantum tunneling dominates transport through the thin potential barrier.",
        "The system follows non-equilibrium dynamics governed by stochastic fluctuations.",
        "We derive the partition function for interacting particles at finite temperature.",
        "Lattice defects alter phonon scattering and thermal conductivity.",
        "The experiment measures spin coherence with Ramsey interferometry.",
        "Relativistic corrections become significant at high-energy collision regimes.",
    ],
    "CHEM": [
        "The catalytic cycle proceeds through oxidative addition and reductive elimination.",
        "We quantify reaction kinetics from concentration profiles in batch synthesis.",
        "NMR spectroscopy confirms the aromatic substitution pattern of the product.",
        "Ligand field effects tune the redox potential of the metal complex.",
        "The mechanism involves nucleophilic attack followed by proton transfer.",
        "We evaluate stereoselectivity in asymmetric hydrogenation of ketones.",
        "Chromatography isolates intermediates before final purification.",
        "The molecular orbital analysis explains frontier electron interactions.",
        "Solvent polarity shifts equilibrium toward the ionized species.",
        "Infrared peaks indicate strong carbonyl stretching frequencies.",
    ],
    "CHEME": [
        "A packed-bed reactor improves conversion under controlled residence time.",
        "Mass transfer limitations dominate performance in membrane separation units.",
        "We optimize distillation column reflux ratio for energy-efficient operation.",
        "Process control uses model predictive control to stabilize reactor temperature.",
        "Heat exchanger network synthesis minimizes utility consumption.",
        "Adsorption isotherms guide design of cyclic gas separation processes.",
        "The pilot plant monitors pressure drop across fluidized catalyst beds.",
        "Reaction engineering balances selectivity and throughput in continuous flow.",
        "We perform pinch analysis for integrated thermal management.",
        "Transport phenomena couple diffusion and convection in porous media.",
    ],
}


def main() -> None:
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = out_dir / f"chatgpt_diagnostics_{ts}"
    jsonl_path = Path(str(base) + ".jsonl")
    csv_path = Path(str(base) + ".csv")
    md_path = Path(str(base) + ".md")

    rows: list[dict[str, object]] = []
    payload_records: list[dict[str, object]] = []

    for src in DOMAINS:
        for tgt in DOMAINS:
            for i, text in enumerate(SAMPLES[src], start=1):
                payload = {
                    "text": text,
                    "src": src,
                    "tgt": tgt,
                    "max_terms": 8,
                    "include_short_explanations": False,
                }
                t0 = time.perf_counter()
                status = 0
                response_json: dict[str, object] = {}
                error = ""
                try:
                    r = requests.post(API, json=payload, timeout=25)
                    status = int(r.status_code)
                    try:
                        decoded = r.json()
                        response_json = decoded if isinstance(decoded, dict) else {"response": decoded}
                    except Exception:
                        response_json = {"raw": r.text[:2000]}
                except Exception as exc:
                    error = str(exc)
                    response_json = {"request_error": error}
                latency = round(time.perf_counter() - t0, 4)

                terms = response_json.get("terms", []) if isinstance(response_json, dict) else []
                row = {
                    "combo": f"{src}->{tgt}",
                    "src": src,
                    "tgt": tgt,
                    "sample_idx": i,
                    "input_text": text,
                    "status": status,
                    "latency_sec": latency,
                    "predicted_src": str(response_json.get("predicted_src", "") if isinstance(response_json, dict) else ""),
                    "src_used": str(response_json.get("src_used", "") if isinstance(response_json, dict) else ""),
                    "terms_count": len(terms) if isinstance(terms, list) else 0,
                    "error": error,
                    "output_json": json.dumps(response_json, ensure_ascii=False),
                }
                rows.append(row)
                payload_records.append(
                    {
                        "request": payload,
                        "status": status,
                        "latency_sec": latency,
                        "response": response_json,
                    }
                )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in payload_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fields = [
        "combo",
        "src",
        "tgt",
        "sample_idx",
        "input_text",
        "status",
        "latency_sec",
        "predicted_src",
        "src_used",
        "terms_count",
        "error",
        "output_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[tuple[str, int, int, float, float, float]] = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            sub = [r for r in rows if r["src"] == src and r["tgt"] == tgt]
            n = len(sub)
            ok = sum(1 for r in sub if int(r["status"]) == 200)
            lats = [float(r["latency_sec"]) for r in sub]
            avg = round(sum(lats) / n, 4) if n else 0.0
            mx = round(max(lats), 4) if n else 0.0
            summary_rows.append((f"{src}->{tgt}", n, ok, round(ok / n, 3) if n else 0.0, avg, mx))

    lines: list[str] = []
    lines.append("# Comprehensive Diagnostics Report (ChatGPT Handoff)")
    lines.append("")
    lines.append(f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Endpoint: {API}")
    lines.append(f"- Total requests: {len(rows)}")
    lines.append(f"- Full payload log: {jsonl_path}")
    lines.append(f"- Full diagnostics CSV: {csv_path}")
    lines.append("")
    lines.append("## Summary by Combination")
    lines.append("")
    lines.append("| Combo | N | OK | Success Rate | Avg Latency (s) | Max Latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for combo, n, ok, success, avg, mx in summary_rows:
        lines.append(f"| {combo} | {n} | {ok} | {success:.3f} | {avg:.4f} | {mx:.4f} |")

    lines.append("")
    lines.append("## Complete Sample Input/Output Table")
    lines.append("")
    lines.append("| # | Combo | Sample | Input | Status | Latency(s) | predicted_src | src_used | terms | error |")
    lines.append("|---:|---|---:|---|---:|---:|---|---|---:|---|")
    for i, r in enumerate(rows, start=1):
        input_text = str(r["input_text"]).replace("|", "\\|")
        err = str(r["error"]).replace("|", "\\|")
        lines.append(
            f"| {i} | {r['combo']} | {r['sample_idx']} | {input_text} | {r['status']} | {float(r['latency_sec']):.4f} | {r['predicted_src']} | {r['src_used']} | {r['terms_count']} | {err} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `output_json` in the CSV contains the full JSON response for each sample.")
    lines.append("- The JSONL file contains complete request+response payloads for all samples.")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "markdown_report": str(md_path.resolve()),
                "diagnostics_csv": str(csv_path.resolve()),
                "payload_jsonl": str(jsonl_path.resolve()),
                "requests": len(rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

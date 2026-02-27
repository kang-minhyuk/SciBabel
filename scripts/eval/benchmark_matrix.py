from __future__ import annotations

import csv
import json
import statistics
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

API = "https://scibabel.onrender.com/annotate"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", choices=DOMAINS, default=None, help="Run only one source domain")
    parser.add_argument("--tag", default="", help="Optional suffix tag for output files")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout seconds")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent request workers")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    timeout_sec = max(1.0, float(args.timeout))
    workers = max(1, int(args.workers))

    src_list = [args.src] if args.src else DOMAINS
    tasks: list[tuple[str, str, int, str]] = []
    for src in src_list:
        for tgt in DOMAINS:
            for i, text in enumerate(SAMPLES[src], start=1):
                tasks.append((src, tgt, i, text))

    def _one(task: tuple[str, str, int, str]) -> dict[str, object]:
        src, tgt, i, text = task
        payload = {
            "text": text,
            "src": src,
            "tgt": tgt,
            "max_terms": 8,
            "include_short_explanations": False,
        }
        t0 = time.perf_counter()
        status = 0
        body: dict[str, object] = {}
        error = ""
        try:
            with requests.Session() as session:
                resp = session.post(API, json=payload, timeout=timeout_sec)
            status = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text[:200]}
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - t0
        terms = body.get("terms", []) if isinstance(body, dict) else []
        return {
            "src": src,
            "tgt": tgt,
            "sample_idx": i,
            "status": status,
            "latency_sec": round(latency, 4),
            "predicted_src": body.get("predicted_src", "") if isinstance(body, dict) else "",
            "src_used": body.get("src_used", "") if isinstance(body, dict) else "",
            "terms_count": len(terms) if isinstance(terms, list) else 0,
            "error": error or (body.get("detail", "") if isinstance(body, dict) else ""),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, task) for task in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())

    summary: list[dict[str, object]] = []
    for src in DOMAINS:
        for tgt in DOMAINS:
            sub = [r for r in rows if r["src"] == src and r["tgt"] == tgt]
            lat = [float(r["latency_sec"]) for r in sub]
            ok = [r for r in sub if int(r["status"]) == 200]
            term_counts = [int(r["terms_count"]) for r in ok]
            summary.append(
                {
                    "combo": f"{src}->{tgt}",
                    "n": len(sub),
                    "ok": len(ok),
                    "success_rate": round(len(ok) / len(sub), 3) if sub else 0.0,
                    "avg_latency_sec": round(sum(lat) / len(lat), 4) if lat else 0.0,
                    "p50_latency_sec": round(statistics.median(lat), 4) if lat else 0.0,
                    "max_latency_sec": round(max(lat), 4) if lat else 0.0,
                    "avg_terms": round(sum(term_counts) / len(term_counts), 2) if term_counts else 0.0,
                }
            )

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    raw_csv = out_dir / f"annotate_raw_{ts}{suffix}.csv"
    summary_csv = out_dir / f"annotate_summary_{ts}{suffix}.csv"
    report_md = out_dir / f"annotate_report_{ts}{suffix}.md"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Annotate Combination Report",
        "",
        f"Endpoint: {API}",
        f"Generated (UTC): {datetime.utcnow().isoformat()}Z",
        "",
        "| Combo | N | OK | Success | Avg(s) | P50(s) | Max(s) | AvgTerms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['combo']} | {s['n']} | {s['ok']} | {s['success_rate']:.3f} | {s['avg_latency_sec']:.4f} | {s['p50_latency_sec']:.4f} | {s['max_latency_sec']:.4f} | {s['avg_terms']:.2f} |"
        )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "raw_csv": str(raw_csv.resolve()),
                "summary_csv": str(summary_csv.resolve()),
                "report_md": str(report_md.resolve()),
                "summary": summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

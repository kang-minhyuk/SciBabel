from __future__ import annotations

import argparse
import statistics
import time

import requests

SAMPLES = [
    "The catalytic cycle proceeds through oxidative addition and reductive elimination.",
    "Mass transfer limitations dominate performance in membrane separation units.",
    "A packed-bed reactor improves conversion under controlled residence time.",
    "Process control uses model predictive control to stabilize reactor temperature.",
    "We optimize distillation column reflux ratio for energy-efficient operation.",
]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return values[0]
    if q >= 100:
        return values[-1]
    idx = int(round((q / 100.0) * (len(values) - 1)))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--p95-threshold", type=float, default=5.0)
    args = parser.parse_args()

    endpoint = args.api_base.rstrip("/") + "/annotate"

    latencies: list[float] = []
    ok = 0
    for i in range(max(1, int(args.requests))):
        text = SAMPLES[i % len(SAMPLES)]
        payload = {
            "text": text,
            "src": "CHEM",
            "tgt": "CHEME",
            "max_terms": 6,
            "include_short_explanations": False,
        }
        t0 = time.perf_counter()
        status = 0
        try:
            resp = requests.post(endpoint, json=payload, timeout=float(args.timeout))
            status = resp.status_code
            if status == 200:
                ok += 1
        except Exception:
            status = 0
        dt = time.perf_counter() - t0
        latencies.append(dt)
        print(f"run={i + 1:02d} status={status} latency_sec={dt:.4f}")

    sorted_lat = sorted(latencies)
    p50 = percentile(sorted_lat, 50)
    p95 = percentile(sorted_lat, 95)
    avg = statistics.mean(sorted_lat) if sorted_lat else 0.0
    max_v = max(sorted_lat) if sorted_lat else 0.0

    print("---")
    print(f"endpoint={endpoint}")
    print(f"requests={len(latencies)} ok={ok}")
    print(f"avg_sec={avg:.4f} p50_sec={p50:.4f} p95_sec={p95:.4f} max_sec={max_v:.4f}")

    if p95 > float(args.p95_threshold):
        raise SystemExit(f"FAIL: p95 {p95:.4f}s > {float(args.p95_threshold):.4f}s")

    print("PASS: bench threshold satisfied")


if __name__ == "__main__":
    main()

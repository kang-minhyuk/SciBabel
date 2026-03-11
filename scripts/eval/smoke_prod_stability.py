from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PAYLOAD = {
    "text": "The catalytic cycle proceeds through oxidative addition and reductive elimination.",
    "src": "CHEM",
    "tgt": "CHEME",
    "max_terms": 6,
    "include_short_explanations": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    base = args.api_base.rstrip("/")
    health = requests.get(base + "/health", timeout=args.timeout)
    if health.status_code != 200:
        raise SystemExit(f"FAIL: initial health={health.status_code}")

    statuses: list[int] = []

    def _one() -> int:
        r = requests.post(base + "/annotate", json=PAYLOAD, timeout=args.timeout)
        return int(r.status_code)

    with ThreadPoolExecutor(max_workers=max(2, int(args.n))) as ex:
        futures = [ex.submit(_one) for _ in range(int(args.n))]
        for fut in as_completed(futures):
            statuses.append(fut.result())

    ok = sum(1 for s in statuses if s == 200)
    busy = sum(1 for s in statuses if s == 429)
    other = [s for s in statuses if s not in {200, 429}]

    health2 = requests.get(base + "/health", timeout=args.timeout)
    if health2.status_code != 200:
        raise SystemExit(f"FAIL: final health={health2.status_code}")

    print({"statuses": statuses, "ok": ok, "busy": busy, "other": other})

    if other:
        raise SystemExit(f"FAIL: unexpected statuses {other}")

    print("PASS: service stable under burst")


if __name__ == "__main__":
    main()

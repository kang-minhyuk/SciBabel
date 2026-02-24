from __future__ import annotations

import argparse
import json
import sys

import requests


REQUIRED_FIELDS = [
    "predicted_src",
    "predicted_src_confidence",
    "predicted_src_probs",
    "src_used",
    "src_warning",
    "src_warning_reason",
]


SAMPLES = [
    ("We optimize a regularized loss with gradient descent and sparse representations.", "PM"),
    ("Phonon dispersion modifies band structure and transport behavior in solids.", "CSM"),
    ("Transition state stabilization and substituent effects control molecular selectivity.", "CHEME"),
    ("Distillation column reflux ratio and mass transfer govern process stability.", "CHEM"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test auto source detection on /annotate")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    health = requests.get(f"{args.api_base}/health", timeout=20)
    if health.status_code != 200:
        raise RuntimeError(f"/health failed: {health.status_code} {health.text}")

    outputs: list[dict[str, object]] = []
    for text, tgt in SAMPLES:
        resp = requests.post(
            f"{args.api_base}/annotate",
            json={
                "text": text,
                "src": "auto",
                "tgt": tgt,
                "max_terms": 6,
            },
            timeout=45,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/annotate failed: {resp.status_code} {resp.text}")
        body = resp.json()

        for key in REQUIRED_FIELDS:
            if key not in body:
                raise RuntimeError(f"missing field: {key}")
        if not isinstance(body.get("predicted_src_probs"), dict):
            raise RuntimeError("predicted_src_probs missing or invalid")
        if not body.get("src_used"):
            raise RuntimeError("src_used missing")

        outputs.append(
            {
                "target": tgt,
                "predicted_src": body.get("predicted_src"),
                "confidence": body.get("predicted_src_confidence"),
                "src_used": body.get("src_used"),
            }
        )

    print(json.dumps({"ok": True, "runs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE_AUTO_FAIL: {exc}")
        sys.exit(1)

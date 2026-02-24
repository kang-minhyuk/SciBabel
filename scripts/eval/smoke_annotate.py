from __future__ import annotations

import argparse
import json
import sys

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test annotate/explain APIs")
    parser.add_argument("--api-base", default="http://localhost:8000")
    args = parser.parse_args()

    text = "We optimize a graph neural network with sparse regularization under distribution shift."

    ann_req = {
        "text": text,
        "src": "auto",
        "tgt": "PM",
        "audience_level": "grad",
        "max_terms": 6,
        "include_short_explanations": False,
    }

    ann = requests.post(f"{args.api_base}/annotate", json=ann_req, timeout=45)
    if ann.status_code != 200:
        raise RuntimeError(f"/annotate failed: {ann.status_code} {ann.text}")
    ann_json = ann.json()

    assert "terms" in ann_json and isinstance(ann_json["terms"], list)
    if not ann_json["terms"]:
        raise RuntimeError("/annotate returned no terms; cannot run explain smoke")

    first = ann_json["terms"][0]
    term = str(first.get("term", "")).strip()
    analogs = [str(x.get("candidate", "")) for x in first.get("analogs", []) if isinstance(x, dict)]

    exp_req = {
        "text": text,
        "term": term,
        "src": "auto",
        "tgt": "PM",
        "audience_level": "grad",
        "analogs": analogs,
        "detail": "long",
    }

    exp = requests.post(f"{args.api_base}/explain", json=exp_req, timeout=60)
    if exp.status_code != 200:
        raise RuntimeError(f"/explain failed: {exp.status_code} {exp.text}")
    exp_json = exp.json()

    required = [
        "term",
        "short_explanation",
        "long_explanation",
        "caution_label",
        "semantic_policy",
    ]
    for k in required:
        if k not in exp_json:
            raise RuntimeError(f"/explain missing required field: {k}")

    blob = (str(exp_json.get("short_explanation", "")) + " " + str(exp_json.get("long_explanation", ""))).lower()
    banned = exp_json.get("semantic_policy", {}).get("banned_phrases", [])
    must_include = exp_json.get("semantic_policy", {}).get("must_include_any", [])

    if any(str(b).lower() in blob for b in banned):
        raise RuntimeError("/explain violated banned phrase policy")
    if not any(str(m).lower() in blob for m in must_include):
        raise RuntimeError("/explain missing required analogy language")

    print(json.dumps({"annotate_terms": len(ann_json["terms"]), "explain_ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE_FAIL: {exc}")
        sys.exit(1)

from __future__ import annotations

import json
import sys

from terms.extract import extract_terms

BANNED = {"under", "while", "reduce"}


def main() -> None:
    sents = [
        "Our transformer uses sparse attention to reduce memory while preserving long-range dependencies.",
        "A SE(3)-equivariant model improves geometric learning in protein structure prediction.",
        "We optimize k-space trajectories for accelerated MRI reconstruction.",
        "The updatedgpt_noself_v2 marker should never appear in extracted terms.",
        "Process intensification in catalytic packed-bed reactors improves conversion under diffusion limits.",
    ]

    rows = []
    for s in sents:
        terms = extract_terms(s, max_terms=12)
        if not terms:
            raise RuntimeError(f"No terms extracted for sentence: {s}")
        for t in terms:
            low = str(t["term"]).lower()
            if any(b == low for b in BANNED):
                raise RuntimeError(f"Banned trivial token extracted: {low}")
            if "updatedgpt" in low or "native=" in low:
                raise RuntimeError(f"Debug artifact leaked: {low}")
        rows.append({"text": s, "terms": terms})

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE_FAIL: {exc}")
        sys.exit(1)

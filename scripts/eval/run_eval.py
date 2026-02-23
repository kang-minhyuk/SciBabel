from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def default_out_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"translation_eval_{ts}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SciBabel eval cases against /translate")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--cases", default="scripts/eval/cases.jsonl")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases = load_cases(cases_path)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    out_path = Path(args.out) if args.out else default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    with out_path.open("w", encoding="utf-8") as wf:
        for idx, case in enumerate(cases, start=1):
            payload = {
                "text": case["text"],
                "src": case["src"],
                "tgt": case["tgt"],
                "k": args.k,
            }

            try:
                resp = requests.post(
                    f"{args.api_base}/translate",
                    json=payload,
                    timeout=args.timeout,
                )
                status = resp.status_code
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text}
            except Exception as exc:
                status = 0
                body = {"detail": f"request_error: {exc}"}

            if status == 200:
                ok_count += 1

            record = {
                "index": idx,
                "id": case.get("id", f"case_{idx}"),
                "src": case["src"],
                "tgt": case["tgt"],
                "text": case["text"],
                "status": status,
                "ok": status == 200,
                "response": body,
            }
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote eval log: {out_path}")
    print(f"Cases: {len(cases)} | OK: {ok_count} | Error: {len(cases) - ok_count}")


if __name__ == "__main__":
    main()

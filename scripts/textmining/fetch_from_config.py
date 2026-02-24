from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_yaml


def run_cmd(cmd: list[str], allow_fail: bool = False) -> None:
    print("RUN:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    if result.returncode != 0 and allow_fail:
        print(f"WARN: command failed (continuing): {' '.join(cmd)}")


def fetch_arxiv(cfg: dict, max_results: int, start_date: str, end_date: str) -> None:
    domains = cfg.get("domains", {})
    for domain, spec in domains.items():
        if "arxiv" not in (spec.get("sources") or []):
            continue
        categories = spec.get("arxiv", {}).get("categories", [])
        out = Path("data/raw/arxiv") / f"{domain.lower()}_arxiv.jsonl"
        cmd = [
            sys.executable,
            "scripts/textmining/fetch_arxiv.py",
            "--domain",
            domain,
            "--max-results",
            str(max_results),
            "--out",
            str(out),
        ]
        if start_date:
            cmd += ["--start-date", start_date]
        if end_date:
            cmd += ["--end-date", end_date]
        for c in categories:
            cmd += ["--category", str(c)]
        run_cmd(cmd)


def fetch_chemrxiv(cfg: dict, max_results: int) -> None:
    domains = cfg.get("domains", {})
    for domain, spec in domains.items():
        if "chemrxiv" not in (spec.get("sources") or []):
            continue
        filters = spec.get("chemrxiv", {}).get("category_filters", [])
        out = Path("data/raw/chemrxiv") / f"{domain.lower()}_chemrxiv.jsonl"
        cmd = [
            sys.executable,
            "scripts/textmining/fetch_chemrxiv.py",
            "--domain",
            domain,
            "--max-results",
            str(max_results),
            "--out",
            str(out),
        ]
        for f in filters:
            cmd += ["--category-filter", str(f)]
        run_cmd(cmd, allow_fail=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv/ChemRxiv by domains.yaml")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--source", choices=["arxiv", "chemrxiv", "all"], default="all")
    parser.add_argument("--max-results", type=int, default=1500)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.source in {"arxiv", "all"}:
        fetch_arxiv(cfg, args.max_results, args.start_date, args.end_date)
    if args.source in {"chemrxiv", "all"}:
        fetch_chemrxiv(cfg, args.max_results)


if __name__ == "__main__":
    main()

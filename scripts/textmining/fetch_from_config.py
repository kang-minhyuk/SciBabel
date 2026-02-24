from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_yaml


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _fetch_arxiv_domain(domain: str, spec: dict, max_results: int, start_date: str, end_date: str) -> None:
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


def _fetch_chemrxiv_domain(domain: str, spec: dict, max_results: int, verbose: bool) -> None:
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
    if verbose:
        cmd.append("--verbose")
    for f in filters:
        cmd += ["--category-filter", str(f)]
    run_cmd(cmd)


def _fetch_openalex_domain(domain: str, spec: dict, max_results: int, verbose: bool) -> None:
    out = Path("data/raw/openalex") / f"{domain.lower()}_openalex.jsonl"
    oa = spec.get("openalex", {})
    concept_ids = oa.get("concept_ids", [])
    keywords = oa.get("keywords", [])
    display_name = str(oa.get("display_name") or "")
    cmd = [
        sys.executable,
        "scripts/textmining/fetch_openalex.py",
        "--domain",
        domain,
        "--max-results",
        str(max_results),
        "--out",
        str(out),
    ]
    if display_name:
        cmd += ["--display-name", display_name]
    for c in concept_ids:
        cmd += ["--concept-id", str(c)]
    for k in keywords:
        cmd += ["--keyword", str(k)]
    if verbose:
        cmd.append("--verbose")
    run_cmd(cmd)


def fetch_by_priority(cfg: dict, max_results: int, start_date: str, end_date: str, source_filter: str, verbose: bool) -> None:
    domains = cfg.get("domains", {})
    for domain, spec in domains.items():
        sources = [str(s) for s in (spec.get("sources") or [])]
        for src in sources:
            if source_filter != "all" and source_filter != src:
                continue
            try:
                if src == "arxiv":
                    _fetch_arxiv_domain(domain, spec, max_results, start_date, end_date)
                elif src == "chemrxiv":
                    _fetch_chemrxiv_domain(domain, spec, max_results, verbose)
                elif src == "openalex":
                    _fetch_openalex_domain(domain, spec, max_results, verbose)
                else:
                    print(f"WARN: unsupported source '{src}' for {domain}, skipping")
            except Exception as exc:
                # If this source has a lower-priority alternative configured, continue.
                src_idx = sources.index(src)
                has_alternative = source_filter == "all" and any(sources[src_idx + 1 :])
                if has_alternative:
                    print(
                        f"WARN: source {src} failed for {domain}: {exc}. "
                        f"Trying next source in priority list."
                    )
                    continue
                raise RuntimeError(
                    f"Source {src} failed for {domain} and no alternate source remains: {exc}"
                ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv/ChemRxiv by domains.yaml")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--source", choices=["arxiv", "chemrxiv", "openalex", "all"], default="all")
    parser.add_argument("--max-results", type=int, default=1500)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    fetch_by_priority(
        cfg=cfg,
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date,
        source_filter=args.source,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

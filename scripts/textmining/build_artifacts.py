from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_step(args: list[str]) -> None:
    print("RUN:", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SciBabel annotation artifacts without running web server startup jobs")
    parser.add_argument("--config", default="configs/textmining/domains.yaml")
    parser.add_argument("--skip-build-corpus", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_build_corpus:
        run_step(
            [
                py,
                "scripts/textmining/build_corpus.py",
                "--config",
                args.config,
                "--out-full",
                "data/processed/corpus_full.parquet",
                "--out-balanced",
                "data/processed/corpus_balanced.parquet",
                "--diagnostics-out",
                "reports/textmining/corpus_scale_report.md",
            ]
        )

    run_step(
        [
            py,
            "scripts/textmining/mine_terms.py",
            "--corpus",
            "data/processed/corpus_full.parquet",
            "--term-stats-out",
            "data/processed/term_stats.csv",
            "--lexicon-out",
            "data/processed/domain_lexicon.json",
            "--report-out",
            "reports/textmining/lexicon_report.md",
        ]
    )

    run_step(
        [
            py,
            "scripts/textmining/train_domain_classifier.py",
            "--corpus",
            "data/processed/corpus_full.parquet",
            "--model-out",
            "models/domain_clf.joblib",
            "--report-out",
            "reports/textmining/classifier_report.md",
            "--metrics-out",
            "models/domain_clf_metrics.json",
        ]
    )

    if not args.skip_validate:
        run_step(
            [
                py,
                "scripts/textmining/validate_artifacts.py",
                "--lexicon",
                "data/processed/domain_lexicon.json",
                "--term-stats",
                "data/processed/term_stats.csv",
                "--clf-metrics",
                "models/domain_clf_metrics.json",
                "--report-out",
                "reports/textmining/validation_report.md",
            ]
        )

    print("build_artifacts complete")


if __name__ == "__main__":
    main()

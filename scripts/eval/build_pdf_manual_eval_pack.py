from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_csv(path: Path) -> None:
    header = [
        "pdf_filename",
        "source_domain_expected",
        "target_domain",
        "extraction_quality_1_to_5",
        "source_prediction_correct_yes_no",
        "ambiguity_warning_reasonable_yes_no",
        "term_fragment_errors_count",
        "good_term_hits_count",
        "important_terms_missed_count",
        "noisy_terms_count",
        "explanation_quality_1_to_5",
        "explanation_latency_note",
        "page_mapping_quality_1_to_5",
        "overall_usefulness_1_to_5",
        "notes",
    ]
    rows = [
        ["sample_cs_paper.pdf", "CSM", "PM", "", "", "", "", "", "", "", "", "", "", "", ""],
        ["sample_pm_paper.pdf", "PM", "CHEM", "", "", "", "", "", "", "", "", "", "", "", ""],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_guide(path: Path) -> None:
    lines = [
        "# PDF Manual Evaluation Guide",
        "",
        f"- Generated: {_utc_ts()}",
        "",
        "## Start Local Services",
        "",
        "1. Start backend + frontend:",
        "   - `make pdf-dev`",
        "2. Open URLs:",
        "   - frontend: `http://localhost:3000`",
        "   - backend health: `http://127.0.0.1:8000/health`",
        "",
        "## Upload and Evaluate",
        "",
        "1. Upload a real paper PDF in the UI.",
        "2. Pick `src=auto` first and your test target domain.",
        "3. Click `Annotate PDF`.",
        "4. Inspect highlighted terms and page mapping.",
        "5. Click flagged terms and test `Explain This Term`.",
        "",
        "## Scoring Guidance",
        "",
        "Good flagged term:",
        "- A domain concept likely unfamiliar in target domain.",
        "",
        "Fragment error:",
        "- Broken phrase fragment like `optimize a graph`, `train a diffusion`, `transition is characterized`, `memory cost on`.",
        "",
        "Missed important term:",
        "- Core paper concept not detected/flagged despite importance.",
        "",
        "Explanation checks:",
        "- Correct concept meaning, faithful to context, useful analogy.",
        "- Not generic, not contradictory, not empty.",
        "",
        "## Fill the CSV",
        "",
        "- Use `reports/pdf_manual_eval/manual_eval_sheet.csv`.",
        "- One row per PDF.",
        "- Use counts for fragment/missed/noisy patterns.",
        "- Add concise notes with page examples.",
        "",
        "## Optional Backend-only Checks",
        "",
        "- `python3 scripts/eval/run_pdf_local_eval.py --pdf path/to/paper.pdf`",
        "- `python3 scripts/eval/run_pdf_local_eval.py --pdf-dir path/to/pdfs --target PM`",
        "- `python3 scripts/eval/run_pdf_local_eval.py --pdf path/to/paper.pdf --explain-top-k 5`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_examples(path: Path) -> None:
    lines = [
        "# PDF Manual Evaluation Examples",
        "",
        f"- Generated: {_utc_ts()}",
        "",
        "## Example Judgments",
        "",
        "### Good concept term",
        "- `graph neural network` flagged for PM target: good hit.",
        "",
        "### Bad fragment term",
        "- `memory cost on` flagged: count as fragment error.",
        "",
        "### Noisy extraction artifact",
        "- Repeated page header appears as terms on many pages: count as noisy terms.",
        "",
        "### Missed key concept",
        "- Paper focuses on `classifier-free guidance`, but term is missing: increment missed important terms.",
        "",
        "### Good vs weak explanation",
        "- Good: ties term to page context and target-domain analogy.",
        "- Weak: generic, short, context-free, or contradictory.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build manual eval pack for real PDF QA")
    ap.add_argument("--out-dir", default="reports/pdf_manual_eval")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    guide = out_dir / "manual_eval_guide.md"
    sheet = out_dir / "manual_eval_sheet.csv"
    examples = out_dir / "manual_eval_examples.md"

    _write_guide(guide)
    _write_csv(sheet)
    _write_examples(examples)

    print(
        json.dumps(
            {
                "manual_eval_guide": str(guide.resolve()),
                "manual_eval_sheet": str(sheet.resolve()),
                "manual_eval_examples": str(examples.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _latest(pattern: str, root: Path) -> str:
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write baseline metadata for term-lens freeze.")
    parser.add_argument("--out-dir", default="reports/baselines")
    parser.add_argument("--backend-test-summary", default="36 passed, 2 warnings")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    local_md = _latest("reports/term_lens_regression_local_*.md", repo_root)
    prod_md = _latest("reports/term_lens_regression_prod_*.md", repo_root)
    compare_md = _latest("reports/term_lens_compare_*.md", repo_root)
    compare_json = _latest("reports/term_lens_compare_*.json", repo_root)

    metadata = {
        "version": "v0.1-term-lens-baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "reports": {
            "local_regression": local_md,
            "prod_regression": prod_md,
            "compare_report": compare_md,
            "compare_json": compare_json,
        },
        "backend_test_summary": args.backend_test_summary,
        "thresholds": {
            "SRC_AUTO_MIN_CONF": os.getenv("SRC_AUTO_MIN_CONF", "0.55"),
            "SRC_AUTO_MIN_GAP": os.getenv("SRC_AUTO_MIN_GAP", "0.10"),
            "SRC_AUTO_MIN_WORDS": os.getenv("SRC_AUTO_MIN_WORDS", "8"),
            "ANALOG_MIN_SCORE": os.getenv("ANALOG_MIN_SCORE", "0.55"),
            "ANALOG_MAX_RETURN": os.getenv("ANALOG_MAX_RETURN", "5"),
        },
        "tag_suggestion": "v0.1-term-lens-baseline",
        "tag_command": "git tag -a v0.1-term-lens-baseline -m \"Term lens baseline freeze\" && git push origin v0.1-term-lens-baseline",
    }

    json_path = out_dir / "v0_1_term_lens_baseline.json"
    md_path = out_dir / "v0_1_term_lens_baseline.md"

    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# v0.1 Term Lens Baseline",
        "",
        f"- Created (UTC): {metadata['created_at']}",
        f"- Git commit: {metadata['git_commit']}",
        f"- Local regression report: {local_md}",
        f"- Prod regression report: {prod_md}",
        f"- Compare report: {compare_md}",
        f"- Backend tests: {metadata['backend_test_summary']}",
        "",
        "## Threshold Snapshot",
        "",
        f"- SRC_AUTO_MIN_CONF={metadata['thresholds']['SRC_AUTO_MIN_CONF']}",
        f"- SRC_AUTO_MIN_GAP={metadata['thresholds']['SRC_AUTO_MIN_GAP']}",
        f"- SRC_AUTO_MIN_WORDS={metadata['thresholds']['SRC_AUTO_MIN_WORDS']}",
        f"- ANALOG_MIN_SCORE={metadata['thresholds']['ANALOG_MIN_SCORE']}",
        f"- ANALOG_MAX_RETURN={metadata['thresholds']['ANALOG_MAX_RETURN']}",
        "",
        "## Tag Suggestion",
        "",
        f"- Suggested tag: `{metadata['tag_suggestion']}`",
        f"- Command: `{metadata['tag_command']}`",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"baseline_md": str(md_path.resolve()), "baseline_json": str(json_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

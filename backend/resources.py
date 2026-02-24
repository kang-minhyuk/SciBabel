from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from domain_detect import SourceDetector
from llm.fake_client import FakeExplainClient
from llm.openai_client import OpenAIExplainClient
from terms.engine import TermAnnotationEngine

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "domain_clf.joblib"
LEXICON_PATH = ROOT / "data" / "processed" / "domain_lexicon.json"
TERM_STATS_PATH = ROOT / "data" / "processed" / "term_stats.csv"
EXPLAIN_CACHE_DB = ROOT / "backend" / "explain_cache.sqlite3"


class ArtifactsMissingError(RuntimeError):
    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        super().__init__("Missing required artifacts")


@dataclass
class ResourceBundle:
    annotation_engine: TermAnnotationEngine
    source_detector: SourceDetector
    explain_client: OpenAIExplainClient | FakeExplainClient | None = None


_LOCK = threading.Lock()
_BUNDLE: ResourceBundle | None = None


def required_artifacts() -> list[Path]:
    return [LEXICON_PATH, TERM_STATS_PATH, MODEL_PATH]


def missing_artifacts() -> list[str]:
    return [str(p) for p in required_artifacts() if not p.exists()]


def _new_explain_client() -> OpenAIExplainClient | FakeExplainClient:
    if os.getenv("SCIBABEL_FAKE_LLM", "0") == "1":
        return FakeExplainClient()
    return OpenAIExplainClient(db_path=EXPLAIN_CACHE_DB)


def _build_bundle(load_explain: bool) -> ResourceBundle:
    missing = missing_artifacts()
    if missing:
        raise ArtifactsMissingError(missing)

    src_th = float(os.getenv("UNFAMILIAR_SRC_THRESHOLD", "0.35"))
    tgt_th = float(os.getenv("UNFAMILIAR_TGT_THRESHOLD", "0.45"))
    analog_th = float(os.getenv("ANALOG_SIM_THRESHOLD", "0.20"))

    bundle = ResourceBundle(
        annotation_engine=TermAnnotationEngine(
            root=ROOT,
            src_threshold=src_th,
            tgt_threshold=tgt_th,
            analog_threshold=analog_th,
        ),
        source_detector=SourceDetector(model_path=MODEL_PATH),
        explain_client=None,
    )

    if load_explain:
        bundle.explain_client = _new_explain_client()

    return bundle


def get_resources(*, load_explain: bool = False) -> ResourceBundle:
    global _BUNDLE

    with _LOCK:
        if _BUNDLE is None:
            _BUNDLE = _build_bundle(load_explain=load_explain)
        elif load_explain and _BUNDLE.explain_client is None:
            _BUNDLE.explain_client = _new_explain_client()

        return _BUNDLE


def check_ready() -> dict[str, object]:
    missing = missing_artifacts()
    if missing:
        return {"ready": False, "missing": missing}
    try:
        get_resources(load_explain=False)
        return {"ready": True}
    except ArtifactsMissingError as exc:
        return {"ready": False, "missing": exc.missing}
    except Exception:
        # Artifacts exist but load failed
        return {"ready": False, "missing": []}

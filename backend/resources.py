from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from domain_detect import SourceDetector
from llm.fake_client import FakeExplainClient
from llm.openai_client import OpenAIExplainClient
from terms.engine import TermAnnotationEngine
from terms.extract import set_nlp

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


class ResourceManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bundle: ResourceBundle | None = None
        self._spacy_nlp: object | None = None

        default_env = "production" if os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"} else "dev"
        env = os.getenv("SCIBABEL_ENV", default_env).strip().lower()
        self.scibabel_env = env if env in {"dev", "production"} else "dev"
        default_evidence = "false" if self.scibabel_env == "production" else "true"
        self.evidence_enabled = os.getenv("EVIDENCE_ENABLED", default_evidence).strip().lower() in {"1", "true", "yes", "on"}

    def check_artifacts_present(self) -> tuple[bool, list[str], dict[str, str]]:
        artifacts = {
            "lexicon": str(LEXICON_PATH),
            "term_stats": str(TERM_STATS_PATH),
            "domain_clf": str(MODEL_PATH),
        }
        missing: list[str] = []
        for p in [LEXICON_PATH, TERM_STATS_PATH, MODEL_PATH]:
            try:
                if (not p.exists()) or p.stat().st_size <= 16:
                    missing.append(str(p))
            except Exception:
                missing.append(str(p))
        return len(missing) == 0, missing, artifacts

    def load_light_resources(self, *, load_explain: bool = False) -> ResourceBundle:
        with self._lock:
            ok, missing, _ = self.check_artifacts_present()
            if not ok:
                raise ArtifactsMissingError(missing)

            if self._bundle is None:
                t0 = perf_counter()
                src_th = float(os.getenv("UNFAMILIAR_SRC_THRESHOLD", "0.35"))
                tgt_th = float(os.getenv("UNFAMILIAR_TGT_THRESHOLD", "0.45"))
                analog_th = float(os.getenv("ANALOG_SIM_THRESHOLD", "0.20"))

                self._bundle = ResourceBundle(
                    annotation_engine=TermAnnotationEngine(
                        root=ROOT,
                        src_threshold=src_th,
                        tgt_threshold=tgt_th,
                        analog_threshold=analog_th,
                    ),
                    source_detector=SourceDetector(model_path=MODEL_PATH),
                    explain_client=None,
                )
                print(
                    "[resources] load_light_resources_sec=",
                    round(perf_counter() - t0, 4),
                    "env=",
                    self.scibabel_env,
                    "evidence_enabled=",
                    self.evidence_enabled,
                )

            if load_explain and self._bundle.explain_client is None:
                self._bundle.explain_client = _new_explain_client()

            return self._bundle

    def get_spacy(self) -> object:
        with self._lock:
            if self._spacy_nlp is not None:
                return self._spacy_nlp

            t0 = perf_counter()
            model = os.getenv("SPACY_MODEL", "en_core_web_sm")
            try:
                import importlib

                spacy = importlib.import_module("spacy")
                use_full_model = True
                if self.scibabel_env == "production":
                    use_full_model = os.getenv("SPACY_LOAD_MODEL_IN_PROD", "false").strip().lower() in {"1", "true", "yes", "on"}
                if use_full_model:
                    self._spacy_nlp = spacy.load(model)
                else:
                    self._spacy_nlp = spacy.blank("en")
                    if "sentencizer" not in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.add_pipe("sentencizer")
                    print("[resources] using_spacy_blank_in_production")
            except Exception as exc:
                try:
                    import importlib

                    spacy = importlib.import_module("spacy")
                    self._spacy_nlp = spacy.blank("en")
                    if "sentencizer" not in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.add_pipe("sentencizer")
                    print(f"[resources] spacy_load_fallback model={model} reason={exc}")
                except Exception:
                    self._spacy_nlp = None
                    print(f"[resources] spacy_unavailable reason={exc}")

            if self._spacy_nlp is not None:
                set_nlp(self._spacy_nlp)
            print("[resources] spacy_load_sec=", round(perf_counter() - t0, 4))
            return self._spacy_nlp


_MANAGER = ResourceManager()


def required_artifacts() -> list[Path]:
    return [LEXICON_PATH, TERM_STATS_PATH, MODEL_PATH]


def missing_artifacts() -> list[str]:
    return [str(p) for p in required_artifacts() if not p.exists()]


def _new_explain_client() -> OpenAIExplainClient | FakeExplainClient:
    if os.getenv("SCIBABEL_FAKE_LLM", "0") == "1":
        return FakeExplainClient()
    return OpenAIExplainClient(db_path=EXPLAIN_CACHE_DB)


def _build_bundle(load_explain: bool) -> ResourceBundle:
    return _MANAGER.load_light_resources(load_explain=load_explain)


def get_resources(*, load_explain: bool = False) -> ResourceBundle:
    return _MANAGER.load_light_resources(load_explain=load_explain)


def get_spacy() -> object:
    return _MANAGER.get_spacy()


def check_ready() -> dict[str, object]:
    ok, missing, artifacts = _MANAGER.check_artifacts_present()
    return {"ready": ok, "missing": missing, "artifacts": artifacts}

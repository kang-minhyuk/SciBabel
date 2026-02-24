from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _snap(paths: list[Path]) -> dict[str, tuple[bool, float | None]]:
    out: dict[str, tuple[bool, float | None]] = {}
    for p in paths:
        if p.exists():
            out[str(p)] = (True, p.stat().st_mtime)
        else:
            out[str(p)] = (False, None)
    return out


def test_import_does_not_train_or_write_artifacts(monkeypatch) -> None:
    import joblib
    import subprocess
    import domain_detect
    import terms.engine

    called = {"subprocess": False, "joblib_dump": False, "engine_init": False, "detector_init": False}

    def _boom_subprocess(*_args, **_kwargs):
        called["subprocess"] = True
        raise AssertionError("subprocess.run should not be used at app import/startup path")

    def _boom_dump(*_args, **_kwargs):
        called["joblib_dump"] = True
        raise AssertionError("joblib.dump should not be used at app import/startup path")

    orig_engine_init = terms.engine.TermAnnotationEngine.__init__
    orig_detector_init = domain_detect.SourceDetector.__init__

    def _engine_init_probe(self, *args, **kwargs):
        called["engine_init"] = True
        return orig_engine_init(self, *args, **kwargs)

    def _detector_init_probe(self, *args, **kwargs):
        called["detector_init"] = True
        return orig_detector_init(self, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _boom_subprocess)
    monkeypatch.setattr(joblib, "dump", _boom_dump)
    monkeypatch.setattr(terms.engine.TermAnnotationEngine, "__init__", _engine_init_probe)
    monkeypatch.setattr(domain_detect.SourceDetector, "__init__", _detector_init_probe)

    root = Path(__file__).resolve().parents[2]
    tracked = [
        root / "models" / "domain_clf.joblib",
        root / "data" / "processed" / "domain_lexicon.json",
        root / "data" / "processed" / "term_stats.csv",
    ]
    before = _snap(tracked)

    sys.modules.pop("app", None)
    importlib.import_module("app")

    after = _snap(tracked)

    assert called["subprocess"] is False
    assert called["joblib_dump"] is False
    assert called["engine_init"] is False
    assert called["detector_init"] is False
    assert before == after

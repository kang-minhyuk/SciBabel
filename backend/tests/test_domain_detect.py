from __future__ import annotations

from pathlib import Path

from domain_detect import SourceDetector


class _FakeClf:
    def __init__(self, classes_: list[str], probs: list[float]) -> None:
        self.classes_ = classes_
        self._probs = probs

    def predict_proba(self, _rows):
        return [self._probs]


def _detector() -> SourceDetector:
    root = Path(__file__).resolve().parents[2]
    return SourceDetector(model_path=root / "models" / "domain_clf.joblib")


def test_detect_too_short_is_ambiguous() -> None:
    det = _detector()
    out = det.detect_source("short text")
    assert out["is_ambiguous"] is True
    assert out["reason"] == "too_short"
    assert isinstance(out["probs"], dict)


def test_detect_samples_have_expected_domains() -> None:
    det = _detector()

    samples = [
        ("We optimize a regularized loss with gradient descent and neural representations for robust generalization in machine learning systems.", "CSM"),
        ("Phonon dispersion modifies band structure and transport behavior in crystalline materials under thermal perturbation.", "PM"),
        ("Transition state stabilization and substituent effects guide reaction mechanism and molecular selectivity in synthesis chemistry.", "CHEM"),
        ("Distillation column reflux ratio and mass transfer constraints determine reactor conversion and process control stability.", "CHEME"),
    ]

    for text, expected in samples:
        out = det.detect_source(text)
        assert out["predicted_src"] == expected
        assert isinstance(out["top2_gap"], float)
        assert expected in out["probs"]


def test_top2_gap_and_ambiguity_rule() -> None:
    det = _detector()
    text = (
        "Reactor design uses optimization and model constraints for process transport and computational planning "
        "under uncertainty in coupled systems."
    )
    out = det.detect_source(text)
    assert "top2_gap" in out
    if out["confidence"] < 0.55 or out["top2_gap"] < 0.10:
        assert out["is_ambiguous"] is True


def test_high_confidence_large_gap_is_not_ambiguous() -> None:
    det = _detector()
    det.clf = _FakeClf(["CSM", "PM", "CHEM", "CHEME"], [0.88, 0.04, 0.04, 0.04])
    out = det.detect_source("This sentence is long enough to bypass the too-short guard by using many tokens in one line.")
    assert out["predicted_src"] == "CSM"
    assert out["confidence"] == 0.88
    assert round(float(out["top2_gap"]), 2) == 0.84
    assert out["is_ambiguous"] is False


def test_low_confidence_small_gap_is_ambiguous() -> None:
    det = _detector()
    det.clf = _FakeClf(["CSM", "PM", "CHEM", "CHEME"], [0.53, 0.47, 0.0, 0.0])
    out = det.detect_source("This sentence is long enough to bypass the too-short guard by using many tokens in one line.")
    assert out["predicted_src"] == "CSM"
    assert out["confidence"] == 0.53
    assert round(float(out["top2_gap"]), 2) == 0.06
    assert out["is_ambiguous"] is True

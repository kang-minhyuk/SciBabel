from pathlib import Path

from terms.engine import TermAnnotationEngine


def test_evidence_lookup_uses_precomputed_index(tmp_path: Path) -> None:
    root = tmp_path
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)

    (root / "data" / "processed" / "domain_lexicon.json").write_text(
        '{"CSM": ["graph neural network"], "PM": ["phase transition"], "CHEM": ["nmr spectroscopy"], "CHEME": ["packed-bed reactor"], "CCE": []}',
        encoding="utf-8",
    )
    (root / "data" / "processed" / "term_stats.csv").write_text(
        "domain,term,z\nCSM,graph neural network,1.2\n",
        encoding="utf-8",
    )
    (root / "models" / "domain_clf.joblib").write_bytes(b"stub")
    (root / "data" / "processed" / "evidence_index.json").write_text(
        '{"PM::phase transition": [{"snippet": "order parameter near criticality", "doc_id": "doc-1", "source": "toy"}]}',
        encoding="utf-8",
    )

    engine = object.__new__(TermAnnotationEngine)
    engine.evidence_enabled = True
    engine.evidence_index_path = root / "data" / "processed" / "evidence_index.json"
    engine.evidence_index = TermAnnotationEngine._load_evidence_index(engine)

    rows = TermAnnotationEngine._evidence_lookup(engine, tgt="PM", phrase="phase transition", max_hits=2)
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "doc-1"
    assert "criticality" in rows[0]["snippet"]

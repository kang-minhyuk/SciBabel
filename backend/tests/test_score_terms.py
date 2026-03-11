from terms.score import TermScoreConfig, score_terms


def _lexicon() -> dict[str, list[str]]:
    return {
        "CSM": ["graph neural network", "distribution shift", "temperature scaling"],
        "PM": ["hamiltonian", "phase transition", "thermal conductivity"],
        "CHEM": ["nmr spectroscopy"],
        "CHEME": ["packed-bed reactor", "model predictive control"],
    }


def _stats() -> dict[str, dict[str, float]]:
    return {
        "CSM": {
            "graph neural network": 3.4,
            "distribution shift": 2.9,
            "temperature scaling": 2.1,
            "packed": 0.4,
            "bed": 0.2,
            "reactor": 0.2,
        },
        "PM": {
            "graph neural network": -1.2,
            "distribution shift": -0.8,
            "temperature scaling": -0.9,
            "thermal conductivity": 2.4,
            "packed": -0.2,
            "bed": -0.2,
            "reactor": -0.1,
        },
        "CHEM": {
            "nmr spectroscopy": 2.5,
        },
        "CHEME": {
            "packed-bed reactor": 2.2,
            "model predictive control": 2.7,
            "packed": 1.5,
            "bed": 1.2,
            "reactor": 1.8,
        },
    }


def test_score_normalized_capitalization_resolves_same_values() -> None:
    extracted = [{"term": "Graph Neural Network", "start": 0, "end": 20}]
    out = score_terms(
        extracted_terms=extracted,
        src="CSM",
        tgt="PM",
        all_domains=["CSM", "PM", "CHEM", "CHEME"],
        term_stats=_stats(),
        lexicon_by_domain=_lexicon(),
        cfg=TermScoreConfig(src_threshold=0.35, tgt_threshold=0.45),
    )
    assert len(out) == 1
    assert out[0]["familiarity_source"] in {"exact", "normalized"}
    assert out[0]["distinctiveness_source"] in {"exact", "normalized"}


def test_score_component_fallback_for_hyphenated_phrase() -> None:
    extracted = [{"term": "packed-bed reactor", "start": 0, "end": 17}]
    stats = _stats()
    stats["CHEME"].pop("packed-bed reactor", None)
    out = score_terms(
        extracted_terms=extracted,
        src="CHEME",
        tgt="PM",
        all_domains=["CSM", "PM", "CHEM", "CHEME"],
        term_stats=stats,
        lexicon_by_domain=_lexicon(),
        cfg=TermScoreConfig(src_threshold=0.35, tgt_threshold=0.45),
    )
    assert len(out) == 1
    assert out[0]["distinctiveness_source"] == "component_fallback"


def test_same_field_normal_mode_is_more_conservative() -> None:
    extracted = [
        {"term": "graph neural network", "start": 0, "end": 20},
        {"term": "distribution shift", "start": 21, "end": 39},
    ]
    out = score_terms(
        extracted_terms=extracted,
        src="CSM",
        tgt="CSM",
        all_domains=["CSM", "PM", "CHEM", "CHEME"],
        term_stats=_stats(),
        lexicon_by_domain=_lexicon(),
        cfg=TermScoreConfig(src_threshold=0.35, tgt_threshold=0.45),
        same_field_mode="normal",
    )
    assert all(bool(row["flagged"]) is False for row in out)


def test_cross_domain_still_flags_distinctive_terms() -> None:
    extracted = [{"term": "graph neural network", "start": 0, "end": 20}]
    out = score_terms(
        extracted_terms=extracted,
        src="CSM",
        tgt="PM",
        all_domains=["CSM", "PM", "CHEM", "CHEME"],
        term_stats=_stats(),
        lexicon_by_domain=_lexicon(),
        cfg=TermScoreConfig(src_threshold=0.35, tgt_threshold=0.45),
        same_field_mode="normal",
    )
    assert out[0]["flagged"] is True

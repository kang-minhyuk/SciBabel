from terms.score import TermScoreConfig, score_terms


def _lexicon() -> dict[str, list[str]]:
    return {
        "CSM": ["graph neural network", "distribution shift", "temperature scaling", "sparse regularization"],
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
            "sparse regularization": 2.6,
            "packed": 0.4,
            "bed": 0.2,
            "reactor": 0.2,
        },
        "PM": {
            "graph neural network": -1.2,
            "distribution shift": -0.8,
            "temperature scaling": -0.9,
            "sparse regularization": -0.7,
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


def test_score_rejects_fragment_but_keeps_concepts() -> None:
    extracted = [
        {"term": "optimize a graph", "start": 0, "end": 16, "leading_lemma": "optimize", "leading_pos": "VERB"},
        {"term": "graph neural network", "start": 3, "end": 23, "leading_lemma": "graph", "leading_pos": "NOUN", "noun_headed": True, "noun_adj_compound": True},
        {"term": "sparse regularization", "start": 29, "end": 50, "leading_lemma": "sparse", "leading_pos": "ADJ", "noun_headed": True, "noun_adj_compound": True},
        {"term": "distribution shift", "start": 57, "end": 75, "leading_lemma": "distribution", "leading_pos": "NOUN", "noun_headed": True, "noun_adj_compound": True},
    ]
    out = score_terms(
        extracted_terms=extracted,
        src="CSM",
        tgt="PM",
        all_domains=["CSM", "PM", "CHEM", "CHEME"],
        term_stats=_stats(),
        lexicon_by_domain=_lexicon(),
        cfg=TermScoreConfig(src_threshold=0.35, tgt_threshold=0.45),
    )
    by_term = {str(r["term"]).lower(): r for r in out}
    assert by_term["optimize a graph"]["flagged"] is False
    assert by_term["graph neural network"]["flagged"] is True
    assert by_term["sparse regularization"]["flagged"] is True
    assert by_term["distribution shift"]["flagged"] is True

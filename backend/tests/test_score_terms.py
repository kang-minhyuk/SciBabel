from terms.score import TermScoreConfig, score_terms


def test_score_flags_unfamiliar_term() -> None:
    terms = [{"term": "graph Laplacian", "start": 0, "end": 15, "source": "lexicon"}]
    stats = {
        ("CSM", "graph laplacian"): 3.2,
        ("PM", "graph laplacian"): -1.2,
    }
    lex = {"CSM": ["graph laplacian"], "PM": [], "CCE": []}

    scored = score_terms(
        extracted_terms=terms,
        src="CSM",
        tgt="PM",
        all_domains=["CSM", "PM", "CCE"],
        term_stats=stats,
        lexicon_by_domain=lex,
        cfg=TermScoreConfig(src_threshold=0.3, tgt_threshold=0.45),
    )

    assert len(scored) == 1
    assert scored[0]["flagged"] is True
    assert "low_tgt_familiarity" in str(scored[0]["reason"])


def test_score_not_flagged_if_familiar_in_target() -> None:
    terms = [{"term": "band gap", "start": 0, "end": 8, "source": "lexicon"}]
    stats = {
        ("PM", "band gap"): 2.0,
        ("CSM", "band gap"): 0.4,
    }
    lex = {"CSM": ["band gap"], "PM": ["band gap"], "CCE": []}

    scored = score_terms(
        extracted_terms=terms,
        src="CSM",
        tgt="PM",
        all_domains=["CSM", "PM", "CCE"],
        term_stats=stats,
        lexicon_by_domain=lex,
        cfg=TermScoreConfig(src_threshold=0.3, tgt_threshold=0.45),
    )
    assert scored[0]["flagged"] is False

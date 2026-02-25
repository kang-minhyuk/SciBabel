import spacy

from terms.extract import extract_terms, set_nlp


def test_extract_no_stopword_only_terms() -> None:
    text = "We evaluate the method while under strict constraints in the system."
    items = extract_terms(text, max_terms=12)
    terms = {str(x["term"]).lower() for x in items}
    assert "while" not in terms
    assert "under" not in terms


def test_extract_se3_equivariant_detected() -> None:
    text = "A SE(3)-equivariant architecture improves geometric robustness."
    items = extract_terms(text, max_terms=12)
    terms = " | ".join(str(x["term"]).lower() for x in items)
    assert "se(3)-equivariant" in terms or "se(3)" in terms


def test_extract_k_space_detected() -> None:
    text = "We optimize k-space sampling trajectories for MRI reconstruction."
    items = extract_terms(text, max_terms=12)
    terms = " | ".join(str(x["term"]).lower() for x in items)
    assert "k-space" in terms


def test_overlap_dedup_prefers_longer_phrase() -> None:
    text = "Sparse attention improves long-range dependencies."
    items = extract_terms(text, max_terms=12)
    terms = [str(x["term"]).lower() for x in items]
    # should prefer phrase-level item rather than trivial overlap token
    assert any("long-range dependencies" in t for t in terms)


def test_debug_artifacts_removed() -> None:
    text = "updatedgpt_noself_v2 proposes an approach (native=0.00) with domain-specific concept."
    items = extract_terms(text, max_terms=12)
    terms = " | ".join(str(x["term"]).lower() for x in items)
    assert "updatedgpt" not in terms
    assert "native=" not in terms


def test_acceptance_sentence_has_expected_phrases() -> None:
    text = "Our transformer uses sparse attention to reduce memory while preserving long-range dependencies."
    items = extract_terms(text, max_terms=12)
    terms = [str(x["term"]).lower() for x in items]
    joined = " | ".join(terms)
    assert "transformer" in joined
    assert "sparse attention" in joined
    assert "long-range dependencies" in joined
    assert "reduce" not in terms
    assert "while" not in terms


def test_extract_with_blank_spacy_still_returns_terms() -> None:
    set_nlp(spacy.blank("en"))
    text = "We optimize a graph neural network with sparse regularization under distribution shift."
    items = extract_terms(text, max_terms=12)
    terms = " | ".join(str(x["term"]).lower() for x in items)
    assert "graph neural" in terms or "neural network" in terms or "distribution shift" in terms


def test_extract_when_spacy_unavailable_uses_fallback(monkeypatch) -> None:
    import terms.extract as ex

    set_nlp(None)

    def _boom():
        raise RuntimeError("spacy unavailable")

    monkeypatch.setattr(ex, "_get_nlp", _boom)
    text = "We optimize a graph neural network with sparse regularization under distribution shift."
    items = extract_terms(text, max_terms=12)
    terms = " | ".join(str(x["term"]).lower() for x in items)
    assert "graph neural" in terms or "neural network" in terms or "distribution shift" in terms

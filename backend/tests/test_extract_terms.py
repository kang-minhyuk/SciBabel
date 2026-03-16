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
    assert "sparse attention" in joined
    assert "long-range dependencies" in joined
    assert "transformer" not in terms
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


def test_extract_rejects_bad_phrase_fragments() -> None:
    text = (
        "We optimize a graph neural network with sparse regularization under distribution shift. "
        "The transformer uses low-rank attention to reduce memory cost on long sequences. "
        "A Monte Carlo simulation estimates magnetization under an external field. "
        "We derive the partition function by an asymptotic method of the system."
    )
    items = extract_terms(text, max_terms=30)
    terms = {str(x["term"]).lower() for x in items}
    banned = {
        "optimize a graph",
        "memory cost on",
        "by an",
        "of the",
        "the thin",
        "we derive the",
        "on long sequences",
    }
    assert terms.isdisjoint(banned)


def test_extract_keeps_good_concept_phrases() -> None:
    text = (
        "We optimize a graph neural network with sparse regularization under distribution shift. "
        "A Monte Carlo simulation estimates thermal conductivity in solids. "
        "A packed-bed reactor uses model predictive control for stable operation."
    )
    items = extract_terms(text, max_terms=30)
    joined = " | ".join(str(x["term"]).lower() for x in items)
    assert "graph neural network" in joined
    assert "sparse regularization" in joined
    assert "distribution shift" in joined
    assert "monte carlo simulation" in joined
    assert "thermal conductivity" in joined
    assert "packed-bed reactor" in joined
    assert "model predictive control" in joined


def test_fragment_case_1_graph_regularization_distribution_shift() -> None:
    text = "We optimize a graph neural network with sparse regularization under distribution shift."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert "graph neural network" in terms
    assert "sparse regularization" in terms
    assert "distribution shift" in terms
    assert "optimize a graph" not in terms


def test_fragment_case_2_transformer_attention_fragments_removed() -> None:
    text = "The transformer uses low-rank attention to reduce memory cost on long sequences."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert "low-rank attention" in terms
    assert "memory cost on" not in terms
    assert "on long sequences" not in terms
    assert "the transformer" not in terms


def test_fragment_case_3_phase_transition_order_parameter() -> None:
    text = "The phase transition is characterized by an order parameter near criticality."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert "phase transition" in terms
    assert "order parameter" in terms
    assert "criticality" in terms
    assert "by an" not in terms
    assert "transition is characterized" not in terms
    assert "parameter near criticality" not in terms


def test_fragment_case_4_diffusion_model_classifier_free_guidance() -> None:
    text = "We train a diffusion model with classifier-free guidance for molecular generation."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert "diffusion model" in terms
    assert "classifier-free guidance" in terms
    assert "molecular generation" in terms
    assert "train a diffusion" not in terms


def test_extract_canonical_optimize_graph_modifier_variant() -> None:
    text = "We optimize a graph neural network with adaptive sparse regularization under distribution shift."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert {"graph neural network", "sparse regularization", "distribution shift"}.issubset(terms)
    assert "optimize a graph" not in terms


def test_extract_canonical_phase_transition_around_criticality() -> None:
    text = "The phase transition is characterized by an order parameter around criticality."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert {"phase transition", "order parameter", "criticality"}.issubset(terms)
    banned = {"by an", "transition is characterized", "parameter around criticality"}
    assert terms.isdisjoint(banned)


def test_extract_canonical_diffusion_robust_guidance_stable_generation() -> None:
    text = "We train a diffusion model with robust classifier-free guidance for stable molecular generation."
    items = extract_terms(text, max_terms=20)
    terms = {str(x["term"]).lower() for x in items}
    assert {"diffusion model", "classifier-free guidance", "molecular generation"}.issubset(terms)
    assert "train a diffusion" not in terms

from terms.canonicalize import canonicalize_span, canonicalize_term


def test_light_verb_trimming() -> None:
    out = set(canonicalize_span("apply classifier-free guidance"))
    assert "classifier-free guidance" in out


def test_modifier_trimming() -> None:
    out = set(canonicalize_span("adaptive sparse regularization"))
    assert "sparse regularization" in out


def test_modifier_trimming_sparse_variants() -> None:
    assert canonicalize_term("robust sparse regularization") == "sparse regularization"
    assert canonicalize_term("stronger sparse regularization") == "sparse regularization"


def test_preposition_cleanup() -> None:
    out = set(canonicalize_span("toward molecular generation"))
    assert "molecular generation" in out


def test_classifier_and_generation_modifier_normalization() -> None:
    assert canonicalize_term("robust classifier-free guidance") == "classifier-free guidance"
    assert canonicalize_term("stable molecular generation") == "molecular generation"


def test_concept_decomposition_parameter_criticality() -> None:
    out = set(canonicalize_span("parameter around criticality"))
    assert "order parameter" in out
    assert "criticality" in out


def test_criticality_mapping_family() -> None:
    out = set(canonicalize_span("criticality studies"))
    assert "criticality" in out


def test_criticality_family_mappings() -> None:
    assert canonicalize_term("criticality analysis") == "criticality"
    assert canonicalize_term("criticality evidence") == "criticality"
    assert canonicalize_term("criticality constraints") == "criticality"


def test_generic_verb_prefix_cases() -> None:
    assert canonicalize_term("calibrate sparse regularization") == "sparse regularization"
    assert canonicalize_term("apply classifier-free guidance") == "classifier-free guidance"
    assert canonicalize_term("optimize a graph neural network") == "graph neural network"
    assert canonicalize_term("train a diffusion model") == "diffusion model"

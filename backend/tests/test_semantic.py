from semantic import semantic_similarity


def test_semantic_similarity_bounds() -> None:
    sim = semantic_similarity("graph neural networks for molecules", "graph neural networks for molecules")
    assert 0.0 <= sim <= 1.0


def test_semantic_similarity_prefers_related_text() -> None:
    a = "sparse graph regularization for distribution shift"
    b = "graph-based regularization improves robustness under shift"
    c = "organic synthesis temperature profile for catalyst activation"

    sim_ab = semantic_similarity(a, b)
    sim_ac = semantic_similarity(a, c)

    assert sim_ab >= sim_ac


def run_checks() -> None:
    test_semantic_similarity_bounds()
    test_semantic_similarity_prefers_related_text()
    print("semantic checks passed")


if __name__ == "__main__":
    run_checks()

from terms.analog import AnalogSuggester


def test_analog_top5_and_threshold() -> None:
    suggester = AnalogSuggester(analog_sim_threshold=0.15)
    pool = [
        "energy barrier",
        "diffusion coefficient",
        "reaction coordinate",
        "free energy landscape",
        "graph Laplacian",
        "band structure",
        "method",  # should be removed
    ]
    out = suggester.suggest("energy landscape", target_candidates=pool, top_k=5)
    assert len(out) <= 5
    assert all(float(x["score"]) >= 0.15 for x in out)
    assert all(str(x["candidate"]).lower() != "method" for x in out)


def test_analog_returns_empty_when_candidates_are_generic() -> None:
    suggester = AnalogSuggester(analog_sim_threshold=0.55)
    pool = ["at low temperature", "pattern of the", "by an", "of the"]
    out = suggester.suggest("temperature scaling", target_candidates=pool, top_k=5)
    assert out == []


def test_transformer_does_not_return_article_phrases() -> None:
    suggester = AnalogSuggester(analog_sim_threshold=0.55)
    pool = ["the model", "a method", "of the", "pattern of the"]
    out = suggester.suggest("The transformer", target_candidates=pool, top_k=5)
    assert out == []

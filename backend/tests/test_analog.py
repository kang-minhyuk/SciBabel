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

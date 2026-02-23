from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from term_strategy import TermStrategyEngine, build_term_instruction_block, strategy_penalty


def run_checks() -> None:
    lex = {
        "CSM": ["loss function", "regularization", "graph neural network", "hilbert space"],
        "PM": ["energy functional", "band gap", "diffusion coefficient", "energy barrier"],
        "CCE": ["activation energy", "reaction rate", "mass transport"],
    }

    engine = TermStrategyEngine(
        lexicon_by_domain=lex,
        aliases_path=Path(__file__).resolve().parents[1] / "term_aliases.json",
        term_log_odds={("PM", "band gap"): 2.0},
    )

    text = "We optimize a loss function with regularization and graph neural network layers."
    terms = engine.extract_key_terms(text, max_terms=6)
    assert len(terms) > 0

    s1 = engine.classify_term("loss function", "PM")
    assert s1.type in {"analogous", "equivalent"}

    s2 = engine.classify_term("band gap", "PM")
    assert s2.type == "equivalent"

    s3 = engine.classify_term("Hilbert space", "CCE")
    assert s3.type in {"intranslatable", "unique", "analogous"}

    block = build_term_instruction_block([s1, s2, s3])
    assert "Term handling rules" in block

    p = strategy_penalty("Use energy functional with constraints.", [s1, s2, s3])
    assert 0.0 <= p <= 0.35

    print("term_strategy checks passed")


if __name__ == "__main__":
    run_checks()

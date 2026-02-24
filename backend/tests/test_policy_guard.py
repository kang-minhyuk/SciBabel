import pytest

from llm.openai_client import OpenAIExplainClient, ExplainPolicyError


def test_policy_accepts_analogy_language() -> None:
    payload = {
        "short_explanation": "This is analogous to diffusion in PM.",
        "long_explanation": "It is conceptually similar under the provided context.",
    }
    OpenAIExplainClient._validate_policy(payload)


def test_policy_rejects_equivalence_phrase() -> None:
    payload = {
        "short_explanation": "This is equivalent to diffusion.",
        "long_explanation": "It is analogous in a loose sense.",
    }
    with pytest.raises(ExplainPolicyError):
        OpenAIExplainClient._validate_policy(payload)


def test_policy_requires_analogy_marker() -> None:
    payload = {
        "short_explanation": "This is related.",
        "long_explanation": "It may appear in target field.",
    }
    with pytest.raises(ExplainPolicyError):
        OpenAIExplainClient._validate_policy(payload)

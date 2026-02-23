from __future__ import annotations

from typing import Dict

DOMAIN_NAMES: Dict[str, str] = {
    "CSM": "Computer Science / Mathematics",
    "PM": "Physics / Materials Science",
    "CCE": "Chemistry / Chemical Engineering",
}

PROMPT_TEMPLATES: Dict[str, str] = {
    "literal": (
        "You are a scientific translator. Rewrite the source text from {src_name} to {tgt_name}. "
        "Preserve meaning, keep technical precision, and output only the translated text.\n\n"
        "Source text:\n{text}\n"
    ),
    "domain_steered": (
        "Translate this scientific text from {src_name} to {tgt_name}. Prioritize target-community terminology, "
        "problem framing, and units/quantities common in {tgt_name}. Do not invent claims. "
        "Output only the translated text.\n\n"
        "Source text:\n{text}\n"
    ),
    "process_framing": (
        "Translate from {src_name} into a {tgt_name} process-oriented framing. Emphasize descriptors, constraints, "
        "screening/optimization language, and practical interpretation. Avoid hallucinated claims; phrase uncertain "
        "statements as 'enables' or 'can be used for'. Output only the translated text.\n\n"
        "Source text:\n{text}\n"
    ),
    "educator": (
        "Perform cross-domain scientific translation from {src_name} to {tgt_name}. Keep the original semantics, "
        "reframe for clarity to {tgt_name} readers, and preserve non-native terms with a short explanation when needed. "
        "Output only one translation.\n\n"
        "Source text:\n{text}\n"
    ),
}


def get_prompt_action_names() -> list[str]:
    return list(PROMPT_TEMPLATES.keys())


def build_prompt(
    action: str,
    text: str,
    src: str,
    tgt: str,
    term_instructions: str = "",
    target_lexicon_hints: list[str] | None = None,
) -> str:
    if action not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt action: {action}")

    src_name = DOMAIN_NAMES.get(src, src)
    tgt_name = DOMAIN_NAMES.get(tgt, tgt)
    prompt = PROMPT_TEMPLATES[action].format(src_name=src_name, tgt_name=tgt_name, text=text)
    if target_lexicon_hints:
        hints = ", ".join(target_lexicon_hints[:20])
        prompt = f"{prompt}\nTarget lexicon hints: {hints}\n"
    if term_instructions.strip():
        prompt = f"{prompt}\n\n{term_instructions.strip()}\n"
    return prompt

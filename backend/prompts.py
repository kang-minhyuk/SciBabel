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
    "term_focused": (
        "Translate this scientific text from {src_name} to {tgt_name}. Prioritize target-domain terminology "
        "without changing core claims. Keep concise and accurate. Output only the translation.\n\n"
        "Source text:\n{text}\n"
    ),
    "explanatory": (
        "Perform cross-domain scientific translation from {src_name} to {tgt_name}. Keep the original semantics, "
        "but reframe wording so experts in {tgt_name} find it natural. Output only one translation.\n\n"
        "Source text:\n{text}\n"
    ),
}


def get_prompt_action_names() -> list[str]:
    return list(PROMPT_TEMPLATES.keys())


def build_prompt(action: str, text: str, src: str, tgt: str, term_instructions: str = "") -> str:
    if action not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt action: {action}")

    src_name = DOMAIN_NAMES.get(src, src)
    tgt_name = DOMAIN_NAMES.get(tgt, tgt)
    prompt = PROMPT_TEMPLATES[action].format(src_name=src_name, tgt_name=tgt_name, text=text)
    if term_instructions.strip():
        prompt = f"{prompt}\n\n{term_instructions.strip()}\n"
    return prompt

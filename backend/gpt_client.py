from __future__ import annotations

import os

import requests

OPENAI_API_URL = "https://api.openai.com/v1/responses"


def generate_with_gpt(prompt: str, temperature: float, top_p: float) -> str:
    """Generate text with OpenAI Responses API.

    Uses model from `OPENAI_MODEL` (default: `gpt-4.1-mini`).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise NotImplementedError(
            "OPENAI_API_KEY is not configured. Set it in your .env to enable GPT generation."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"

    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENAI_API_URL, json=payload, headers=headers, timeout=45)
    response.raise_for_status()
    data = response.json()

    # Preferred field when available.
    output_text = data.get("output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback for structured response payload.
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text", "")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())

    return "\n".join(chunks).strip()

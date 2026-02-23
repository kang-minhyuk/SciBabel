from __future__ import annotations

import os

import requests

GEMINI_MODEL = "gemini-1.5-flash"


def generate_with_gemini(prompt: str, temperature: float, top_p: float) -> str:
    """Generate text with Gemini via REST API.

    TODO: If SDK-based integration is preferred in your environment, replace this
    function with `google-generativeai` client usage.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise NotImplementedError(
            "GEMINI_API_KEY is not configured. Set it in your .env to enable translation."
        )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": 512,
        },
    }

    response = requests.post(url, json=payload, timeout=45)
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(tp.strip() for tp in text_parts if tp.strip()).strip()

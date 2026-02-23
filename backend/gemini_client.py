from __future__ import annotations

import os

import requests

DEFAULT_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


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

    configured_model = os.getenv("GEMINI_MODEL", "").strip()
    models = [configured_model] if configured_model else DEFAULT_GEMINI_MODELS

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": 512,
        },
    }

    data: dict = {}
    last_error: Exception | None = None
    for model in models:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        response = requests.post(url, json=payload, timeout=45)
        if response.status_code == 404 and not configured_model:
            # Try next default model if endpoint/model combo is unavailable.
            continue
        try:
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:
            last_error = exc
            if configured_model:
                raise

    if not data and last_error is not None:
        raise last_error

    candidates = data.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(tp.strip() for tp in text_parts if tp.strip()).strip()

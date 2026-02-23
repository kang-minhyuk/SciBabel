from __future__ import annotations

import os

import requests

DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
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
    models = [configured_model] + DEFAULT_GEMINI_MODELS if configured_model else DEFAULT_GEMINI_MODELS
    # De-duplicate while preserving order.
    models = list(dict.fromkeys(models))

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
    for idx, model in enumerate(models):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        response = requests.post(url, json=payload, timeout=45)
        status_code = response.status_code
        try:
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:
            last_error = exc
            is_last = idx == (len(models) - 1)
            # Auth errors should fail immediately.
            if status_code in (401, 403):
                raise
            # Try next model on model-not-found or throttling.
            if status_code in (404, 429) and not is_last:
                continue
            # For other transient/provider errors, try next model if available.
            if not is_last:
                continue
            raise

    if not data and last_error is not None:
        raise last_error

    candidates = data.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if "text" in p]
    return "\n".join(tp.strip() for tp in text_parts if tp.strip()).strip()

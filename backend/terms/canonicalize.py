from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/()]*")
WS_RE = re.compile(r"\s+")

LIGHT_VERBS = {
    "apply",
    "applying",
    "use",
    "using",
    "used",
    "handle",
    "handling",
    "calibrate",
    "calibrating",
    "optimize",
    "optimizing",
    "study",
    "studying",
    "evaluate",
    "evaluating",
    "estimate",
    "estimating",
    "characterize",
    "characterizing",
    "perform",
    "performing",
}

LEADING_ATTACH = {
    "through",
    "toward",
    "towards",
    "across",
    "under",
    "for",
    "in",
    "over",
    "via",
}

ARTICLES = {"a", "an", "the"}
GENERIC_PREFIX_MODIFIERS = {
    "robust",
    "adaptive",
    "stronger",
    "stable",
    "novel",
    "effective",
    "efficient",
}

EXACT_MAP = {
    "robust sparse regularization": "sparse regularization",
    "adaptive sparse regularization": "sparse regularization",
    "stronger sparse regularization": "sparse regularization",
    "calibrate sparse regularization": "sparse regularization",
    "apply classifier-free guidance": "classifier-free guidance",
    "robust classifier-free guidance": "classifier-free guidance",
    "stable molecular generation": "molecular generation",
    "de novo molecular generation": "molecular generation",
    "toward molecular generation": "molecular generation",
    "criticality analysis": "criticality",
    "criticality evidence": "criticality",
    "criticality studies": "criticality",
    "criticality constraints": "criticality",
    "apply classifier-free guidance": "classifier-free guidance",
    "handle distribution shift": "distribution shift",
    "calibrate sparse regularization": "sparse regularization",
}

HIGH_VALUE_CONCEPTS = [
    "graph neural network",
    "sparse regularization",
    "distribution shift",
    "low-rank attention",
    "memory cost",
    "long sequences",
    "phase transition",
    "order parameter",
    "criticality",
    "critical point",
    "critical regime",
    "diffusion model",
    "classifier-free guidance",
    "molecular generation",
]


def normalize_phrase_text(text: str) -> str:
    t = WS_RE.sub(" ", text.strip().lower())
    return t.strip(".,;:!?\"'`()[]{}")


def normalize_text(text: str) -> str:
    # Backward-compatible alias used by existing call sites.
    return normalize_phrase_text(text)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens).strip()


def _trim_light_verb(tokens: list[str]) -> list[str]:
    if len(tokens) < 3:
        return tokens
    out = list(tokens)
    if out and out[0] in LIGHT_VERBS:
        out = out[1:]
        if out and out[0] in ARTICLES:
            out = out[1:]
    return out


def _trim_leading_attach(tokens: list[str]) -> list[str]:
    if len(tokens) < 2:
        return tokens
    out = list(tokens)
    while out and (out[0] in LEADING_ATTACH or out[0] in ARTICLES):
        out = out[1:]
    return out


def _trim_prefix_modifiers(tokens: list[str]) -> list[str]:
    out = list(tokens)
    if len(out) >= 3 and out[0] in GENERIC_PREFIX_MODIFIERS:
        out = out[1:]
    if len(out) >= 4 and out[0] == "de" and out[1] == "novo":
        out = out[2:]
    return out


def _map_criticality_suffix(text: str) -> str | None:
    if re.search(r"\bcriticality\s+(analysis|evidence|studies|constraints)\b", text):
        return "criticality"
    return None


def _map_pattern_families(text: str) -> list[str]:
    out: list[str] = []
    if re.search(r"\b(calibrate|robust|adaptive|stronger)?\s*sparse\s+regularization\b", text):
        out.append("sparse regularization")
    if "regularization across distribution" in text:
        out.append("sparse regularization")
    if re.search(r"\bdistribution\s+shift\b", text) or "regularization across distribution" in text:
        out.append("distribution shift")
    if re.search(r"\b(apply|robust)?\s*classifier-free\s+guidance\b", text):
        out.append("classifier-free guidance")
    if re.search(r"\b(de\s+novo\s+)?molecular\s+generation\b", text) or "toward molecular generation" in text:
        out.append("molecular generation")
    if re.search(r"\bphase\s+transition\s+is\b", text):
        out.append("phase transition")
    return out


def _decompose(text: str) -> list[str]:
    out: list[str] = []

    if re.search(r"\border\s+parameter\b", text):
        out.append("order parameter")
    if re.search(r"\bparameter\s+(near|around|at|close|under|with|for)\s+(criticality|critical point|critical regime)\b", text):
        out.append("order parameter")

    m = re.search(r"\b(criticality|critical point|critical regime)\b", text)
    if m:
        out.append(m.group(1))

    if "classifier-free guidance" in text:
        out.append("classifier-free guidance")
    if "molecular generation" in text:
        out.append("molecular generation")

    if "sparse regularization" in text:
        out.append("sparse regularization")
    if "distribution shift" in text:
        out.append("distribution shift")

    # Prefer explicit high-value concepts embedded in larger spans.
    for concept in HIGH_VALUE_CONCEPTS:
        if concept in text:
            out.append(concept)

    dedup: list[str] = []
    seen: set[str] = set()
    for c in out:
        c_norm = normalize_text(c)
        if not c_norm or c_norm in seen:
            continue
        seen.add(c_norm)
        dedup.append(c_norm)
    return dedup


def canonicalize_span(surface_term: str) -> list[str]:
    surface = normalize_text(surface_term)
    if not surface:
        return []

    candidates: list[str] = [surface]

    mapped = EXACT_MAP.get(surface)
    if mapped:
        candidates.append(mapped)

    mapped_crit = _map_criticality_suffix(surface)
    if mapped_crit:
        candidates.append(mapped_crit)

    candidates.extend(_map_pattern_families(surface))

    toks = _tokens(surface)
    if toks:
        step1 = _trim_light_verb(toks)
        step2 = _trim_leading_attach(step1)
        step3 = _trim_prefix_modifiers(step2)

        for step in [step1, step2, step3]:
            s = _join_tokens(step)
            if s:
                candidates.append(s)
                mapped_step = EXACT_MAP.get(s)
                if mapped_step:
                    candidates.append(mapped_step)

    candidates.extend(_decompose(surface))

    dedup: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c_norm = normalize_text(c)
        if not c_norm or c_norm in seen:
            continue
        seen.add(c_norm)
        dedup.append(c_norm)

    high_value_hits = [c for c in dedup if c in HIGH_VALUE_CONCEPTS]
    if high_value_hits:
        return high_value_hits
    return dedup


def canonicalize_term(term: str) -> str:
    vals = canonicalize_span(term)
    if not vals:
        return normalize_phrase_text(term)
    return vals[0]

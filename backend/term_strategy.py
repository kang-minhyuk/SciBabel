from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]*")
FORMAL_KEYWORDS = {
    "sigma-algebra",
    "lebesgue",
    "hilbert",
    "banach",
    "measure theory",
    "sobolev",
    "riemann",
    "stokes theorem",
    "lagrangian density",
}
ANCHOR_TOKENS = {
    "energy",
    "diffusion",
    "entropy",
    "rate",
    "stability",
    "transport",
    "barrier",
    "functional",
    "reaction",
    "graph",
    "constraint",
}

ACRONYM_VARIANTS = {
    "dft": "density functional theory",
    "density functional theory": "density functional theory",
    "pde": "partial differential equation",
    "partial differential equation": "partial differential equation",
    "ode": "ordinary differential equation",
    "ordinary differential equation": "ordinary differential equation",
    "gnn": "graph neural network",
    "graph neural network": "graph neural network",
    "ml": "machine learning",
    "machine learning": "machine learning",
}


@dataclass
class TermStrategy:
    term: str
    type: str
    native_score: float
    neighbor: str | None
    reason: str


@dataclass
class KeyTerm:
    term: str
    span: tuple[int, int]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def normalize_term(term: str) -> str:
    t = " ".join(term.lower().strip().split())
    t = re.sub(r"[^a-z0-9\-_/ ]+", "", t)
    return ACRONYM_VARIANTS.get(t, t)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TermStrategyEngine:
    def __init__(
        self,
        lexicon_by_domain: dict[str, list[str]],
        aliases_path: str | Path,
        term_log_odds: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.lexicon_by_domain = lexicon_by_domain
        self.lexicon_norm = {
            d: {normalize_term(t) for t in terms}
            for d, terms in lexicon_by_domain.items()
        }
        self.union_lexicon = set().union(*self.lexicon_norm.values()) if self.lexicon_norm else set()
        self.term_log_odds = term_log_odds or {}

        path = Path(aliases_path)
        self.aliases: dict[str, dict[str, str]] = {}
        if path.exists():
            self.aliases = json.loads(path.read_text(encoding="utf-8"))
        self.aliases = {normalize_term(k): v for k, v in self.aliases.items()}

    def extract_key_terms(self, text: str, max_terms: int = 10) -> list[KeyTerm]:
        matches = list(TOKEN_RE.finditer(text))
        if not matches:
            return []

        # Build n-grams from token matches (1-3 grams).
        candidates: dict[str, tuple[float, tuple[int, int]]] = {}
        n_tokens = len(matches)
        for i in range(n_tokens):
            for n in (1, 2, 3):
                j = i + n - 1
                if j >= n_tokens:
                    continue
                start = matches[i].start()
                end = matches[j].end()
                raw = text[start:end]
                norm = normalize_term(raw)
                if len(norm) < 3:
                    continue

                toks = token_set(norm)
                if not toks:
                    continue
                in_union = 1.0 if norm in self.union_lexicon else 0.0
                score = (3.0 * in_union) + (0.5 * n) + (0.1 * len(norm))
                prev = candidates.get(norm)
                if prev is None or score > prev[0]:
                    candidates[norm] = (score, (start, end))

        ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)[:max_terms]
        return [KeyTerm(term=t, span=s[1]) for t, s in ranked]

    def term_native_score(self, term: str, tgt: str) -> float:
        norm = normalize_term(term)
        tgt_lex = self.lexicon_norm.get(tgt, set())

        if norm in tgt_lex:
            return 1.0

        canonical = ACRONYM_VARIANTS.get(norm)
        if canonical and canonical in tgt_lex:
            return 0.8

        log_odds = self.term_log_odds.get((tgt, norm))
        if log_odds is not None:
            return float(max(0.0, min(1.0, _sigmoid(log_odds))))

        # weak lexical hint
        overlap = [t for t in token_set(norm) if t in {tok for x in tgt_lex for tok in x.split()}]
        if overlap:
            return 0.25

        return 0.0

    def find_concept_neighbor(self, term: str, tgt: str) -> tuple[str | None, float]:
        norm = normalize_term(term)

        alias_hit = self.aliases.get(norm, {}).get(tgt)
        if alias_hit:
            return alias_hit, 0.95

        term_tokens = token_set(norm)
        if not term_tokens:
            return None, 0.0

        tgt_terms = self.lexicon_norm.get(tgt, set())
        best_term = None
        best_score = 0.0
        for cand in tgt_terms:
            cand_tokens = token_set(cand)
            if not cand_tokens:
                continue
            jacc = len(term_tokens & cand_tokens) / len(term_tokens | cand_tokens)
            anchor_bonus = 0.15 if (term_tokens & cand_tokens & ANCHOR_TOKENS) else 0.0
            score = jacc + anchor_bonus
            if score > best_score:
                best_score = score
                best_term = cand

        if best_term is None or best_score < 0.25:
            return None, 0.0
        return best_term, float(min(1.0, best_score))

    def classify_term(self, term: str, tgt: str) -> TermStrategy:
        norm = normalize_term(term)
        native_score = self.term_native_score(norm, tgt)

        if native_score >= 0.6:
            return TermStrategy(
                term=term,
                type="equivalent",
                native_score=native_score,
                neighbor=None,
                reason="native or near-native in target domain",
            )

        neighbor, neighbor_conf = self.find_concept_neighbor(norm, tgt)
        if neighbor and neighbor_conf >= 0.35:
            return TermStrategy(
                term=term,
                type="analogous",
                native_score=native_score,
                neighbor=neighbor,
                reason=f"concept neighbor found (confidence={neighbor_conf:.2f})",
            )

        symbol_ratio = sum(1 for c in term if not c.isalnum() and not c.isspace()) / max(1, len(term))
        formal = any(k in norm for k in FORMAL_KEYWORDS)
        if formal or symbol_ratio > 0.2:
            return TermStrategy(
                term=term,
                type="intranslatable",
                native_score=native_score,
                neighbor=None,
                reason="formal/domain-specific concept; preserve without forced analogy",
            )

        return TermStrategy(
            term=term,
            type="unique",
            native_score=native_score,
            neighbor=None,
            reason="not native and no strong analog detected",
        )


def build_term_instruction_block(strategies: list[TermStrategy], max_terms: int = 8) -> str:
    if not strategies:
        return ""

    lines = [
        "Term handling rules:",
        "- equivalent: translate normally using target-domain wording",
        "- analogous: prefer mapped neighbor term",
        "- unique: preserve original term and add short parenthetical explanation",
        "- intranslatable: preserve term and append '(domain-specific concept)'",
        "",
        "Apply these for key terms:",
    ]

    for s in strategies[:max_terms]:
        neighbor = f" | neighbor={s.neighbor}" if s.neighbor else ""
        lines.append(f"- {s.term} => {s.type} (native={s.native_score:.2f}{neighbor})")

    return "\n".join(lines)


def strategy_penalty(candidate: str, strategies: list[TermStrategy]) -> float:
    cand_lower = candidate.lower()
    penalty = 0.0

    for s in strategies:
        term_present = s.term.lower() in cand_lower
        if s.type == "unique":
            if not term_present:
                penalty += 0.08
        elif s.type == "analogous":
            neighbor_present = bool(s.neighbor and s.neighbor.lower() in cand_lower)
            if not neighbor_present:
                penalty += 0.04
        elif s.type == "intranslatable":
            if not term_present:
                penalty += 0.1

    return float(min(0.35, penalty))

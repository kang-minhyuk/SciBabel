from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib

from terms.analog import AnalogSuggester
from terms.extract import extract_terms_profiled
from terms.score import TermScoreConfig, score_terms

Domain = Literal["CSM", "PM", "CHEM", "CHEME", "CCE"]


class AnnotationArtifactsMissing(RuntimeError):
    pass


class TermAnnotationEngine:
    def __init__(
        self,
        root: Path,
        src_threshold: float = 0.35,
        tgt_threshold: float = 0.45,
        analog_threshold: float = 0.2,
    ) -> None:
        self.root = root
        self.lexicon_path = root / "data" / "processed" / "domain_lexicon.json"
        self.term_stats_path = root / "data" / "processed" / "term_stats.csv"
        self.model_path = root / "models" / "domain_clf.joblib"
        self.evidence_index_path = root / "data" / "processed" / "evidence_index.json"

        default_env = "production" if os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes", "on"} else "dev"
        env = os.getenv("SCIBABEL_ENV", default_env).strip().lower()
        self.is_production = env == "production"
        default_evidence = "false" if self.is_production else "true"
        self.evidence_enabled = os.getenv("EVIDENCE_ENABLED", default_evidence).strip().lower() in {"1", "true", "yes", "on"}
        self.yake_enabled = os.getenv("YAKE_ENABLED", "false" if self.is_production else "true").strip().lower() in {"1", "true", "yes", "on"}
        self.analog_max_candidates = max(50, int(os.getenv("ANALOG_MAX_CANDIDATES", "300")))
        self.analog_max_terms = max(1, int(os.getenv("ANALOG_MAX_TERMS", "6")))

        missing = [
            p
            for p in [self.lexicon_path, self.term_stats_path, self.model_path]
            if not p.exists()
        ]
        if missing:
            raise AnnotationArtifactsMissing(
                "Missing required artifacts: "
                + ", ".join(str(p) for p in missing)
                + ". Run make textmining-all to generate local mining assets."
            )

        self.lexicon_by_domain, self.style_lexicon_by_domain = self._load_lexicon(self.lexicon_path)
        self.lexicon_lower_by_domain = {
            d: {t.lower() for t in terms}
            for d, terms in self.lexicon_by_domain.items()
        }
        self.all_phrases = sorted({p for arr in self.lexicon_by_domain.values() for p in arr}, key=len, reverse=True)
        self.term_stats = self._load_term_stats(self.term_stats_path)
        self.clf = joblib.load(self.model_path)
        self.scoring_cfg = TermScoreConfig(src_threshold=src_threshold, tgt_threshold=tgt_threshold)
        self.analog = AnalogSuggester(analog_sim_threshold=analog_threshold)
        self.domains = sorted(self.lexicon_by_domain.keys())
        self.evidence_index = self._load_evidence_index() if self.evidence_enabled else {}

    @staticmethod
    def _load_lexicon(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        style_out: dict[str, list[str]] = {}
        preferred = ["CSM", "PM", "CHEM", "CHEME", "CCE"]
        for d in preferred:
            node = raw.get(d, {})
            merged: list[str] = []
            style_terms: list[str] = []
            if isinstance(node, list):
                merged = [str(x) for x in node]
            elif isinstance(node, dict):
                style_terms = [str(x) for x in node.get("style", [])]
                merged = (
                    [str(x) for x in node.get("bigrams", [])]
                    + [str(x) for x in node.get("trigrams", [])]
                    + style_terms
                    + [str(x) for x in node.get("top_bigrams", [])]
                    + [str(x) for x in node.get("top_trigrams", [])]
                    + [str(x) for x in node.get("top_terms", [])]
                )
            deduped: list[str] = []
            seen: set[str] = set()
            for t in merged:
                tl = t.strip().lower()
                if not tl or tl in seen:
                    continue
                seen.add(tl)
                deduped.append(t.strip())
            out[d] = deduped
            style_deduped: list[str] = []
            style_seen: set[str] = set()
            for t in style_terms:
                tl = t.strip().lower()
                if not tl or tl in style_seen:
                    continue
                style_seen.add(tl)
                style_deduped.append(t.strip())
            style_out[d] = style_deduped if style_deduped else deduped
        return out, style_out

    @staticmethod
    def _load_term_stats(path: Path) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = str(row.get("domain", "")).strip().upper()
                t = str(row.get("term", "")).strip().lower()
                if not d or not t:
                    continue
                z_raw = row.get("z", row.get("log_odds", "0"))
                try:
                    z = float(z_raw or 0.0)
                except ValueError:
                    z = 0.0
                if d not in stats:
                    stats[d] = {}
                stats[d][t] = z
        return stats

    def _load_evidence_index(self) -> dict[str, list[dict[str, str]]]:
        if not self.evidence_index_path.exists():
            return {}
        try:
            raw = json.loads(self.evidence_index_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            out: dict[str, list[dict[str, str]]] = {}
            for key, val in raw.items():
                if not isinstance(val, list):
                    continue
                cleaned_rows: list[dict[str, str]] = []
                for row in val:
                    if not isinstance(row, dict):
                        continue
                    cleaned_rows.append(
                        {
                            "snippet": str(row.get("snippet", "")),
                            "doc_id": str(row.get("doc_id", "")),
                            "source": str(row.get("source", "")),
                        }
                    )
                out[str(key).strip().lower()] = cleaned_rows
            return out
        except Exception:
            return {}

    def _evidence_lookup(self, tgt: str, phrase: str, max_hits: int = 2) -> list[dict[str, str]]:
        if not self.evidence_enabled:
            return []
        key = f"{tgt.upper()}::{phrase.strip().lower()}"
        rows = self.evidence_index.get(key, [])
        if not rows:
            return []
        return rows[:max(0, int(max_hits))]

    def predict_src(self, text: str) -> tuple[str | None, float | None]:
        labels = list(getattr(self.clf, "classes_", []))
        if not labels:
            return None, None
        probs = self.clf.predict_proba([text])[0]
        idx = int(probs.argmax())
        return str(labels[idx]), float(probs[idx])

    def annotate(self, text: str, src: str, tgt: Domain, max_terms: int = 8) -> dict[str, object]:
        t_all = time.perf_counter()

        predicted_src, pred_conf = self.predict_src(text)
        src_final = predicted_src if src == "auto" and predicted_src else src
        src_final = src_final if src_final in set(self.domains) else "CSM"

        t0 = time.perf_counter()
        try:
            extracted_pack = extract_terms_profiled(
                text=text,
                max_terms=max(16, max_terms * 3),
                yake_enabled=self.yake_enabled,
            )
            extracted = extracted_pack.get("terms", []) if isinstance(extracted_pack, dict) else []
            extract_t = extracted_pack.get("timings", {}) if isinstance(extracted_pack, dict) else {}
        except Exception as exc:
            print(f"[annotate] extract_terms_error={exc}")
            extracted = []
            extract_t = {}
        t_extract = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            scored = score_terms(
                extracted_terms=extracted,
                src=src_final,
                tgt=tgt,
                all_domains=self.domains,
                term_stats=self.term_stats,
                lexicon_by_domain=self.lexicon_by_domain,
                lexicon_lower_by_domain=self.lexicon_lower_by_domain,
                cfg=self.scoring_cfg,
            )
        except Exception as exc:
            print(f"[annotate] score_terms_error={exc}")
            scored = []
        t_score = time.perf_counter() - t0

        enriched: list[dict[str, object]] = []
        t_analog_total = 0.0
        t_evidence_total = 0.0
        comparison_budget = self.analog_max_candidates * self.analog_max_terms
        comparisons_used = 0
        analog_terms_limit = min(max_terms, self.analog_max_terms)
        analog_candidates = self.style_lexicon_by_domain.get(tgt, self.lexicon_by_domain.get(tgt, []))
        for row in scored:
            if len(enriched) >= max_terms:
                break
            term = str(row["term"])
            t0 = time.perf_counter()
            try:
                if len(enriched) >= analog_terms_limit or comparisons_used >= comparison_budget:
                    analogs = []
                else:
                    remaining_budget = max(0, comparison_budget - comparisons_used)
                    local_cap = min(self.analog_max_candidates, remaining_budget)
                    analogs = self.analog.suggest(
                        term=term,
                        target_candidates=analog_candidates,
                        top_k=5,
                        max_candidates=local_cap,
                    )
                    comparisons_used += max(0, local_cap)
            except Exception as exc:
                print(f"[annotate] analog_suggest_error term={term!r} err={exc}")
                analogs = []
            t_analog_total += time.perf_counter() - t0

            evidence_term = str(analogs[0]["candidate"]) if analogs else term
            evidence: list[dict[str, str]] = []
            if self.evidence_enabled:
                t0 = time.perf_counter()
                try:
                    evidence = self._evidence_lookup(tgt=tgt, phrase=evidence_term, max_hits=2)
                except Exception as exc:
                    print(f"[annotate] evidence_lookup_error term={evidence_term!r} err={exc}")
                    evidence = []
                t_evidence_total += time.perf_counter() - t0

            row_out = dict(row)
            row_out["term"] = str(row_out.get("term", "")).replace("(native=0.00)", "").replace("(domain-specific concept)", "").strip()
            row_out["analogs"] = analogs
            row_out["evidence"] = evidence
            row_out["explain_available"] = bool(row_out.get("flagged", False))
            enriched.append(row_out)

        src_warning = bool(predicted_src and src != "auto" and predicted_src != src and (pred_conf or 0.0) >= 0.55)

        return {
            "predicted_src": predicted_src,
            "predicted_src_confidence": pred_conf,
            "src_warning": src_warning,
            "src_effective": src_final,
            "_timings": {
                "spacy_extract_sec": float(extract_t.get("spacy_extract_sec", 0.0)),
                "yake_extract_sec": float(extract_t.get("yake_extract_sec", 0.0)),
                "merge_dedupe_sec": float(extract_t.get("merge_dedupe_sec", 0.0)),
                "extract_terms_sec": round(t_extract, 4),
                "score_terms_sec": round(t_score, 4),
                "analog_search_sec": round(t_analog_total, 4),
                "evidence_sec": round(t_evidence_total, 4),
                "total_sec": round(time.perf_counter() - t_all, 4),
                "yake_enabled": self.yake_enabled,
            },
            "terms": enriched,
        }

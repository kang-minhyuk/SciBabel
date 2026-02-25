from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib

from terms.analog import AnalogSuggester
from terms.extract import extract_terms
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

        self.lexicon_by_domain = self._load_lexicon(self.lexicon_path)
        self.all_phrases = sorted({p for arr in self.lexicon_by_domain.values() for p in arr}, key=len, reverse=True)
        self.term_stats = self._load_term_stats(self.term_stats_path)
        self.clf = joblib.load(self.model_path)
        self.scoring_cfg = TermScoreConfig(src_threshold=src_threshold, tgt_threshold=tgt_threshold)
        self.analog = AnalogSuggester(analog_sim_threshold=analog_threshold)
        self.domains = sorted(self.lexicon_by_domain.keys())
        self.evidence_index = self._load_evidence_index() if self.evidence_enabled else {}

    @staticmethod
    def _load_lexicon(path: Path) -> dict[str, list[str]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, list[str]] = {}
        preferred = ["CSM", "PM", "CHEM", "CHEME", "CCE"]
        for d in preferred:
            node = raw.get(d, {})
            merged: list[str] = []
            if isinstance(node, list):
                merged = [str(x) for x in node]
            elif isinstance(node, dict):
                merged = (
                    [str(x) for x in node.get("bigrams", [])]
                    + [str(x) for x in node.get("trigrams", [])]
                    + [str(x) for x in node.get("style", [])]
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
        return out

    @staticmethod
    def _load_term_stats(path: Path) -> dict[tuple[str, str], float]:
        stats: dict[tuple[str, str], float] = {}
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
                stats[(d, t)] = z
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
        extracted = extract_terms(text=text, max_terms=max(16, max_terms * 3))
        t_extract = time.perf_counter() - t0

        t0 = time.perf_counter()
        scored = score_terms(
            extracted_terms=extracted,
            src=src_final,
            tgt=tgt,
            all_domains=self.domains,
            term_stats=self.term_stats,
            lexicon_by_domain=self.lexicon_by_domain,
            cfg=self.scoring_cfg,
        )
        t_score = time.perf_counter() - t0

        enriched: list[dict[str, object]] = []
        t_analog_total = 0.0
        t_evidence_total = 0.0
        for row in scored:
            if len(enriched) >= max_terms:
                break
            term = str(row["term"])
            t0 = time.perf_counter()
            analogs = self.analog.suggest(term=term, target_candidates=self.lexicon_by_domain.get(tgt, []), top_k=5)
            t_analog_total += time.perf_counter() - t0

            evidence_term = str(analogs[0]["candidate"]) if analogs else term
            t0 = time.perf_counter()
            evidence = self._evidence_lookup(tgt=tgt, phrase=evidence_term, max_hits=2)
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
                "extract_terms_sec": round(t_extract, 4),
                "score_terms_sec": round(t_score, 4),
                "analog_search_sec": round(t_analog_total, 4),
                "evidence_sec": round(t_evidence_total, 4),
                "total_sec": round(time.perf_counter() - t_all, 4),
            },
            "terms": enriched,
        }

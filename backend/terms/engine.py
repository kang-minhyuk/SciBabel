from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from terms.analog import AnalogSuggester
from terms.evidence import find_evidence_snippets
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
        self.corpus_parquet = root / "data" / "processed" / "corpus.parquet"
        self.corpus_jsonl = root / "data" / "processed" / "corpus.jsonl"

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
        self.corpus_df = self._load_corpus()
        self.clf = joblib.load(self.model_path)
        self.scoring_cfg = TermScoreConfig(src_threshold=src_threshold, tgt_threshold=tgt_threshold)
        self.analog = AnalogSuggester(analog_sim_threshold=analog_threshold)
        self.domains = sorted(self.lexicon_by_domain.keys())

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

    def _load_corpus(self) -> pd.DataFrame:
        if self.corpus_parquet.exists():
            try:
                df = pd.read_parquet(self.corpus_parquet)
            except Exception:
                if self.corpus_jsonl.exists():
                    df = pd.read_json(self.corpus_jsonl, lines=True)
                else:
                    df = pd.DataFrame()
        elif self.corpus_jsonl.exists():
            df = pd.read_json(self.corpus_jsonl, lines=True)
        else:
            df = pd.DataFrame()
        for c in ["id", "source", "domain", "abstract", "text"]:
            if c not in df.columns:
                df[c] = ""
        return df[["id", "source", "domain", "abstract", "text"]].copy()

    def predict_src(self, text: str) -> tuple[str | None, float | None]:
        labels = list(getattr(self.clf, "classes_", []))
        if not labels:
            return None, None
        probs = self.clf.predict_proba([text])[0]
        idx = int(probs.argmax())
        return str(labels[idx]), float(probs[idx])

    def annotate(self, text: str, src: str, tgt: Domain, max_terms: int = 8) -> dict[str, object]:
        predicted_src, pred_conf = self.predict_src(text)
        src_final = predicted_src if src == "auto" and predicted_src else src
        src_final = src_final if src_final in set(self.domains) else "CSM"

        extracted = extract_terms(text=text, max_terms=max(16, max_terms * 3))
        scored = score_terms(
            extracted_terms=extracted,
            src=src_final,
            tgt=tgt,
            all_domains=self.domains,
            term_stats=self.term_stats,
            lexicon_by_domain=self.lexicon_by_domain,
            cfg=self.scoring_cfg,
        )

        enriched: list[dict[str, object]] = []
        for row in scored:
            if len(enriched) >= max_terms:
                break
            term = str(row["term"])
            analogs = self.analog.suggest(term=term, target_candidates=self.lexicon_by_domain.get(tgt, []), top_k=5)
            evidence_term = str(analogs[0]["candidate"]) if analogs else term
            evidence = find_evidence_snippets(self.corpus_df, tgt=tgt, phrase=evidence_term, max_hits=2)
            row_out = dict(row)
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
            "terms": enriched,
        }

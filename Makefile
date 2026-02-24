SHELL := /bin/zsh
ROOT := $(PWD)
BACKEND := $(ROOT)/backend
FRONTEND := $(ROOT)/frontend

.PHONY: setup dev dev-backend dev-frontend fetch-sample train-sample test eval autotune diagnose \
	fetch-arxiv fetch-chemrxiv build-corpus mine-terms train-clf validate-artifacts textmining-all

setup:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r $(BACKEND)/requirements.txt
	cd $(FRONTEND) && npm install

dev:
	( cd $(BACKEND) && uvicorn app:app --reload --host 0.0.0.0 --port 8000 ) & \
	( cd $(FRONTEND) && npm run dev )

dev-backend:
	cd $(BACKEND) && uvicorn app:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd $(FRONTEND) && npm run dev

fetch-sample:
	mkdir -p data/raw data/processed
	cp data/processed/sample_corpus.jsonl data/raw/sample_corpus.jsonl
	@echo "Sample copied to data/raw/sample_corpus.jsonl"

train-sample:
	python3 scripts/03_mine_terms.py --corpus data/processed/sample_corpus.jsonl --lexicon-out data/processed/domain_lexicon.json --term-stats-out data/processed/term_stats.csv --top-n 120
	python3 scripts/04_train_domain_classifier.py --corpus data/processed/sample_corpus.jsonl --model-out models/domain_clf.joblib

test:
	python3 -m py_compile backend/*.py scripts/*.py
	@echo "Basic syntax checks passed"

eval:
	python3 scripts/eval/run_eval.py --api-base http://localhost:8000 --k 4
	@LOG=$$(ls -t logs/translation_eval_*.jsonl | head -1); \
	python3 scripts/eval/analyze_eval.py --input $$LOG

autotune:
	@LOG=$$(ls -t logs/translation_eval_*.jsonl | head -1); \
	python3 scripts/eval/autotune_reranker.py --input $$LOG

diagnose: eval autotune
	@echo "Diagnosis complete. See reports/diagnosis_*.md and reports/autotune_*.md"

fetch-arxiv:
	python3 scripts/textmining/fetch_from_config.py --config configs/textmining/domains.yaml --source arxiv --max-results 800

fetch-chemrxiv:
	python3 scripts/textmining/fetch_from_config.py --config configs/textmining/domains.yaml --source chemrxiv --max-results 800

build-corpus:
	python3 scripts/textmining/build_corpus.py --config configs/textmining/domains.yaml --out data/processed/corpus.parquet

mine-terms:
	python3 scripts/textmining/mine_terms.py --corpus data/processed/corpus.parquet --term-stats-out data/processed/term_stats.csv --lexicon-out data/processed/domain_lexicon.json --report-out reports/textmining/lexicon_report.md

train-clf:
	python3 scripts/textmining/train_domain_classifier.py --corpus data/processed/corpus.parquet --model-out models/domain_clf.joblib --report-out reports/textmining/classifier_report.md --metrics-out models/domain_clf_metrics.json

validate-artifacts:
	python3 scripts/textmining/validate_artifacts.py --lexicon data/processed/domain_lexicon.json --term-stats data/processed/term_stats.csv --clf-metrics models/domain_clf_metrics.json --report-out reports/textmining/validation_report.md

textmining-all: fetch-arxiv fetch-chemrxiv build-corpus mine-terms train-clf validate-artifacts
	@echo "Text-mining pipeline complete. See reports/textmining/*.md"

SHELL := /bin/zsh
ROOT := $(PWD)
BACKEND := $(ROOT)/backend
FRONTEND := $(ROOT)/frontend

.PHONY: setup dev dev-backend dev-frontend fetch-sample train-sample test

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

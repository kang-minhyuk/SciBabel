SHELL := /bin/zsh
ROOT := $(PWD)
BACKEND := $(ROOT)/backend
FRONTEND := $(ROOT)/frontend
ARXIV_TARGET ?= 2000

.PHONY: setup dev dev-backend dev-frontend fetch-sample train-sample test smoke smoke-phrases check spacy-model eval autotune diagnose \
	test-syntax fetch-arxiv fetch-chemrxiv fetch-openalex fetch-textmining build-corpus mine-terms train-clf validate-artifacts textmining-all diagnose-chemrxiv \
	fetch-arxiv-csm fetch-arxiv-pm fetch-openalex-chem fetch-openalex-cheme fetch-all corpus-report test-backend smoke-auto build-artifacts check-startup

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

test-syntax:
	python3 -m py_compile backend/*.py scripts/*.py
	@echo "Basic syntax checks passed"

test:
	cd $(BACKEND) && pytest -q

test-backend:
	cd $(BACKEND) && pytest -q

smoke:
	@set -e; \
	cd $(BACKEND) && SCIBABEL_FAKE_LLM=1 uvicorn app:app --host 127.0.0.1 --port 8000 >/tmp/scibabel_smoke_backend.log 2>&1 & \
	PID=$$!; \
	trap 'kill $$PID >/dev/null 2>&1 || true' EXIT; \
	sleep 3; \
	cd $(ROOT) && python3 scripts/eval/smoke_annotate.py --api-base http://127.0.0.1:8000

smoke-auto:
	@set -e; \
	cd $(BACKEND) && SCIBABEL_FAKE_LLM=1 uvicorn app:app --host 127.0.0.1 --port 8000 >/tmp/scibabel_smoke_auto_backend.log 2>&1 & \
	PID=$$!; \
	trap 'kill $$PID >/dev/null 2>&1 || true' EXIT; \
	for i in $$(seq 1 40); do \
		curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; \
		sleep 1; \
	done; \
	curl -sf http://127.0.0.1:8000/health >/dev/null; \
	cd $(ROOT) && python3 scripts/eval/smoke_auto_source.py --api-base http://127.0.0.1:8000

check-startup:
	cd $(ROOT) && python3 scripts/eval/smoke_render_startup.py

spacy-model:
	python3 -m spacy download en_core_web_sm

smoke-phrases:
	cd $(ROOT) && PYTHONPATH=$(BACKEND):$$PYTHONPATH python3 scripts/eval/smoke_phrase_extraction.py

check: test-backend smoke-auto
	@echo "Auto-source checks passed"

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

fetch-arxiv-csm:
	mkdir -p data/raw/arxiv logs
	python3 scripts/textmining/fetch_arxiv.py --domain CSM --categories cs.LG cs.AI cs.CL cs.CV stat.ML math.OC math.ST --category-mode true --target-doc-count $(ARXIV_TARGET) --sleep-sec 3.0 --max-retries 6 --backoff-base 2.0 --time-slice-days 180 --max-slices 40 --out-dir data/raw/arxiv --merge-out data/raw/arxiv/csm_arxiv.jsonl --resume true --out data/raw/arxiv/csm_arxiv.jsonl --log-file logs/textmining_fetch_arxiv_csm.log

fetch-arxiv-pm:
	mkdir -p data/raw/arxiv logs
	python3 scripts/textmining/fetch_arxiv.py --domain PM --categories cond-mat.mtrl-sci cond-mat.mes-hall cond-mat.soft physics.comp-ph physics.chem-ph --category-mode true --target-doc-count $(ARXIV_TARGET) --sleep-sec 3.0 --max-retries 6 --backoff-base 2.0 --time-slice-days 180 --max-slices 40 --out-dir data/raw/arxiv --merge-out data/raw/arxiv/pm_arxiv.jsonl --resume true --out data/raw/arxiv/pm_arxiv.jsonl --log-file logs/textmining_fetch_arxiv_pm.log

fetch-chemrxiv:
	python3 scripts/textmining/fetch_from_config.py --config configs/textmining/domains.yaml --source chemrxiv --max-results 800

diagnose-chemrxiv:
	mkdir -p reports/chemrxiv
	python3 scripts/textmining/diagnose_chemrxiv.py --out reports/chemrxiv/diagnosis_latest.md

fetch-openalex:
	python3 scripts/textmining/fetch_from_config.py --config configs/textmining/domains.yaml --source openalex --max-results 800 --verbose

fetch-openalex-chem:
	mkdir -p data/raw/openalex logs
	python3 scripts/textmining/fetch_openalex.py --domain CHEM --display-name "Chemistry" --concept-id C178790620 --target-doc-count 10000 --sleep-sec 0.3 --out data/raw/openalex/chem_openalex.jsonl --other-out data/raw/openalex/cheme_openalex.jsonl --log-file logs/textmining_fetch_openalex_chem.log --verbose

fetch-openalex-cheme:
	mkdir -p data/raw/openalex logs
	python3 scripts/textmining/fetch_openalex.py --domain CHEME --display-name "Chemical Engineering" --concept-id C185592680 --keyword reactor --keyword separation --keyword distillation --keyword membrane --keyword adsorption --keyword catalysis --keyword "mass transfer" --keyword transport --keyword "process control" --keyword "reaction engineering" --keyword "heat transfer" --target-doc-count 10000 --sleep-sec 0.3 --out data/raw/openalex/cheme_openalex.jsonl --other-out data/raw/openalex/chem_openalex.jsonl --log-file logs/textmining_fetch_openalex_cheme.log --verbose

fetch-all: fetch-arxiv-csm fetch-arxiv-pm fetch-openalex-cheme fetch-openalex-chem
	@echo "Fetch complete for CSM/PM/CHEM/CHEME"

fetch-textmining:
	$(MAKE) fetch-all

build-corpus:
	python3 scripts/textmining/build_corpus.py --config configs/textmining/domains.yaml --out-full data/processed/corpus_full.parquet --out-balanced data/processed/corpus_balanced.parquet --diagnostics-out reports/textmining/corpus_scale_report.md

build-artifacts:
	python3 scripts/textmining/build_artifacts.py --config configs/textmining/domains.yaml

corpus-report:
	python3 scripts/textmining/corpus_report.py --corpus data/processed/corpus_full.parquet --out reports/textmining/corpus_scale_report.md

mine-terms:
	python3 scripts/textmining/mine_terms.py --corpus data/processed/corpus_full.parquet --term-stats-out data/processed/term_stats.csv --lexicon-out data/processed/domain_lexicon.json --report-out reports/textmining/lexicon_report.md

train-clf:
	python3 scripts/textmining/train_domain_classifier.py --corpus data/processed/corpus_full.parquet --model-out models/domain_clf.joblib --report-out reports/textmining/classifier_report.md --metrics-out models/domain_clf_metrics.json

validate-artifacts:
	python3 scripts/textmining/validate_artifacts.py --lexicon data/processed/domain_lexicon.json --term-stats data/processed/term_stats.csv --clf-metrics models/domain_clf_metrics.json --report-out reports/textmining/validation_report.md

textmining-all: fetch-textmining build-corpus mine-terms train-clf validate-artifacts
	@echo "Text-mining pipeline complete. See reports/textmining/*.md"

# SciBabel Architecture (MVP)

## 1) Offline pipeline

1. Fetch abstracts (arXiv API) with [scripts/01_fetch_arxiv.py](../scripts/01_fetch_arxiv.py).
2. Build a unified domain-labeled corpus with [scripts/02_build_corpus.py](../scripts/02_build_corpus.py).
3. Mine domain terms with TF-IDF + unigram log-odds using [scripts/03_mine_terms.py](../scripts/03_mine_terms.py):
   - `data/processed/domain_lexicon.json`
   - `data/processed/term_stats.csv`
4. Train domain classifier (TF-IDF + LogisticRegression) with [scripts/04_train_domain_classifier.py](../scripts/04_train_domain_classifier.py):
   - `models/domain_clf.joblib`

## 2) Online translation API

[backend/app.py](../backend/app.py) exposes:

- `GET /health`
- `POST /translate`

`/translate` flow:

1. Validate payload (`text`, `src`, `tgt`, `k`).
2. Choose prompt action template from [backend/prompts.py](../backend/prompts.py).
3. Generate $k$ candidates via Gemini with varied temperature.
4. Score each candidate with:

$$
R = w_{domain} P(tgt \mid candidate) + w_{meaning} J(src, cand) + w_{lex} C_{lex}(cand, tgt)
$$

Where:
- $P(tgt \mid candidate)$ from trained classifier
- $J$ is token Jaccard overlap proxy
- $C_{lex}$ is target-domain lexicon coverage

5. Return best output, score breakdown, and sorted candidates.

## 3) Optional exploration policy

[backend/bandit.py](../backend/bandit.py) implements in-memory epsilon-greedy action choice keyed by `src->tgt`.

## 4) Frontend MVP

[frontend/app/page.tsx](../frontend/app/page.tsx) provides:

- Text input
- `src/tgt` domain selectors (`CSM`, `PM`, `CCE`)
- `k` slider
- Translate button
- Result panel with best candidate and candidate rankings

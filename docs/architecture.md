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
2. Build multi-action prompts from [backend/prompts.py](../backend/prompts.py):
   - `literal`
   - `domain_steered`
   - `process_framing`
   - `educator`
3. Generate exactly $k$ candidates in parallel (one per action first, then additional exploration slots if needed).
4. Filter and score each candidate with semantic constraints:

$$
   ext{eligible if } S_{sem}(src,cand) \ge \tau_{sem}
$$

$$
R = P(tgt \mid cand) + \alpha\,E_{lex}(cand,tgt) - \beta\,\max(0, C_{copy}(src,cand)-\tau_{copy})
$$

Where:
- $P(tgt \mid candidate)$ from trained classifier
- $S_{sem}$ is sentence-embedding similarity (fallback: token overlap)
- $E_{lex}$ is weighted lexical evidence from `term_stats.csv` log-odds (fallback: lexicon coverage)
- $C_{copy}$ is ROUGE-L based source-copy score

5. Apply strategy penalty from term policy layer and rerank.
6. Return best output with transparent metadata (`semantic_sim`, `copy_score`, `lex_terms_hit`, actions used, source-warning signals).

## 2.1) Term strategy layer

Before generation, backend builds key-term strategies in [backend/term_strategy.py](../backend/term_strategy.py):

1. Extract 1-3 gram key terms from input text.
2. Rank terms by lexicon membership and simple length/frequency heuristics.
3. Compute target-domain native score from:
   - direct/near lexicon membership,
   - optional `term_stats.csv` log-odds.
4. Classify each term into one of:
   - `equivalent`: native in target community
   - `analogous`: use conceptual neighbor in target domain
   - `unique`: preserve term + short parenthetical explanation
   - `intranslatable`: preserve term + mark as domain-specific concept

Prompt gets a compact term-handling instruction block for top terms.

During reranking, a small strategy penalty is applied if candidates violate policy:

- missing `unique` term,
- missing `analogous` neighbor,
- replacing `intranslatable` term.

`/translate` returns transparent `term_strategies` metadata.

## 2.2) Source-domain mismatch warning

For each request, backend also predicts the source-domain label of the input text using the classifier.
If predicted source differs from user-provided `src` and confidence exceeds configured threshold,
response includes:

- `src_warning = true`
- `predicted_src`
- `predicted_src_confidence`

## 3) Optional exploration policy

[backend/bandit.py](../backend/bandit.py) implements in-memory epsilon-greedy action choice keyed by `src->tgt`.

## 4) Frontend MVP

[frontend/app/page.tsx](../frontend/app/page.tsx) provides:

- Text input
- `src/tgt` domain selectors (`CSM`, `PM`, `CCE`)
- `k` slider
- Translate button
- Result panel with best candidate and candidate rankings

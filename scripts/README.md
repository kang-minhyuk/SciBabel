# Data pipeline scripts

## Order

1. `01_fetch_arxiv.py` - fetch titles/abstracts from arXiv API to JSONL
2. `02_build_corpus.py` - merge domain files into one corpus parquet/jsonl
3. `03_mine_terms.py` - produce domain lexicon and term stats
4. `04_train_domain_classifier.py` - train/save TF-IDF + logistic model

All scripts support `--help`.

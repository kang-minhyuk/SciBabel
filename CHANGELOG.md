# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-23

### Added

- Initial SciBabel repository scaffold.
- FastAPI backend with `GET /health` and `POST /translate`.
- Gemini integration entrypoint via `generate_with_gemini()` using `GEMINI_API_KEY`.
- Reward-based reranking with domain probability, Jaccard meaning proxy, and lexicon coverage.
- Optional in-memory epsilon-greedy prompt bandit keyed by route (`src->tgt`).
- Offline scripts for arXiv fetching, corpus build, term mining, and classifier training.
- Next.js + TypeScript + Tailwind frontend MVP UI.
- Sample corpus and generated sample artifacts for local end-to-end run.
- Project docs, setup instructions, `.env.example`, `.gitignore`, and `Makefile` targets.

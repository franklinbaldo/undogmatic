# Repository Guidelines

## Project Structure & Module Organization
- `undogmatic/`: core library (no heavy I/O on import). Key files: `prompts.py` (PT‑BR prompt, strict JSON output), `llm_scorer.py` (`score_text` with temperature 0.0, JSON parsing with brief retries).
- `scripts/`: small CLIs. `make_ab_pairs.py` (build A/B from `data/curated/temas_seed.jsonl`), `eval_ab.py` (paired A/B, Wilcoxon, report).
- `data/curated/`: tiny seed datasets (≤ ~100 linhas). Example file: `temas_seed.jsonl`.
- `reports/`: light outputs (`ab_test.md`, small CSVs/figures).
- `runs/`: local logs/cache; not versioned.

## Setup, Build & Local Commands
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt || pip install pandas scipy scikit-learn pydantic tqdm python-dotenv openai
mkdir -p data/curated reports runs
```
Env vars (via `.env`): `OPENAI_API_KEY=...`, `UNDOGMATIC_MODEL=gpt-4o-mini` (provider/base URL optional).

Key commands:
- Generate A/B pairs: `python scripts/make_ab_pairs.py --in data/curated/temas_seed.jsonl --out data/curated/ab_pairs.jsonl`
- Run A/B eval: `python scripts/eval_ab.py --in data/curated/ab_pairs.jsonl --out-md reports/ab_test.md --out-csv reports/ab_results.csv`
- Tests: `pytest -q`
- Lint/format: `ruff check .` • `black --check .`

## Coding Style & Naming Conventions
- Style: Black + Ruff + Isort; clear names; short docstrings. Code/comments in English; prompts/README in PT‑BR ok.
- LLM: deterministic (`temperature=0.0`), strict JSON; handle parse errors with 1–2 quick retries; avoid duplicate calls (optional cache under `runs/cache/`, SHA1 of text).

## Testing Guidelines
- Do not call real LLMs in tests; mock I/O. Prefer small, fast tests in `tests/` named `test_*.py`.
- Keep smoke tests green before PR: `pytest -q`.

## Commit & Pull Request Guidelines
- Branches: `feat/<short>`, `fix/<short>`, `docs/<short>`.
- Commits: imperative, concise (e.g., “add scorer cache”, “fix json parse”).
- PRs include: motivation (1–2 lines), bullet changes, how to test (commands), risks/limits, checklist (lint, tests, report updated).

## Security & Configuration Tips
- Never commit secrets (`OPENAI_API_KEY`) or large artifacts. Ensure `.env`, `.venv/`, `runs/`, `data/raw/`, `reports/*.csv` are ignored.
- No scraping or external data pulls by default; use manual seeds.

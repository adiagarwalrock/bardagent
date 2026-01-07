# Repository Guidelines

## Project Structure & Modules

- Core agent flow lives in `main.py`; tool registry in `tools/__init__.py` with concrete tools under `tools/action_tool.py`, `tools/info_tools.py`, and `tools/web_fetch.py`.
- Streamlit UI entrypoint: `app.py` (stores chat history in `chat_history.json` and uses `utilities/history.py` helpers).
- Shared helpers: `utilities/` (logging, prompts, history, text cleaning). Model/system prompts are in `utilities/prompts.py`.
- Evaluations: `evals/dataset.jsonl` and runner `evals/run_eval.py` write results to `eval_result.json`.
- Runtime artifacts: `bardagent.log`, `chat_history.json`, and any Playwright/Streamlit cache; keep them out of commits unless intentionally updated.

## Setup, Build, and Dev Commands

- uv-first workflow (preferred):
  - Install deps: `uv sync` (uses `uv.lock`).
  - Run CLI agent: `uv run python main.py`.
  - Run Streamlit UI: `uv run streamlit run app.py` (add `--server.port 8501` if needed).
  - Run evaluations: `uv run python evals/run_eval.py --dataset evals/dataset.jsonl --max-workers 4`.
  - Add packages: `uv add <package>`; remove with `uv remove <package>`. Regenerate lock with `uv lock` if needed.
- Fallback (only if uv unavailable): `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
- Environment: copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (required for Gemini). Do not commit secrets.
- Optional web-scrape tooling: install Playwright browser once with `uv run python -m playwright install chromium` (or pip equivalent if not using uv).

## Coding Style & Naming Conventions

- Python 3.11; follow PEP 8 with 4-space indents and type hints (existing code uses `typing` and `TypedDict`).
- Prefer pure functions and small modules; keep tool names snake_case and descriptive (`math_evaluator`, `wikipedia_search`).
- Logging: use `utilities.logger.logger` (aliased as `logging` in modules); avoid bare `print` outside CLI entrypoints.
- Normalize model outputs with `utilities.utils.normalize_content` before display or persistence.

## Testing & Quality Checks

- Primary check is the eval runner: `python evals/run_eval.py` (writes summary to `eval_result.json`; inspect pass rate by category).
- When adding prompts/tools, add representative cases to `evals/dataset.jsonl` (JSONL with `prompt`, `expected_contains` or `expected_exact`).
- For web-scrape changes, smoke test with a short prompt via CLI and via UI to ensure tool calls serialize correctly.

## Commit & Pull Request Guidelines

- Commit messages follow the current history: imperative, single-sentence summaries (e.g., “Refactor chat history management, enhance tool integration…”).
- Keep commits scoped: one behavior or bug fix per commit; avoid bundling formatting with logic changes.
- PRs should include: brief summary, testing notes (commands run), and mention of any new env vars or files. Add screenshots/gifs for UI tweaks.
- Verify that generated artifacts (`chat_history.json`, `eval_result.json`, caches) are excluded or intentionally updated before opening a PR.

## Security & Configuration Tips

- Never commit API keys. Validate `.env` locally and strip secrets from logs before sharing.
- User data lives in `chat_history.json`; clear it with the UI “Clear Chat” button or via `utilities/history.clear_history` during testing.
- When enabling scraping tools, keep max result counts and timeouts modest to avoid long-running Playwright sessions.

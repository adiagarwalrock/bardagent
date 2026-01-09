# AGENTS Guide (Root Scope)

Repository-wide instructions for all agentic coding assistants working in `/home/adiagarwal/bardagent`. Keep this file close while editing; more specific rules override these if nested.

## Quick Reminders

- Mission: maximize hiring-assignment alignment (see `TASK.pdf`); keep behavior scoped to that product brief.
- Python 3.11; prefer uv workflow; avoid committing runtime artifacts (`bardagent.log`, `chat_history.json`, caches, `eval_result.json`) unless explicitly intended.
- Secrets: never commit API keys; `.env.example` documents required vars. Real `.env` must stay local.
- No Cursor or Copilot rule files present; follow guidelines here.

## Build / Run / Lint / Test

- Install deps (preferred): `uv sync` (uses `uv.lock`).
- Add dependency: `uv add <package>`; remove with `uv remove <package>`; relock with `uv lock` if needed.
- Fallback without uv: `python -m venv .venv && source .venv/bin/activate && pip install -e .`.
- CLI agent: `uv run python main.py`.
- Streamlit UI: `uv run streamlit run app.py` (optionally `--server.port 8501`).
- Evaluations (primary check): `uv run python evals/run_eval.py --dataset evals/dataset.jsonl --max-workers 4` (writes `eval_result.json`).
- Focused/smaller eval run ("single test" equivalent): create a temp JSONL with one case (or filter via `jq 'NR==1' evals/dataset.jsonl > /tmp/one.jsonl`) then run `uv run python evals/run_eval.py --dataset /tmp/one.jsonl --max-workers 1`.
- Smoke prompt via CLI after tool/prompt changes to ensure tool call serialization works.
- No dedicated lint/format config present; rely on PEP 8 conventions and module patterns below.

## Environment & Data

- Required: `GOOGLE_API_KEY` in `.env` for Gemini. Copy `.env.example` to `.env` before running.
- Optional scrape: install Playwright browser once: `uv run python -m playwright install chromium`.
- Chat history stored in `chat_history.json`; clear via UI "Clear Chat" or `utilities/history.clear_history` when needed.

## Project Structure

- `main.py`: core agent run loop, calls `core.get_agent`, normalizes outputs.
- `tools/`: tool registry (`__init__.py`) plus concrete tools in `action_tool.py`, `info_tools.py`, `web_fetch.py`.
- `utilities/`: prompts, logging, history persistence, text cleaning, normalization utilities.
- `app.py`: Streamlit UI wiring; persists chat history.
- `evals/`: `dataset.jsonl` prompt set; `run_eval.py` runner emits `eval_result.json`.

## Coding Style (General)

- Follow PEP 8 with 4-space indents; keep lines reasonably short (~100 chars).
- Prefer pure, small functions; keep modules focused.
- Use type hints everywhere practical (`from __future__ import annotations` if adding new modules).
- Avoid introducing new global state; prefer explicit parameters.
- Do not over-engineer; keep changes minimal and scoped to the task.

## Imports

- Order: standard library, third-party, local; separate groups with blank lines.
- Use absolute imports within repo (e.g., `from utilities.logger import logger`).
- Avoid unused imports; keep aliasing consistent (`logger as logging` mirrors existing pattern).

## Naming Conventions

- Modules/files/functions: snake_case; classes: CamelCase; constants: UPPER_SNAKE_CASE.
- Tool identifiers should be descriptive (`math_evaluator`, `wikipedia_search`, etc.).
- Variables should be clear and non-abbreviated; avoid single-letter names except loop indices.

## Types & Data Handling

- Prefer explicit `TypedDict`/dataclasses for structured data when expanding schemas.
- Use `list[str]`/`dict[str, Any]` style over `List`/`Dict` unless matching existing code.
- Normalize external/model outputs with `utilities.utils.normalize_content` before display or storage.
- For JSON IO, use helpers `utilities.utils.read_json`/`write_json`.

## Error Handling

- Validate user inputs early (see `run_chat` guard on empty messages).
- Prefer explicit exceptions with helpful messages over silent failures.
- When tool execution can fail, log and return safe fallbacks; avoid leaking stack traces to end users.
- Keep control flow simple; avoid broad `except Exception` unless re-raising with context.

## Logging

- Use `utilities.logger.logger` (import as `logger` or alias to `logging` as in `main.py`).
- Avoid `print` except in CLI entrypoints (`if __name__ == "__main__"` blocks or Streamlit UI).
- Keep log messages concise; include counts/identifiers that aid debugging.
- Log tool usage summaries where helpful (see `main.py` tool tracking).

## Formatting & Docstrings

- Use triple-double-quoted docstrings for public functions/classes; include brief purpose and args/returns when non-obvious.
- Favor explicit named arguments; avoid positional booleans when adding new call sites.
- Keep whitespace tidy; no trailing spaces; one blank line between top-level defs.

## Control Flow & Side Effects

- Keep functions single-responsibility; factor helper functions for repeated blocks.
- Avoid hidden I/O in utilities unless clearly documented.
- When mutating inputs, document it; prefer returning new objects.

## Streamlit / UI Notes

- UI state stored in `chat_history.json`; guard against corruption; handle empty history gracefully.
- Keep UI responses short and sanitized; use `normalize_content` for model outputs.

## Tools & Prompts

- Register new tools in `tools/__init__.py`; follow existing naming and signatures.
- Keep tool outputs structured and sanitized; avoid excessive verbosity in tool return strings.
- When adding prompts, align with `utilities/prompts.py` patterns; keep date/time formatting consistent (`%Y-%m-%d`).

## Evaluations

- Update `evals/dataset.jsonl` with representative cases when adding capabilities; include `expected_contains` or `expected_exact` fields.
- Run `uv run python evals/run_eval.py --dataset evals/dataset.jsonl --max-workers 4` before shipping relevant changes; inspect `eval_result.json` for regressions.
- For rapid iteration, run with a pruned JSONL (single entry) as noted above; keep temporary files out of commits.

## Persistence & Artifacts

- Do not commit: `bardagent.log`, `chat_history.json`, Streamlit/Playwright caches, temporary eval JSONL slices, `eval_result.json` unless intentionally updating baselines.
- Ensure new artifacts are documented and, if necessary, added to `.gitignore`.

## Testing Guidance

- No pytest/unittest suite present; evaluations act as primary tests.
- If you add tests, keep them fast and colocated; prefer `uv run pytest <path>::<test_name>` pattern (if pytest is introduced) and document additions here.
- When reproducing bugs, craft minimal eval entries to capture expected behavior.

## Error Surfaces & Fallbacks

- For agent responses, ensure final message is an `AIMessage`; append normalized fallback when tool streams end unexpectedly (see `main.py`).
- When tool calls fail, return user-friendly notices and log details for debugging.

## Performance & Timeouts

- Keep tool calls bounded; respect timeouts for web/scrape actions; avoid long-running Playwright sessions.
- Limit result counts for searches; prefer lightweight parsing in scraping utilities.

## Contribution & PR Hygiene

- Commit messages: imperative, single sentence (e.g., "Refactor chat history management...").
- Keep commits scoped; avoid mixing formatting-only changes with logic updates.
- PRs should summarize changes, list commands run (especially evals), and mention new env vars/files; include UI screenshots/GIFs when relevant.

## Safety & Security

- Strip sensitive data from logs and responses; never echo secrets.
- Handle user inputs defensively; sanitize external text before display.

## File-Specific Notes

- Respect existing aliasing: `from utilities.logger import logger as logging` is intentional in `main.py`.
- Maintain current prompt templating conventions in `utilities/prompts.py` (date insertion, formatting).
- Keep `utilities/utils.py` helpers pure and side-effect-light; prefer reuse over reimplementation.

## When in Doubt

- Prefer uv commands; stay minimal; keep artifacts clean.
- Ask for clarification (via comments/issues) if requirements conflict with `TASK.pdf` goals.
- Align changes with maximizing eval performance and user hiring-readiness outcomes.

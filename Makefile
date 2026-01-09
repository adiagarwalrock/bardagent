.PHONY: setup run run-ui eval clean help

# Default target
help:
	@echo "BardAgent Management Commands"
	@echo "============================="
	@echo "make setup      - Install dependencies (using uv)"
	@echo "make run        - Run the CLI agent"
	@echo "make run-ui     - Launch the Streamlit Dashboard"
	@echo "make eval       - Run the full evaluation suite"
	@echo "make eval-fast  - Run a fast evaluation (1 worker, small batch)"
	@echo "make clean      - Remove temporary artifacts and caches"
	@echo "make format     - Run code formatting (ruff)"

setup:
	uv sync
	uv run python -m playwright install chromium

run:
	uv run python main.py

run-ui:
	uv run streamlit run app.py

eval:
	uv run python evals/run_eval.py --max-workers 4

eval-fast:
	uv run python evals/run_eval.py --max-workers 1 --batch-size 1 --no-judge

clean:
	rm -rf __pycache__ .ruff_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f eval_result.json chat_history.json temp_*.jsonl

format:
	uv run ruff check . --fix
	uv run ruff format .

# Question-Answering Agent

## Overview

LLM-powered chatbot that uses tool calling (at least two tools), provides evaluations, and exposes a web dashboard for interaction and metrics.

## Setup

1. Clone repo and install deps (`python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` or `npm install` if JS stack).
2. Set environment variables in `.env` (LLM keys, search key, etc.).

## Run

- Backend: `uvicorn app:app --reload` (Python) or `npm run dev` (JS/TS).
- Frontend: `npm run dev` (if separate).
- Evaluation: `python eval/run_eval.py` (or equivalent script).

## Keys to request

- LLM: OpenAI/Anthropic (and optional Fireworks).
- Search: SerpAPI or Tavily.
- Optional: vector store hosting (if not local) and any rate-limit quotas for eval.

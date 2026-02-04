"""GAIA Benchmark Runner - Fetches questions, runs agent, and submits answers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from langchain_core.messages import AIMessage

from main import run_chat
from utilities.answer_extractor import clean_for_exact_match, extract_clean_answer
from utilities.logger import logger
from utilities.utils import normalize_content

# GAIA API Configuration
GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"
REQUEST_TIMEOUT = 30
FILE_DOWNLOAD_TIMEOUT = 60

# File type to prompt template mapping
FILE_PROMPTS = {
    "audio": (
        {".mp3", ".wav", ".m4a"},
        "I have an audio file at: {file_path}\n\n"
        "Please transcribe or analyze this audio file to answer the following question:\n\n"
        "{question}",
    ),
    "image": (
        {".png", ".jpg", ".jpeg", ".gif", ".webp"},
        "I have an image file at: {file_path}\n\n"
        "Please analyze this image to answer the following question:\n\n"
        "{question}",
    ),
    "excel": (
        {".xlsx", ".xls"},
        "I have an Excel file at: {file_path}\n\n"
        "Please read and analyze this Excel file to answer the following question:\n\n"
        "{question}",
    ),
    "python": (
        {".py"},
        "I have a Python file at: {file_path}\n\n"
        "Please execute this Python code and answer the following question:\n\n"
        "{question}",
    ),
}

EXACT_MATCH_INSTRUCTIONS = """
IMPORTANT: Provide ONLY the direct answer. No explanations, no "The answer is...", just the answer itself.
- For lists: use comma-separated format
- For numbers: just the number
- For names: just the name as requested
- Be precise and concise."""


def fetch_questions() -> list[dict[str, Any]]:
    """Fetch all questions from the GAIA API."""
    url = f"{GAIA_API_BASE}/questions"
    logger.info(f"Fetching questions from {url}")

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        questions = response.json()
        logger.info(f"Fetched {len(questions)} questions")
        return questions
    except requests.RequestException as e:
        logger.error(f"Failed to fetch questions: {e}")
        raise


def download_file(task_id: str, file_name: str, output_dir: Path) -> Optional[Path]:
    """Download a file associated with a task."""
    if not file_name:
        return None

    url = f"{GAIA_API_BASE}/files/{task_id}"
    output_path = output_dir / file_name

    logger.info(f"Downloading file for task {task_id}: {file_name}")

    try:
        response = requests.get(url, timeout=FILE_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logger.info(f"Downloaded file to {output_path}")
        return output_path
    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        return None


def build_prompt_for_question(
    question: dict[str, Any],
    file_path: Optional[Path] = None,
) -> str:
    """
    Build a prompt for the agent based on the GAIA question.

    Includes special instructions for exact-match formatting.
    """
    base_question = question.get("question", "")

    if file_path:
        file_ext = file_path.suffix.lower()

        # Find matching file type prompt
        for _, (extensions, template) in FILE_PROMPTS.items():
            if file_ext in extensions:
                base_question = template.format(
                    file_path=file_path, question=base_question
                )
                break
        else:
            # Generic fallback for unknown file types
            base_question = (
                f"I have a file at: {file_path}\n\n"
                f"Please analyze this file to answer the following question:\n\n"
                f"{base_question}"
            )

    return f"{base_question}\n{EXACT_MATCH_INSTRUCTIONS}"


def run_gaia_question(
    question: dict[str, Any],
    file_path: Optional[Path] = None,
) -> tuple[str, str]:
    """
    Run the agent on a single GAIA question.

    Args:
        question: Question dict from GAIA API
        file_path: Optional path to downloaded file

    Returns:
        Tuple of (raw_response, clean_answer)
    """
    task_id = question.get("task_id", "unknown")
    logger.info(f"Running question {task_id}")

    prompt = build_prompt_for_question(question, file_path)

    try:
        messages, _ = run_chat(prompt)

        # Extract the final AI message
        ai_msg = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage)),
            None,
        )
        raw_response = normalize_content(ai_msg.content) if ai_msg else ""

        # Clean the answer for exact-match
        question_text = question.get("question", "")
        clean_answer = clean_for_exact_match(
            extract_clean_answer(raw_response, question_text)
        )

        logger.info(f"Task {task_id} - Raw: {raw_response[:100]}...")
        logger.info(f"Task {task_id} - Clean: {clean_answer}")

        return raw_response, clean_answer

    except Exception as e:
        logger.error(f"Failed to run question {task_id}: {e}", exc_info=True)
        return f"Error: {e}", ""


def run_all_questions(
    questions: Optional[list[dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[dict[str, Any]]:
    """
    Run the agent on all GAIA questions.

    Args:
        questions: Optional list of questions (fetches if not provided)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of result dicts with task_id and submitted_answer
    """
    if questions is None:
        questions = fetch_questions()

    results = []
    total = len(questions)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for i, question in enumerate(questions):
            task_id = question.get("task_id", "")
            file_name = question.get("file_name", "")

            # Download file if present
            file_path = (
                download_file(task_id, file_name, temp_path) if file_name else None
            )

            # Run the question
            raw_response, clean_answer = run_gaia_question(question, file_path)

            results.append(
                {
                    "task_id": task_id,
                    "question": question.get("question", ""),
                    "file_name": file_name,
                    "raw_response": raw_response,
                    "submitted_answer": clean_answer,
                }
            )

            if progress_callback:
                progress_callback(i + 1, total)

    return results


def submit_answers(
    username: str,
    agent_code: str,
    answers: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Submit answers to the GAIA scoring API.

    Args:
        username: HuggingFace username
        agent_code: URL to the agent's code (HF Space tree/main)
        answers: List of {"task_id": ..., "submitted_answer": ...}

    Returns:
        API response with score
    """
    url = f"{GAIA_API_BASE}/submit"

    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": [
            {"task_id": a["task_id"], "submitted_answer": a["submitted_answer"]}
            for a in answers
        ],
    }

    logger.info(f"Submitting {len(answers)} answers for {username}")

    try:
        response = requests.post(url, json=payload, timeout=FILE_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Submission result: {result}")
        return result
    except requests.RequestException as e:
        logger.error(f"Submission failed: {e}")
        raise


def test_single_question(task_id: str) -> dict[str, Any]:
    """Test the agent on a single question by task ID."""
    questions = fetch_questions()
    question = next((q for q in questions if q.get("task_id") == task_id), None)

    if not question:
        raise ValueError(f"Question with task_id {task_id} not found")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_name = question.get("file_name", "")
        file_path = download_file(task_id, file_name, temp_path) if file_name else None

        raw, clean = run_gaia_question(question, file_path)

        return {
            "task_id": task_id,
            "question": question.get("question", ""),
            "file_name": file_name,
            "raw_response": raw,
            "submitted_answer": clean,
        }


def _print_progress(current: int, total: int) -> None:
    """Print progress to stdout."""
    print(f"Progress: {current}/{total}")


def main() -> None:
    """CLI entry point for GAIA runner."""
    import argparse

    parser = argparse.ArgumentParser(description="GAIA Benchmark Runner")
    parser.add_argument(
        "--test-local", action="store_true", help="Run all questions locally"
    )
    parser.add_argument(
        "--test-single", type=str, help="Test a single question by task_id"
    )
    parser.add_argument("--list", action="store_true", help="List all questions")
    args = parser.parse_args()

    if args.list:
        for q in fetch_questions():
            print(f"- {q['task_id']}: {q['question'][:80]}...")
            if q.get("file_name"):
                print(f"  File: {q['file_name']}")

    elif args.test_single:
        result = test_single_question(args.test_single)
        print(json.dumps(result, indent=2))

    elif args.test_local:
        results = run_all_questions(progress_callback=_print_progress)

        with open("gaia_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nCompleted {len(results)} questions")
        print("Results saved to gaia_results.json")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

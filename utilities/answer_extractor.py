"""Answer extraction utilities for GAIA benchmark exact-match scoring."""

from __future__ import annotations

import re
from typing import Optional


def extract_clean_answer(response: str, question: Optional[str] = None) -> str:
    """
    Extract a clean answer from an agent response for exact-match scoring.

    The GAIA benchmark uses exact string matching, so this function:
    - Removes common prefixes like "The answer is", "FINAL ANSWER:", etc.
    - Extracts the core answer from explanatory text
    - Handles formatting for lists, numbers, and special cases

    Args:
        response: The full agent response
        question: Optional question text for context-aware extraction

    Returns:
        Clean answer suitable for exact-match comparison
    """
    if not response:
        return ""

    text = response.strip()
    text = _remove_answer_prefixes(text)
    text = _remove_trailing_explanations(text)

    if question:
        text = _apply_question_specific_formatting(text, question.lower())

    return text.strip()


def _remove_answer_prefixes(text: str) -> str:
    """Remove common answer prefixes like 'The answer is...'."""
    prefixes = [
        r"^FINAL ANSWER:\s*",
        r"^The answer is:?\s*",
        r"^Answer:?\s*",
        r"^The final answer is:?\s*",
        r"^Based on .*?, the answer is:?\s*",
        r"^In conclusion,?\s*",
        r"^Therefore,?\s*",
        r"^So,?\s*",
    ]
    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _remove_trailing_explanations(text: str) -> str:
    """Remove trailing explanations after the answer."""
    patterns = [
        r"\.\s*This is because.*$",
        r"\.\s*Note that.*$",
        r"\.\s*I found this by.*$",
        r"\.\s*According to.*$",
        r"\n\n.*$",  # Remove anything after double newline
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _apply_question_specific_formatting(text: str, question_lower: str) -> str:
    """Apply formatting based on question keywords."""
    # Comma-separated list questions
    if "comma separated" in question_lower or "comma-separated" in question_lower:
        text = format_comma_list(text)

    # Alphabetize if requested
    if "alphabetize" in question_lower or "alphabetical order" in question_lower:
        text = alphabetize_list(text)

    # Currency formatting
    if "usd" in question_lower or "dollar" in question_lower:
        text = format_currency(text)

    # Just the number
    if "provide the number" in question_lower or "give the number" in question_lower:
        text = extract_number(text)

    # First name only
    if "first name" in question_lower and "only" in question_lower:
        text = extract_first_name(text)

    # Last name only
    if "last name" in question_lower and "only" in question_lower:
        text = extract_last_name(text)

    return text


def format_comma_list(text: str) -> str:
    """Format text as a comma-separated list."""
    if "," in text:
        items = [item.strip() for item in text.split(",")]
        return ", ".join(items)

    # Try to extract list items from bullet points or newlines
    items = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\d.\-*•]+\s*", "", line)  # Remove bullets/numbers
        if line:
            items.append(line)

    return ", ".join(items) if len(items) > 1 else text


def alphabetize_list(text: str) -> str:
    """Alphabetize a comma-separated list."""
    if "," not in text:
        return text

    items = [item.strip() for item in text.split(",")]
    items.sort(key=str.lower)
    return ", ".join(items)


def format_currency(text: str) -> str:
    """Format as USD currency with two decimal places."""
    match = re.search(r"[$]?(\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if not match:
        return text

    num_str = match.group(1).replace(",", "")
    try:
        return f"${float(num_str):.2f}"
    except ValueError:
        return text


def extract_number(text: str) -> str:
    """Extract just the numeric value from text."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else text


def extract_first_name(text: str) -> str:
    """Extract just the first name from a full name."""
    text = re.sub(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s*", "", text, flags=re.IGNORECASE)
    parts = text.strip().split()
    return parts[0] if parts else text


def extract_last_name(text: str) -> str:
    """Extract just the last name from a full name."""
    parts = text.strip().split()
    return parts[-1] if parts else text


def clean_for_exact_match(text: str) -> str:
    """
    Final cleanup for exact match comparison.

    Removes extra whitespace and normalizes punctuation.
    """
    text = " ".join(text.split())  # Normalize whitespace
    return text.rstrip(".").strip()  # Remove trailing period


# Question-type specific extractors


def extract_ioc_code(text: str) -> str:
    """Extract IOC country code (3 letters uppercase)."""
    match = re.search(r"\b([A-Z]{3})\b", text)
    return match.group(1) if match else text


def extract_page_numbers(text: str) -> str:
    """Extract page numbers as comma-delimited list in ascending order."""
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return text

    sorted_nums = sorted({int(n) for n in numbers})
    return ", ".join(str(n) for n in sorted_nums)

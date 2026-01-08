from __future__ import annotations

import json
import random
from typing import Any, Mapping

AGENT_NAME = "BardAgent"

_ADJECTIVES = [
    "bright",
    "calm",
    "crisp",
    "daring",
    "eager",
    "fuzzy",
    "glossy",
    "happy",
    "kind",
    "lively",
    "mellow",
    "nimble",
    "quick",
    "sunny",
    "tidy",
    "vivid",
]

_NOUNS = [
    "otter",
    "panther",
    "sparrow",
    "willow",
    "ember",
    "harbor",
    "meadow",
    "quartz",
    "river",
    "sierra",
    "maple",
    "pine",
    "aurora",
    "comet",
    "breeze",
    "canvas",
]

_USER_AGENT_POOL = [
    # Broad, non-identifying browser UA strings
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    # Lightweight client identifiers
    "python-requests/2.x",
    "curl/8.x",
]


def write_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_content(content: Any) -> str:
    """Convert message content into a displayable string.
    Handles Gemini/LangChain content blocks (list of dicts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(p for p in parts if p.strip())
    return str(content)


def clean_text(text: str) -> str:
    """Lightly clean scraped text: strip, drop duplicate lines, clamp length."""
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # keep single blank lines only
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # drop common noisy strings
        if line.lower().startswith("there was an error while loading"):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    cleaned = "\n".join(lines)
    return cleaned


def random_title() -> str:
    """Generate a simple two-word session title."""
    return f"{random.choice(_ADJECTIVES).title()} {random.choice(_NOUNS).title()}"


def resolve_user_agent(env_value: str | None = None) -> str:
    """
    Return a user agent string.

    Precedence:
    1) Explicit env value (USER_AGENT)
    2) Random pick from a small, non-identifying pool.
    """

    if env_value and env_value.strip():
        return env_value.strip()

    return random.choice(_USER_AGENT_POOL)

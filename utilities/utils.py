from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


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

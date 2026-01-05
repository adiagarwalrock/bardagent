from __future__ import annotations

from typing import Any, Mapping, Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


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


def extract_tool_calls_since_last_user(
    messages: Sequence[AnyMessage],
) -> list[dict[str, Any]]:
    """Collect tool calls made after the most recent user message.
    This matches what a user thinks of as "tools used in this turn"."""
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    tool_calls: list[dict[str, Any]] = []
    for msg in messages[last_human_idx + 1 :]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    return tool_calls

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from utilities.utils import read_json, write_json

HISTORY_FILEPATH: str = "chat_history.json"

# Sample history structure for initial setup
"""
SAMPLE_HISTORY = {
    "history": [
        {
            "id": "123e456",
            "title": "Sample Chat Session",
            "messages": [
                {
                    "role": "human",
                    "content": "Hello, who won the World Series in 2020?",
                    "tt": "2024-10-01T12:00:00Z",
                },
                {
                    "role": "ai",
                    "content": "The Los Angeles Dodgers won the World Series in 2020.",
                    "tt": "2024-10-01T12:00:05Z",
                },
            ],
        },
        {
            "id": "789e012",
            "title": "Another Chat Session",
            "messages": [
                {
                    "role": "human",
                    "content": "What's the capital of France?",
                    "tt": "2024-10-02T15:30:00Z",
                },
                {
                    "role": "ai",
                    "content": "The capital of France is Paris.",
                    "tt": "2024-10-02T15:30:03Z",
                },
            ],
        },
    ]
}
"""

if not os.path.exists(HISTORY_FILEPATH):
    # Create empty history file matching the documented sample shape
    write_json(HISTORY_FILEPATH, {"history": []})


def _load_history() -> Dict[str, Any]:
    """Load history file, guaranteeing a {'history': []} shape."""
    try:
        data = read_json(HISTORY_FILEPATH)
        if not isinstance(data, dict) or "history" not in data:
            # migrate legacy shapes into new structure
            data = {"history": []}
        return data
    except FileNotFoundError:
        return {"history": []}


def save_chat_history(
    session_id: Optional[str],
    history: List[AnyMessage],
    title: Optional[str] = None,
) -> str:
    """Save or append chat history for a session as AI/Human messages.

    Returns the session_id used (helpful when None was passed).
    """
    if session_id is None:
        session_id = str(uuid4())

    data = _load_history()
    sessions: List[Dict[str, Any]] = data.get("history", [])

    serialized = [_message_to_record(msg) for msg in history]

    # Find existing session
    for session in sessions:
        if session.get("id") == session_id:
            session["messages"] = session.get("messages", []) + serialized
            break
    else:
        sessions.append(
            {
                "id": session_id,
                "title": title or "Chat Session",
                "messages": serialized,
            }
        )

    data["history"] = sessions
    write_json(HISTORY_FILEPATH, data)
    return session_id


def read_history(session_id: str) -> List[AnyMessage]:
    """Read the chat history for a given session ID as message objects."""
    data = _load_history()
    for session in data.get("history", []):
        if session.get("id") == session_id:
            return [_record_to_message(r) for r in session.get("messages", [])]
    return []


def get_history_ids() -> List[str]:
    """Get all session IDs with saved chat history."""
    data = _load_history()
    return [s.get("id") for s in data.get("history", []) if s.get("id")]


def clear_history(session_id: str) -> None:
    """Clear the chat history for a given session ID."""
    data = _load_history()
    sessions = [s for s in data.get("history", []) if s.get("id") != session_id]
    data["history"] = sessions
    write_json(HISTORY_FILEPATH, data)


# Helpers --------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _message_to_record(msg: AnyMessage) -> Dict[str, Any]:
    if isinstance(msg, HumanMessage):
        role = "human"
    elif isinstance(msg, AIMessage):
        role = "ai"
    else:
        role = "other"
    sent_time = None
    if hasattr(msg, "additional_kwargs"):
        sent_time = msg.additional_kwargs.get("sent_time") or msg.additional_kwargs.get(
            "tt"
        )
    return {
        "role": role,
        "content": msg.content,
        "sent_time": sent_time or _now_iso(),
    }


def _record_to_message(record: Dict[str, Any]) -> AnyMessage:
    role = record.get("role")
    content = record.get("content", "")
    sent_time = record.get("sent_time") or record.get("tt")
    kwargs = {"sent_time": sent_time} if sent_time else {}

    if role == "human":
        return HumanMessage(content=content, additional_kwargs=kwargs)
    if role == "ai":
        return AIMessage(content=content, additional_kwargs=kwargs)

    # Fallback to AIMessage for unknown roles to keep type consistency
    return AIMessage(content=content, additional_kwargs=kwargs)

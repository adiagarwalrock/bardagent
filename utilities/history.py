import os

from utilities.utils import write_json, read_json
from uuid import uuid4

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage

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
    # Create empty history file
    write_json(HISTORY_FILEPATH, {})


def save_chat_history(session_id: str | None, history: list[dict]) -> None:
    """Save or append the chat history to a JSON file."""
    if session_id is None:
        session_id = str(uuid4())

    try:
        chat_history = read_json(HISTORY_FILEPATH)
    except FileNotFoundError:
        chat_history = {}

    existing_history = chat_history.get(session_id, [])
    chat_history[session_id] = existing_history + history

    write_json(HISTORY_FILEPATH, chat_history)


def read_history(session_id: str) -> list[AnyMessage]:
    """Read the chat history for a given session ID from the JSON file."""
    try:
        chat_history = read_json(HISTORY_FILEPATH)
        return chat_history.get(session_id, [])
    except FileNotFoundError:
        return []


def get_history_ids() -> list[str]:
    """Get all session IDs with saved chat history."""
    try:
        chat_history = read_json(HISTORY_FILEPATH)
        return list(chat_history.keys())
    except FileNotFoundError:
        return []


def clear_history(session_id: str) -> None:
    """Clear the chat history for a given session ID."""
    try:
        chat_history = read_json(HISTORY_FILEPATH)
        if session_id in chat_history:
            del chat_history[session_id]
            write_json(HISTORY_FILEPATH, chat_history)
    except FileNotFoundError:
        pass

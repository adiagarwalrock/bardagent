import logging
import time
from datetime import datetime
from typing import Any, List, Tuple

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from core import get_agent
from utilities.logger import logger as logging
from utilities.prompts import QUERY_MESSAGE_TEMPLATE
from utilities.utils import normalize_content


def run_chat(
    user_message: str, history: List[AnyMessage] | None = None
) -> Tuple[List[AnyMessage], List]:
    """
    Run a chat turn with the agent.

    Args:
        user_message: The user's input message.
        history: The chat history as a list of messages.

    Returns:
        A tuple containing the list of new messages from the agent and a list of tool calls made.

    """

    if not user_message.strip():
        return [AIMessage(content="Please provide a non-empty message.")], []
    if history is None:
        history = []

    agent = get_agent()

    user_msg = QUERY_MESSAGE_TEMPLATE.format(
        query=user_message, current_date=datetime.now().strftime("%Y-%m-%d")
    )

    turn_input: List[AnyMessage] = history + [HumanMessage(content=user_msg)]
    pre_len = len(turn_input)

    state: dict[str, Any] = agent.invoke({"messages": list(turn_input)})

    msgs: list[AnyMessage] = list(state.get("messages") or [])
    delta: list[AnyMessage] = msgs[pre_len:]

    logging.info(f"Agent produced {len(delta)} new messages this turn")

    if not delta:
        logging.warning("Agent returned no new messages; synthesizing fallback reply")
        delta = [AIMessage(content="We ran into an issue generating a response.")]

    if not isinstance(delta[-1], AIMessage):
        logging.warning("Final message is not AIMessage; appending normalized fallback")
        delta.append(
            AIMessage(content=normalize_content(getattr(delta[-1], "content", "")))
        )

    tool_calls: list = [
        {"name": tool.name, "content": tool.content, "status": tool.status}
        for tool in delta
        if isinstance(tool, ToolMessage)
    ]

    return delta, tool_calls


if __name__ == "__main__":

    from uuid import uuid4

    session_id = str(uuid4())

    print(f"Starting bardagent chat session {session_id}. Type 'exit' to quit.")
    print("")

    chat_history: List[AnyMessage] = []
    total_history: List[AnyMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"} or not user_input.strip():
            print("Exiting chat. Goodbye!")
            break

        start = time.perf_counter()
        progress, tool_calls = run_chat(user_input, chat_history)

        # Persist only the user and AI messages in history
        ai_msg = next(
            (m for m in reversed(progress) if isinstance(m, AIMessage)), progress[-1]
        )
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(ai_msg)

        total_history.append(HumanMessage(content=user_input))
        total_history.extend(progress)

        print(
            f"BardAgent (tt: {time.perf_counter() - start:.3f}s) : {normalize_content(ai_msg.content)}"
        )

        print(f"Tools used: {set([tc['name'] for tc in tool_calls])}")

import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_tools
from utilities.logger import logger as logging
from utilities.prompts import AGENT_SYS_MESSAGE, QUERY_MESSAGE_TEMPLATE
from utilities.utils import normalize_content

load_dotenv(".env")

GEMINI_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

_llm = None
_agent = None


@lru_cache(maxsize=1)
def get_model() -> ChatGoogleGenerativeAI:
    """Return a singleton instance of the Gemini chat model."""

    global _llm

    if _llm:
        logging.info("Reusing existing LLM instance")
        return _llm

    logging.info("Creating new LLM instance")
    _llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=GEMINI_API_KEY,
    )
    return _llm


@lru_cache(maxsize=1)
def get_agent(**kwargs):
    """Return a singleton instance of the agent."""

    global _agent

    if _agent:
        logging.info("Reusing existing agent instance")
        return _agent

    logging.info("Creating new agent instance")

    _agent = create_agent(
        get_model(),
        tools=get_tools(),
        name="BardAgent",
        system_prompt=AGENT_SYS_MESSAGE,
        **kwargs,
    )

    return _agent


def run_chat(
    user_message: str, history: List[AnyMessage] | None = None
) -> List[AnyMessage]:
    """Invoke the agent and return only the new messages it produced this turn.

    The caller provides the existing `history`; we append the formatted user
    message for this turn, run the agent, and return the delta (tool messages
    and AI reply). We guarantee the delta ends with an AIMessage so downstream
    printing always has a reply.
    """

    if not user_message.strip():
        return [AIMessage(content="Please provide a non-empty message.")]
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

    return delta


def main():
    print("Hello from bardagent! Run `streamlit run app.py` to launch the UI.")


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
        progress = run_chat(user_input, chat_history)

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

        tool_calls = [tool.name for tool in progress if isinstance(tool, ToolMessage)]
        print(f"Tools used: {tool_calls}")

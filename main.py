import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_tools
from utilities.prompts import AGENT_SYS_MESSAGE, QUERY_MESSAGE_TEMPLATE
from utilities.utils import extract_tool_calls_since_last_user, normalize_content
from utilities.logger import logger as logging

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


def run_chat(user_message: str, history: List[AnyMessage]) -> List[AnyMessage]:
    """Invoke the agent and return the last AIMessage.
    Ensures we always return an AIMessage for downstream printing."""

    agent = get_agent()

    user_msg = QUERY_MESSAGE_TEMPLATE.format(
        query=user_message, current_date=datetime.now().strftime("%Y-%m-%d")
    )
    history.append(HumanMessage(content=user_msg))

    state: dict[str, Any] = agent.invoke({"messages": list(history)})

    msgs: list[AnyMessage] = list(state.get("messages") or [])

    if not msgs or not isinstance(msgs[-1], AIMessage):
        logging.warning("Agent did not return an AIMessage, attempting to fix")

        return [AIMessage(content="We ran into an issue generating a response.")]

    return msgs[len(history) :]


def main():
    print("Hello from bardagent! Run `streamlit run app.py` to launch the UI.")


if __name__ == "__main__":

    history: List[AnyMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"} or not user_input.strip():
            print("Exiting chat. Goodbye!")
            break

        start = time.time()
        progress: List[AnyMessage] = run_chat(user_input, history)
        ai_msg: AIMessage = progress[-1]  # type: ignore

        history.append(HumanMessage(content=user_input))
        history.append(ai_msg)

        print(
            f"BardAgent (tt: {time.time() - start}) : {normalize_content(ai_msg.content)}"
        )
        print(
            f"Tools used: {ai_msg.tool_calls if isinstance(ai_msg, AIMessage) else []}"
        )

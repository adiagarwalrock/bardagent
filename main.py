import logging
import os
import time
from functools import lru_cache
from typing import Any, List, Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_tools
from utils import normalize_content

load_dotenv(".env")

GEMINI_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="bardagent.log",
    filemode="a",
)

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
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return _llm


@lru_cache(maxsize=1)
def get_agent(*args, **kwargs):
    """Return a singleton instance of the agent."""

    global _agent

    if _agent:
        logging.info("Reusing existing agent instance")
        return _agent

    logging.info("Creating new agent instance")

    _agent = create_agent(get_model(), tools=get_tools(), *args, **kwargs)

    return _agent


def run_chat(messages: Sequence[AnyMessage]) -> AIMessage:
    """Convenience helper to invoke the chat model with a list of messages.

    Streamlit imports this to keep LLM setup in one place.
    """

    agent = get_agent()

    response = agent.invoke({"messages": list(messages)})

    # if isinstance(response, AIMessage):
    #     return response
    # Fallback for defensive typing.
    # x = AIMessage(content=str(response))

    # print(x)

    return response.get("messages", {})[-1]


def main():
    print("Hello from bardagent! Run `streamlit run app.py` to launch the UI.")


if __name__ == "__main__":

    history: List[AnyMessage] = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"} or not user_input.strip():
            print("Exiting chat. Goodbye!")
            break
        history.append(HumanMessage(content=user_input))

        start = time.time()
        ai_msg = run_chat(history)
        history.append(ai_msg)

        print(f"BardAgent (tt: {time.time() - start}) : {ai_msg.content}")
        print(f"Tools used: {ai_msg.tool_calls}")

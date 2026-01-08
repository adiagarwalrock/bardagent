import logging
import os
from functools import lru_cache
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import get_tools
from utilities.logger import logger as logging
from utilities.prompts import AGENT_SYS_MESSAGE
from utilities.utils import resolve_user_agent

load_dotenv(".env")

# LangChain warns if USER_AGENT is missing; set it early using env override or a
# generic fallback pool defined in utilities.utils to avoid repo-identifying
# strings.
USER_AGENT = resolve_user_agent(os.getenv("USER_AGENT"))
os.environ["USER_AGENT"] = USER_AGENT

GEMINI_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.5"))

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

_llm_cache: Dict = {}
_agent = None


def get_model(
    model_name: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """
    Return a Gemini chat model.

    Args:
        model_name: Optional override, otherwise env GEMINI_MODEL or default.
        temperature: Optional override, otherwise env GEMINI_TEMPERATURE.
        llm_kwargs: Extra kwargs passed to ChatGoogleGenerativeAI.
    """

    name = model_name or GEMINI_MODEL_NAME
    temp = GEMINI_TEMPERATURE if temperature is None else temperature
    key = (name, temp, tuple(sorted(llm_kwargs.items())))

    if key in _llm_cache:
        logging.info("Reusing LLM instance %s", key)
        return _llm_cache[key]

    logging.info("Creating new LLM instance: model=%s temp=%s", name, temp)
    llm = ChatGoogleGenerativeAI(
        model=name,
        temperature=temp,
        max_tokens=None,
        timeout=12,
        max_retries=2,
        api_key=GEMINI_API_KEY,
        **llm_kwargs,
    )
    _llm_cache[key] = llm
    return llm


@lru_cache(maxsize=1)
def get_agent(
    model_name: str | None = None, temperature: float | None = None, **kwargs
):
    """Return a singleton instance of the agent."""

    global _agent

    if _agent:
        logging.info("Reusing existing agent instance")
        return _agent

    logging.info("Creating new agent instance")

    _agent = create_agent(
        get_model(model_name=model_name, temperature=temperature),
        tools=get_tools(),
        name="BardAgent",
        system_prompt=AGENT_SYS_MESSAGE,
        **kwargs,
    )

    return _agent

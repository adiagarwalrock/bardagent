import logging
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from tools import get_tools
from utilities.logger import logger as logging
from utilities.prompts import AGENT_SYS_MESSAGE

load_dotenv(".env")

GEMINI_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.5"))

if not GEMINI_API_KEY and not OPENAI_API_KEY:
    raise ValueError("AI model API key environment variable not set.")

P_TYPE = Literal["gemini", "openai"]


_llm_cache: Dict = {}
_agent = None


def get_gemini_model(
    model_name: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """
    Return a Gemini chat model.

    Args:
        model_name: Optional override, otherwise env GEMINI_MODEL or default.
        temperature: Optional override, otherwise env MODEL_TEMPERATURE.
        llm_kwargs: Extra kwargs passed to ChatGoogleGenerativeAI.
    """

    name = model_name or GEMINI_MODEL_NAME
    temp = MODEL_TEMPERATURE if temperature is None else temperature
    key = (name, temp, tuple(sorted(llm_kwargs.items())))

    if key in _llm_cache:
        logging.info("Reusing LLM instance %s", key)
        return _llm_cache[key]

    logging.info("Creating new LLM instance: model=%s temp=%s", name, temp)
    llm = ChatGoogleGenerativeAI(
        model=name,
        temperature=temp,
        max_tokens=None,
        timeout=30,
        max_retries=2,
        api_key=GEMINI_API_KEY,
        **llm_kwargs,
    )
    _llm_cache[key] = llm
    return llm


def get_openai_model(
    model_name: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> ChatOpenAI:
    """
    Return an OpenAI chat model.

    Args:
        model_name: Optional override, otherwise env OPENAI_MODEL or default.
        temperature: Optional override, otherwise env OPENAI_TEMPERATURE.
        llm_kwargs: Extra kwargs passed to OpenAI.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    name = model_name or OPENAI_MODEL_NAME
    temp = MODEL_TEMPERATURE if temperature is None else temperature
    key = (name, temp, tuple(sorted(llm_kwargs.items())))

    if key in _llm_cache:
        logging.info("Reusing LLM instance %s", key)
        return _llm_cache[key]

    logging.info("Creating new LLM instance: model=%s temp=%s", name, temp)
    llm = ChatOpenAI(
        model=name,
        temperature=temp,
        max_completion_tokens=None,
        timeout=30,
        max_retries=2,
        api_key=OPENAI_API_KEY,
        **llm_kwargs,
    )
    _llm_cache[key] = llm

    return llm


PROVIDERS_MAP: Dict[P_TYPE, Callable] = {
    "gemini": get_gemini_model,
    "openai": get_openai_model,
}
PROVIDER: str = os.getenv("USE_PROVIDERS", "").lower()

if PROVIDER in PROVIDERS_MAP:
    USE_PROVIDERS = PROVIDER
elif GEMINI_API_KEY:
    USE_PROVIDERS = "gemini"
elif OPENAI_API_KEY:
    USE_PROVIDERS = "openai"
else:
    # Fallback to OpenAI if no keys found (or let it fail later)
    USE_PROVIDERS = "openai"

logging.info(f"Using AI provider: {USE_PROVIDERS}")


def get_model(
    model_name: str | None = None,
    temperature: float | None = None,
    **llm_kwargs: Any,
) -> ChatGoogleGenerativeAI | ChatOpenAI:
    """
    Return a chat model based on the specified model name or provider default.

    Args:
        model_name: Optional override to select the model.
        temperature: Optional override for model temperature.
        llm_kwargs: Extra kwargs passed to the model constructor.
    """

    if USE_PROVIDERS == "openai":
        return get_openai_model(
            model_name=model_name, temperature=temperature, **llm_kwargs
        )
    else:
        return get_gemini_model(
            model_name=model_name, temperature=temperature, **llm_kwargs
        )


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

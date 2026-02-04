import os
from typing import List

from tools.action_tool import math_tool, shell_tool
from tools.info_tools import (
    finance_news_tool,
    internet_search,
    recent_context,
    wikipedia_search,
    yt_search_tool,
)
from tools.multimodal_tools import get_multimodal_tools
from tools.web_fetch import search_and_scrape
from utilities.logger import logger


def get_tools() -> List:
    """Return the list of available tools."""
    tools = []
    # tools.extend(internet_search())  # duckduckgo
    tools.append(wikipedia_search)
    tools.append(recent_context)
    tools.append(search_and_scrape)
    tools.extend(finance_news_tool())
    tools.extend(yt_search_tool())

    if "nt" == os.name:
        logger.warning("Shell tool is only available on Windows systems.")
    else:
        tools.extend(shell_tool())
    tools.extend(math_tool())

    # Add multimodal tools (audio, image, excel, python, youtube)
    tools.extend(get_multimodal_tools())

    logger.info(f"Loaded {len(tools)} tools.")
    return tools

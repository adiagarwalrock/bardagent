from typing import List

from tools.info_tools import (
    finance_news_tool,
    internet_search,
    wikipedia_search,
    yt_search_tool,
)


def get_tools() -> List:
    """Return the list of available tools."""
    tools = []
    tools.extend(internet_search())
    tools.append(wikipedia_search)
    tools.extend(finance_news_tool())
    tools.extend(yt_search_tool())

    return tools

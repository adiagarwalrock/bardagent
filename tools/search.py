from typing import List

from langchain_community.tools import DuckDuckGoSearchRun


def get_search_tools() -> List:
    """Return the list of available search tools."""

    tools = [DuckDuckGoSearchRun()]

    return tools

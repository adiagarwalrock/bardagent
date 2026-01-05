from typing import List

from tools.search import get_search_tools


def get_tools() -> List:
    """Return the list of available tools."""
    tools = []
    tools.extend(get_search_tools())
    return tools

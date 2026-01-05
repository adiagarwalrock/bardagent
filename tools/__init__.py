from typing import List

from tools.search import get_search_tools, wikipedia_search


def get_tools() -> List:
    """Return the list of available tools."""
    tools = []
    tools.extend(get_search_tools())
    tools.append(wikipedia_search)
    return tools

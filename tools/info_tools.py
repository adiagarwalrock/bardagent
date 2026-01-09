from datetime import datetime
from typing import List

from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from utilities.logger import logger


def internet_search() -> List:
    """Return the list of available search tools."""
    from langchain_community.tools import DuckDuckGoSearchRun

    ddg = DuckDuckGoSearchRun()
    ddg.name = "duckduckgo"
    ddg.description = (
        "Use for fresh/time-sensitive queries; include source + date in answers. "
        "Prefer wikipedia_search first for stable facts; use DuckDuckGo to double-check recency."
    )
    tools = [ddg]

    return tools


def finance_news_tool() -> List:
    """Return the Yahoo Finance News Tool."""
    from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

    return [YahooFinanceNewsTool()]


def yt_search_tool() -> List:
    """Return the YouTube Search Tool."""
    from langchain_community.tools import YouTubeSearchTool

    return [YouTubeSearchTool()]


def nasa_tool() -> List: ...


_RECENT_SNIPPETS: List[str] = []


def _add_recent_snippet(snippet: str) -> None:
    _RECENT_SNIPPETS.append(snippet)
    # Keep the buffer small and recency-biased
    if len(_RECENT_SNIPPETS) > 12:
        del _RECENT_SNIPPETS[0 : len(_RECENT_SNIPPETS) - 12]


class RecentContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., description="Number of recent snippets to return. Default 3.")


@tool(args_schema=RecentContextArgs)
def recent_context(n: int = 3) -> str:
    """
    Return the freshest snippets captured from recent tool calls (max n).
    Use to ground answers with recency. Falls back with guidance if empty.
    """

    if not _RECENT_SNIPPETS:
        return "No recent snippets available. Run a search tool first."

    n = max(1, min(n, 5))
    latest = _RECENT_SNIPPETS[-n:][::-1]  # newest first
    return "\n---\n".join(latest)


class WikipediaSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., description="Wikipedia search query")


@tool(args_schema=WikipediaSearchArgs)
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information. Use this for factual queries.
    Use first for stable facts; include title + year/location if present.

    Args:
        query: Wikipedia search query

    Returns:
        Wikipedia article summary
    """
    import wikipedia

    try:
        logger.debug(f"Searching Wikipedia for: {query}")

        # Search for the query
        results = wikipedia.search(query, results=3)  # Get top 3 results

        if not results:
            logger.warning(f"No Wikipedia search results for: {query}")
            return f"No Wikipedia articles found for: {query}"

        logger.debug(f"Wikipedia search results: {results}")

        # Try each result until we find one that works
        for result_title in results:
            try:
                logger.debug(f"Attempting to fetch summary for: {result_title}")
                summary = wikipedia.summary(
                    result_title, sentences=5, auto_suggest=False
                )
                logger.info(
                    f"Successfully retrieved Wikipedia summary for: {result_title}"
                )
                stamped = f"{result_title} (retrieved {datetime.utcnow().isoformat()} UTC):\n{summary}"
                _add_recent_snippet(stamped)
                return stamped
            except wikipedia.exceptions.PageError:
                logger.debug(f"PageError for {result_title}, trying next result")
                continue
            except wikipedia.exceptions.DisambiguationError as e:
                logger.debug(
                    f"DisambiguationError for {result_title}, trying first option"
                )
                # Try the first disambiguation option
                if e.options:
                    try:
                        summary = wikipedia.summary(
                            e.options[0], sentences=5, auto_suggest=False
                        )
                        logger.info(
                            f"Retrieved disambiguated summary for: {e.options[0]}"
                        )
                        stamped = f"{e.options[0]} (retrieved {datetime.utcnow().isoformat()} UTC):\n{summary}"
                        _add_recent_snippet(stamped)
                        return stamped
                    except:
                        continue

        # If we exhausted all results
        logger.warning(
            f"Failed to retrieve summary for any Wikipedia result for: {query}"
        )
        return f"No Wikipedia page found for: {query}"

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle top-level disambiguation
        options = ", ".join(e.options[:5])
        logger.debug(f"Top-level disambiguation for '{query}': {options}")
        return f"Multiple results found for '{query}'. Please be more specific. Options: {options}"
    except Exception as e:
        logger.error(
            f"Unexpected error searching Wikipedia for '{query}': {e}", exc_info=True
        )
        return f"Error searching Wikipedia: {str(e)}"

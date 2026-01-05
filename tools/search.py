from typing import List

import wikipedia
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from utilities.logger import logger


def get_search_tools() -> List:
    """Return the list of available search tools."""

    tools = [DuckDuckGoSearchRun()]

    return tools


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information. Use this for factual queries.

    Args:
        query: Wikipedia search query

    Returns:
        Wikipedia article summary
    """
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
                return f"{result_title}:\n{summary}"
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
                        return f"{e.options[0]}:\n{summary}"
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

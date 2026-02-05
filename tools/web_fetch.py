"""Headless search-and-scrape helpers using Playwright."""

import sys

sys.path.append(".")

from typing import List, Optional, TypedDict

from ddgs import DDGS
from langchain.tools import tool
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict, Field

from utilities.logger import logger
from utilities.utils import clean_text


class SearchState(TypedDict, total=False):
    query: str
    max_results: int
    selector: Optional[str]
    wait_for: Optional[str]
    timeout: int
    urls: List[str]
    snippets: List[str]


def _search_urls(query: str, max_results: int) -> List[str]:
    """Return top search result URLs from DuckDuckGo (text API)."""
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=max_results))
    return [h.get("href") for h in hits if h.get("href")]


def _fetch_with_requests(url: str) -> str | None:
    """Simple HTTP fetch using requests - works without Playwright."""
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        resp.raise_for_status()

        # Check for non-HTML content
        ctype = resp.headers.get("content-type", "")
        if "application/pdf" in ctype:
            return f"{url}\nDownloadable PDF detected; cannot render."

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        return clean_text(text[:15000])  # Limit length
    except Exception as e:
        logger.debug(f"requests fetch failed for {url}: {e}")
        return None


def _fetch_with_playwright(
    url: str, selector: Optional[str], wait_for: Optional[str], timeout: int
) -> str:
    # Use a realistic User-Agent to avoid bot detection and protocol errors
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )

    try:
        from bs4 import BeautifulSoup
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import error surfaced to user
        # Playwright not available - try requests fallback
        logger.warning("Playwright not available, trying requests fallback")
        result = _fetch_with_requests(url)
        if result:
            return result
        return f"{url}\nPlaywright/bs4 not available: {exc}"

    html = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                if wait_for:
                    try:
                        page.wait_for_selector(wait_for, timeout=timeout)
                    except Exception:
                        logger.warning(
                            "wait_for selector not found; continuing", exc_info=True
                        )
                html = page.content()
            except Exception as nav_exc:
                error_msg = str(nav_exc)
                # Fallback for downloads, protocol errors, or general navigation failures.
                # Playwright is often more sensitive to protocol/SSL issues than a simple GET.
                logger.warning(
                    f"Playwright navigation failed for {url}: {error_msg}. Attempting fallback with requests."
                )
                try:
                    import requests

                    # Use a slightly longer timeout for the fallback if the original was short
                    fallback_timeout = max(timeout / 1000, 15)

                    resp = requests.get(
                        url,
                        timeout=fallback_timeout,
                        headers={
                            "User-Agent": USER_AGENT,
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.5",
                            "Accept-Encoding": "gzip, deflate, br",
                            "DNT": "1",
                            "Connection": "keep-alive",
                            "Upgrade-Insecure-Requests": "1",
                        },
                        verify=True,  # Keep it secure but standard
                    )
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "application/pdf" in ctype:
                        return f"{url}\nDownloadable PDF detected; skipping rendering. (content-type: {ctype})"

                    html = resp.text
                except Exception as fallback_exc:
                    logger.error(
                        f"Fallback fetch for {url} failed: {fallback_exc}",
                        exc_info=True,
                    )
                    return (
                        f"{url}\nPlaywright navigation failed: {nav_exc}\n"
                        f"Fallback fetch also failed: {fallback_exc}"
                    )

            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        if selector:
            nodes = soup.select(selector)
            if not nodes:
                return f"{url}\n(no elements matched selector '{selector}')"
            text = "\n\n".join(node.get_text(" ", strip=True) for node in nodes)
        else:
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(" ", strip=True) if main else ""

        text = clean_text(text)
        return f"{url}\n{text}"
    except Exception as exc:  # pragma: no cover - runtime errors surfaced to user
        logger.error("Playwright fetch failed", exc_info=True)
        return f"{url}\nError fetching page: {exc}"


def _build_search_scrape_graph():
    """Build a tiny LangGraph pipeline: Search -> Fetch."""

    def search_node(state: SearchState) -> SearchState:
        query = state["query"]
        max_results = state.get("max_results", 3)
        urls = _search_urls(query, max_results)
        return {"urls": urls}

    def fetch_node(state: SearchState) -> SearchState:
        selector = state.get("selector")
        wait_for = state.get("wait_for")
        timeout = state.get("timeout", 12000)
        urls: List[str] = state.get("urls", [])
        snippets = [
            _fetch_with_playwright(
                url, selector=selector, wait_for=wait_for, timeout=timeout
            )
            for url in urls
        ]
        return {"snippets": snippets}

    graph: StateGraph[SearchState] = StateGraph(SearchState)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "fetch")
    graph.set_finish_point("fetch")

    return graph.compile()


_SEARCH_SCRAPE_GRAPH = _build_search_scrape_graph()


class SearchAndScrapeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., description="Search query")
    max_results: int = Field(..., description="Max results (default 3)")
    selector: Optional[str] = Field(
        ..., description="CSS selector (optional, send null if none)"
    )
    wait_for: Optional[str] = Field(
        ..., description="Wait for selector (optional, send null if none)"
    )
    timeout_ms: int = Field(..., description="Timeout in ms (default 12000)")


@tool(args_schema=SearchAndScrapeArgs)
def search_and_scrape(
    query: str,
    max_results: int = 3,
    selector: Optional[str] = None,
    wait_for: Optional[str] = None,
    timeout_ms: int = 12000,
) -> str:
    """
    Search DuckDuckGo for a query, then scrape top results with headless Chromium.
    Implemented via a LangGraph pipeline (Search -> Fetch).
    """

    state: SearchState = {
        "query": query,
        "max_results": max_results,
        "selector": selector,
        "wait_for": wait_for,
        "timeout": max(timeout_ms, 1000),
    }

    result = _SEARCH_SCRAPE_GRAPH.invoke(state)
    snippets = result.get("snippets") or []
    if not snippets:
        return "No content returned from search_and_scrape."
    return "\n\n---\n\n".join(snippets)


if __name__ == "__main__":
    # Simple test run
    state: SearchState = {
        "query": "nobel laureates in physics 2023",
        "max_results": 2,
        "selector": "p",
        "wait_for": "body",
        "timeout": 10000,
    }

    result = _SEARCH_SCRAPE_GRAPH.invoke(state)
    snippets = result.get("snippets") or []
    output = (
        "\n\n---\n\n".join(snippets)
        if snippets
        else "No content returned from search_and_scrape."
    )

    print(output)

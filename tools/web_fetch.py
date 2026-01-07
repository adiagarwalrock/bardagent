"""Headless search-and-scrape helpers using Playwright."""

import sys

sys.path.append(".")

from typing import List, Optional, TypedDict

from ddgs import DDGS
from langchain.tools import tool
from langgraph.graph import StateGraph

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


def _fetch_with_playwright(
    url: str, selector: Optional[str], wait_for: Optional[str], timeout: int
) -> str:
    try:
        from bs4 import BeautifulSoup
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import error surfaced to user
        logger.error("Required packages missing for Playwright fetch", exc_info=True)
        return f"{url}\nPlaywright/bs4 not available: {exc}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            except Exception as nav_exc:
                # Some URLs (e.g., PDFs) trigger a download and Playwright raises.
                if "Download is starting" in str(nav_exc):
                    try:
                        import requests

                        resp = requests.get(
                            url,
                            timeout=max(timeout / 1000, 5),
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        ctype = resp.headers.get("content-type", "")
                        if "application/pdf" in ctype:
                            return f"{url}\nDownloadable PDF detected; skipping Playwright rendering. (content-type: {ctype})"
                        return (
                            f"{url}\nDownloaded content (truncated): {resp.text[:4000]}"
                        )
                    except Exception as dl_exc:
                        logger.error(
                            "Fallback download after Playwright download trigger failed",
                            exc_info=True,
                        )
                        return (
                            f"{url}\nDownload started; fallback fetch failed: {dl_exc}"
                        )
                raise

            if wait_for:
                try:
                    page.wait_for_selector(wait_for, timeout=timeout)
                except Exception:
                    logger.warning(
                        "wait_for selector not found; continuing", exc_info=True
                    )

            html = page.content()
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


@tool
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

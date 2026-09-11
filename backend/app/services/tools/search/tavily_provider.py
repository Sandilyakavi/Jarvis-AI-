"""
Search Tool — Tavily Provider

Concrete implementation of BaseSearchProvider using the Tavily Search API.
Provides high-accuracy, AI-optimized web search results using direct HTTP requests.

Security & Design:
- Reads TAVILY_API_KEY from configuration/environment.
- Never hardcodes or logs API keys.
- Lightweight async HTTP using httpx (no external proprietary SDK required).
- Adheres to the BaseSearchProvider interface returning standard SearchResult objects.
"""

import logging
from typing import List, Optional
import httpx

from app.config import settings
from app.services.tools.search.search_provider import BaseSearchProvider, SearchResult

logger = logging.getLogger("jarvis.services.tools.search.tavily")


class TavilyProvider(BaseSearchProvider):
    """
    Search provider backed by Tavily Search API.

    Characteristics:
    - AI-native search engine designed for LLM context retrieval.
    - Requires TAVILY_API_KEY environment variable.
    - Results include title, URL, and AI-optimized content snippet.
    """

    TAVILY_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        """
        Initialize the Tavily search provider.

        Args:
            api_key: Optional explicit API key. Defaults to settings.TAVILY_API_KEY.
            timeout_seconds: Optional timeout in seconds. Defaults to settings.TOOL_TIMEOUT_SECONDS.
        """
        self._api_key = (
            api_key if api_key is not None else getattr(settings, "TAVILY_API_KEY", "")
        )
        self._timeout_seconds = (
            timeout_seconds or getattr(settings, "TOOL_TIMEOUT_SECONDS", 10)
        )

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Execute an asynchronous search request to the Tavily API.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (capped at 10).

        Returns:
            List of SearchResult objects.

        Raises:
            ValueError: If TAVILY_API_KEY is not configured.
            TimeoutError: If the request times out.
            RuntimeError: On HTTP or network communication failures.
        """
        if not self._api_key or not self._api_key.strip():
            logger.error("[TavilyProvider] Search attempted without configured TAVILY_API_KEY.")
            raise ValueError(
                "Tavily API key is not configured. Please set TAVILY_API_KEY in the environment."
            )

        effective_max = min(max(1, max_results), 10)
        logger.debug(f"[TavilyProvider] Searching: '{query}' (max={effective_max})")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": effective_max,
            "search_depth": "basic",
            "include_answer": False,
        }

        try:
            async with httpx.AsyncClient(timeout=float(self._timeout_seconds)) as client:
                response = await client.post(
                    self.TAVILY_API_URL,
                    json=payload,
                )

            if response.status_code == 401 or response.status_code == 403:
                logger.error("[TavilyProvider] Authentication failed (401/403).")
                raise RuntimeError("Tavily authentication failed: invalid API credentials.")

            if response.status_code == 429:
                logger.error("[TavilyProvider] Rate limit exceeded (429).")
                raise RuntimeError("Tavily API rate limit exceeded.")

            if response.status_code != 200:
                logger.error(
                    f"[TavilyProvider] Search request failed with status HTTP {response.status_code}."
                )
                raise RuntimeError(
                    f"Tavily search API returned HTTP error status: {response.status_code}"
                )

            data = response.json()
            raw_results = data.get("results", [])

            if not isinstance(raw_results, list):
                logger.warning("[TavilyProvider] Unexpected results format received from Tavily API.")
                return []

            results: List[SearchResult] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                title = item.get("title") or "No title"
                url = item.get("url") or ""
                snippet = (
                    item.get("content")
                    or item.get("snippet")
                    or "No description available."
                )
                results.append(
                    SearchResult(
                        title=title.strip(),
                        url=url.strip(),
                        snippet=snippet.strip(),
                    )
                )

            logger.info(
                f"[TavilyProvider] Search complete: '{query}' → {len(results)} results"
            )
            return results

        except httpx.TimeoutException as exc:
            logger.error(f"[TavilyProvider] Search timed out for query '{query}': {exc}")
            raise TimeoutError("Tavily search request timed out.") from exc

        except httpx.RequestError as exc:
            logger.error(f"[TavilyProvider] Network error during Tavily search: {exc.__class__.__name__}")
            raise RuntimeError(
                f"Network error communicating with Tavily search API: {exc.__class__.__name__}"
            ) from exc

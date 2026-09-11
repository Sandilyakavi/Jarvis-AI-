"""
Search Tool — DuckDuckGo Provider

Concrete implementation of BaseSearchProvider using the duckduckgo-search
library. Requires NO API key, making it the ideal default backend for
development and initial production use.

Swap strategy: To replace with Tavily, Serper, or Brave, create a new
file (e.g., tavily_provider.py) implementing BaseSearchProvider, then
update the SearchTool instantiation in search_tool.py — nothing else changes.
"""

import logging
from typing import List

from app.services.tools.search.search_provider import BaseSearchProvider, SearchResult

logger = logging.getLogger("jarvis.services.tools.search.duckduckgo")


class DuckDuckGoProvider(BaseSearchProvider):
    """
    Search provider backed by DuckDuckGo's anonymous search API via the
    `duckduckgo-search` library.

    Characteristics:
    - No API key or account required.
    - Rate-limited by DuckDuckGo on high volume (suitable for personal AI use).
    - Results include title, URL, and body snippet.
    """

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform a DuckDuckGo text search.

        Uses run_in_executor to run the synchronous duckduckgo-search
        library in a thread pool, keeping the async event loop unblocked.

        Args:
            query:       The search query string.
            max_results: Maximum number of results to return (capped at 10).

        Returns:
            List of SearchResult objects.

        Raises:
            Exception: On network failure or DuckDuckGo API unavailability.
        """
        import asyncio
        # pyrefly: ignore [missing-import]
        from duckduckgo_search import DDGS

        effective_max = min(max_results, 10)  # Safety cap

        logger.debug(f"[DuckDuckGoProvider] Searching: '{query}' (max={effective_max})")

        def _sync_search() -> List[SearchResult]:
            """Synchronous search run in a thread pool executor."""
            results: List[SearchResult] = []
            with DDGS() as ddgs:
                raw_results = ddgs.text(
                    query,
                    max_results=effective_max,
                )
                for item in raw_results:
                    results.append(
                        SearchResult(
                            title=item.get("title", "No title"),
                            url=item.get("href", ""),
                            snippet=item.get("body", "No description available."),
                        )
                    )
            return results

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _sync_search)

        logger.info(
            f"[DuckDuckGoProvider] Search complete: '{query}' → {len(results)} results"
        )
        return results

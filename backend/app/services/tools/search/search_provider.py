"""
Search Tool — Abstract Search Provider Interface

Defines the contract that any search backend must implement.
This decouples SearchTool from any specific search service,
making the backend swappable without touching SearchTool itself.

Current implementation: DuckDuckGo (no API key required)
Future options:         Tavily, Serper, Brave Search, SerpAPI
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """
    A single search result returned by a search provider.

    Attributes:
        title:   The page/article title.
        url:     The canonical URL of the result.
        snippet: A short text excerpt summarising the result's content.
    """
    title: str
    url: str
    snippet: str

    def to_formatted_string(self) -> str:
        """Returns a clean, single-line summary suitable for AI context injection."""
        return f"**{self.title}**\n{self.snippet}\nSource: {self.url}"


class BaseSearchProvider(ABC):
    """
    Abstract interface for pluggable search backends.

    Any class implementing this interface can be injected into SearchTool
    as the active search backend. No other code changes are required.
    """

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """
        Execute a web search and return a list of results.

        Args:
            query:       The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects, ordered by relevance.

        Raises:
            Exception: Providers may raise on network failure; SearchTool
                       catches and wraps these into a ToolResult error.
        """
        ...

"""
Search Tool — SearchTool Implementation

SearchTool is the first concrete tool registered in JARVIS.
It exposes a clean, provider-agnostic web search capability to the AI.

Architecture:
  SearchTool(BaseTool)
        │
        └── BaseSearchProvider  ← swappable backend
              └── DuckDuckGoProvider (default)

To replace the backend:
  1. Create a new file implementing BaseSearchProvider.
  2. Change the default_provider parameter below.
  Nothing else changes.
"""

import logging
from typing import Any, Optional

from app.config import settings
from app.services.tools.base_tool import (
    BaseTool,
    ParameterSchema,
    ToolMetadata,
    ToolResult,
)
from app.services.tools.exceptions import ToolExecutionError, ToolValidationError
from app.services.tools.search.search_provider import BaseSearchProvider

logger = logging.getLogger("jarvis.services.tools.search.search_tool")


class SearchTool(BaseTool):
    """
    JARVIS Web Search Tool.

    Provides the AI with the ability to search the web and retrieve
    current information beyond its training data cutoff.

    Registered name: "web_search"
    """

    # Maximum results the tool will ever request from the provider
    _MAX_RESULTS_HARD_CAP = 10

    def __init__(self, provider: Optional[BaseSearchProvider] = None) -> None:
        """
        Args:
            provider: A concrete BaseSearchProvider instance.
                      Defaults to DuckDuckGoProvider if not supplied.
                      Injecting a custom provider is useful for testing.
        """
        if provider is None:
            # Lazy import to avoid circular issues and allow provider swap
            from app.services.tools.search.duckduckgo_provider import DuckDuckGoProvider
            provider = DuckDuckGoProvider()

        self._provider: BaseSearchProvider = provider
        self._max_results: int = min(
            getattr(settings, "SEARCH_MAX_RESULTS", 5),
            self._MAX_RESULTS_HARD_CAP,
        )

    # ------------------------------------------------------------------
    # BaseTool Interface
    # ------------------------------------------------------------------

    def metadata(self) -> ToolMetadata:
        """
        Describe this tool to the AI model.

        The returned schema is converted by ToolRegistry.get_function_schemas()
        into the format expected by Groq / OpenAI function-calling APIs.
        """
        return ToolMetadata(
            name="web_search",
            description=(
                "Search the web for current information, news, facts, or any topic "
                "that may require up-to-date knowledge beyond the AI's training data. "
                "Use this when the user asks about recent events, specific facts, or "
                "anything that benefits from real-time web results."
            ),
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description=(
                        "The search query to look up on the web. "
                        "Be specific and concise for best results."
                    ),
                    required=True,
                ),
                ParameterSchema(
                    name="max_results",
                    type="integer",
                    description=(
                        f"Maximum number of results to return (1-{self._MAX_RESULTS_HARD_CAP}). "
                        f"Defaults to {self._max_results}."
                    ),
                    required=False,
                ),
            ],
            version="1.0.0",
        )

    def validate(self, **kwargs: Any) -> bool:
        """
        Validate input arguments before executing the search.

        Raises:
            ToolValidationError: If `query` is missing, not a string, or empty.
        """
        query = kwargs.get("query")

        if query is None:
            raise ToolValidationError(
                tool_name="web_search",
                detail="The 'query' parameter is required but was not provided.",
            )

        if not isinstance(query, str):
            raise ToolValidationError(
                tool_name="web_search",
                detail=f"The 'query' parameter must be a string, got {type(query).__name__}.",
            )

        if not query.strip():
            raise ToolValidationError(
                tool_name="web_search",
                detail="The 'query' parameter must not be empty or whitespace.",
            )

        max_results = kwargs.get("max_results")
        if max_results is not None:
            if not isinstance(max_results, int) or max_results < 1:
                raise ToolValidationError(
                    tool_name="web_search",
                    detail=f"'max_results' must be a positive integer, got: {max_results}.",
                )

        return True

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute a web search and return formatted results.

        Args:
            query (str):        The search query.
            max_results (int):  Optional override for result count.

        Returns:
            ToolResult with:
            - data: list of SearchResult dicts
            - formatted_output: clean, numbered list ready for AI context
        """
        query: str = kwargs["query"].strip()
        max_results: int = int(kwargs.get("max_results", self._max_results))
        max_results = min(max_results, self._MAX_RESULTS_HARD_CAP)

        logger.info(f"[SearchTool] Executing search: '{query}' (max={max_results})")

        try:
            results = await self._provider.search(query=query, max_results=max_results)
        except Exception as exc:
            logger.error(
                f"[SearchTool] Search provider failed for query '{query}': {exc}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                tool_name="web_search",
                error_message=f"Search service temporarily unavailable: {str(exc)}",
                error_code="SEARCH_PROVIDER_ERROR",
                formatted_output=(
                    "I was unable to complete the web search at this time. "
                    "The search service may be temporarily unavailable. "
                    "Please try again in a moment, or I can answer based on my training data."
                ),
            )

        if not results:
            logger.info(f"[SearchTool] No results found for query: '{query}'")
            return ToolResult(
                success=True,
                tool_name="web_search",
                data=[],
                formatted_output=(
                    f"No web results were found for the query: '{query}'. "
                    f"This may be a very niche topic or the query may need to be rephrased."
                ),
            )

        # Format results into clean, numbered text for AI context injection
        formatted_lines = [
            f"Web search results for: '{query}'\n"
        ]
        result_data = []

        for i, result in enumerate(results, start=1):
            formatted_lines.append(
                f"{i}. **{result.title}**\n"
                f"   {result.snippet}\n"
                f"   Source: {result.url}\n"
            )
            result_data.append(
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                }
            )

        formatted_output = "\n".join(formatted_lines)

        logger.info(
            f"[SearchTool] Search successful: '{query}' → {len(results)} results"
        )

        return ToolResult(
            success=True,
            tool_name="web_search",
            data=result_data,
            formatted_output=formatted_output,
        )

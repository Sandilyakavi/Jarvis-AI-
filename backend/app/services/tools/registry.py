"""
Tool Calling Framework — Tool Registry

ToolRegistry is the central catalogue of all tools available to JARVIS.
It mirrors the design of ProviderRegistry and follows the open-closed principle:
new tools are registered here and become immediately available to ToolManager
without any other code changes.

Usage:
    # Register a tool (done once at startup / in search/__init__.py)
    ToolRegistry.register("web_search", SearchTool)

    # Retrieve a singleton instance
    tool = ToolRegistry.get_tool("web_search")

    # List all registered tools (for /api/tools endpoint or AI schema building)
    tools = ToolRegistry.list_tools()

    # Get OpenAI/Groq-compatible function schemas for ALL registered tools
    schemas = ToolRegistry.get_function_schemas()
"""

import logging
from typing import Dict, List, Type

from app.services.tools.base_tool import BaseTool, ToolMetadata
from app.services.tools.exceptions import ToolNotFoundError

logger = logging.getLogger("jarvis.services.tools.registry")


class ToolRegistry:
    """
    Central registry that catalogues and lazily instantiates JARVIS tools.

    Class-level storage means the registry is effectively a singleton
    across the application lifetime, consistent with ProviderRegistry.
    """

    _tools: Dict[str, Type[BaseTool]] = {}
    _instances: Dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, tool_class: Type[BaseTool]) -> None:
        """
        Register a tool class under the given name.

        The name is normalised to lowercase and must match the value returned
        by tool_class().metadata().name for consistency.

        Args:
            name:       Unique identifier for the tool (e.g. "web_search").
            tool_class: Concrete subclass of BaseTool to register.
        """
        normalised = name.lower()
        cls._tools[normalised] = tool_class
        # Invalidate any stale singleton instance
        cls._instances.pop(normalised, None)
        logger.info(f"[ToolRegistry] Registered tool: '{normalised}' → {tool_class.__name__}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        """
        Retrieve (or lazily instantiate) the tool registered under `name`.

        Singleton pattern: each tool class is instantiated once and reused
        across requests, which allows tools to maintain lightweight state
        (e.g., an HTTP client session) if needed.

        Args:
            name: Tool identifier, case-insensitive.

        Returns:
            An instantiated BaseTool subclass.

        Raises:
            ToolNotFoundError: If `name` has not been registered.
        """
        normalised = name.lower()

        # Return cached instance
        if normalised in cls._instances:
            return cls._instances[normalised]

        # Validate registration
        if normalised not in cls._tools:
            logger.error(f"[ToolRegistry] Tool '{normalised}' is not registered.")
            raise ToolNotFoundError(normalised)

        # Instantiate and cache
        tool_class = cls._tools[normalised]
        instance = tool_class()
        cls._instances[normalised] = instance
        logger.debug(f"[ToolRegistry] Instantiated tool: '{normalised}'")
        return instance

    # ------------------------------------------------------------------
    # Discovery / Introspection
    # ------------------------------------------------------------------

    @classmethod
    def list_tools(cls) -> List[ToolMetadata]:
        """
        Return metadata for every registered tool.

        Used by:
        - A future GET /api/tools endpoint to expose available tools to clients.
        - ToolManager when building the function schema list for the AI provider.
        """
        metadatas: List[ToolMetadata] = []
        for name in cls._tools:
            tool = cls.get_tool(name)
            metadatas.append(tool.metadata())
        return metadatas

    @classmethod
    def get_function_schemas(cls) -> List[dict]:
        """
        Return the OpenAI / Groq-compatible function call schemas for ALL
        registered tools. This list is passed directly to the provider's
        `tools` parameter when making a chat completion request.

        Returns:
            List of function schema dicts in the format:
            [{"type": "function", "function": {...}}, ...]
        """
        schemas = []
        for metadata in cls.list_tools():
            schemas.append(metadata.to_function_schema())
        return schemas

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether a tool name is currently registered."""
        return name.lower() in cls._tools

    @classmethod
    def registered_names(cls) -> List[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(cls._tools.keys())

    @classmethod
    def clear(cls) -> None:
        """
        Reset the registry. Intended for use in tests only.
        Clears both class-level dicts so tests start with a clean slate.
        """
        cls._tools.clear()
        cls._instances.clear()
        logger.debug("[ToolRegistry] Registry cleared (test mode).")

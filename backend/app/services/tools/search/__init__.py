"""
Search Tool Package — Initializer

Registers SearchTool with the global ToolRegistry on import.
This file is imported once during application startup, ensuring
the search tool is available before any requests are processed.
"""

from app.services.tools.registry import ToolRegistry
from app.services.tools.search.search_tool import SearchTool

# Auto-register the search tool — adding a new tool means creating a new
# package and adding a single ToolRegistry.register() call here or in main.py
ToolRegistry.register("web_search", SearchTool)

__all__ = ["SearchTool"]

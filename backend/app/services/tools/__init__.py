"""
Tool Calling Framework — Package Initializer

Exposes the primary public surface of the tools module.
"""

from app.services.tools.base_tool import BaseTool, ToolMetadata, ToolResult, ParameterSchema
from app.services.tools.registry import ToolRegistry
from app.services.tools.manager import ToolManager
from app.services.tools.exceptions import (
    ToolException,
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
    ToolTimeoutError,
)

__all__ = [
    # Core interface
    "BaseTool",
    "ToolMetadata",
    "ToolResult",
    "ParameterSchema",
    # Infrastructure
    "ToolRegistry",
    "ToolManager",
    # Exceptions
    "ToolException",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolTimeoutError",
]

"""
Tool Calling Framework — Base Tool Interface

Every tool in JARVIS must subclass BaseTool and implement three methods:
    - metadata()  → Describes the tool to the AI model (name, schema, etc.)
    - validate()  → Checks incoming arguments before execution
    - execute()   → Performs the actual work and returns a ToolResult

ToolMetadata is used to register the tool's function schema with AI providers
that support function/tool calling (e.g. Groq, OpenAI).

ToolResult is the standardised return envelope from every tool execution,
carrying success status, data payload, error context, and timing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class ParameterSchema:
    """
    Describes a single parameter accepted by a tool.
    Maps directly to the JSON Schema 'properties' format expected by AI APIs.
    """
    name: str
    type: str                          # "string" | "integer" | "number" | "boolean"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None   # Allowed values, if constrained


@dataclass
class ToolMetadata:
    """
    Full description of a tool, consumable by AI model APIs.

    The `parameters` list is auto-converted to JSON Schema format by
    ToolRegistry when building the function-call definitions sent to providers.
    """
    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    version: str = "1.0.0"

    def to_function_schema(self) -> Dict[str, Any]:
        """
        Serialises this metadata into the standard OpenAI / Groq function schema:

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { "type": "object", "properties": {...}, "required": [...] }
            }
        }
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    """
    Standardised return envelope from every tool execution.

    Attributes:
        success:          True if the tool completed without errors.
        data:             The raw result payload (tool-specific structure).
        formatted_output: Human-readable string, safe to inject into AI context.
        error_message:    User-friendly error string if success=False.
        error_code:       Machine-readable code if success=False.
        execution_time_ms: Wall-clock execution time in milliseconds.
        tool_name:        Name of the tool that produced this result.
    """
    success: bool
    tool_name: str
    data: Optional[Any] = None
    formatted_output: str = ""
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """
    Abstract base class that every JARVIS tool must implement.

    The contract guarantees that:
    1. ToolManager can call any tool through a uniform interface.
    2. Tools are self-describing via metadata(), enabling automatic
       registration with AI provider function-call schemas.
    3. Validation is always run before execution.

    Subclassing example:
        class MyTool(BaseTool):
            def metadata(self) -> ToolMetadata: ...
            def validate(self, **kwargs) -> bool: ...
            async def execute(self, **kwargs) -> ToolResult: ...
    """

    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """
        Return the tool's self-description.

        Called by ToolRegistry to:
        - Build the function schema list sent to AI providers.
        - Display tool info via the /tools list endpoint.
        """
        ...

    @abstractmethod
    def validate(self, **kwargs: Any) -> bool:
        """
        Validate incoming keyword arguments before execution.

        Should raise ToolValidationError (not return False) so that the
        ToolManager can surface a meaningful error to the caller.

        Returns True if the arguments are valid.
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool's primary action.

        Must ALWAYS return a ToolResult — never raise an unhandled exception.
        Wrap all internal failures in a ToolResult(success=False, ...).

        Args:
            **kwargs: Tool-specific arguments, validated beforehand.

        Returns:
            ToolResult with success status, data payload, and human-readable output.
        """
        ...

    @property
    def name(self) -> str:
        """Convenience accessor — returns the tool's registered name."""
        return self.metadata().name

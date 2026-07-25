"""
Tool Calling Framework — Exception Hierarchy

All tool-related errors descend from ToolException so callers can catch them
with a single except clause while still being able to distinguish error types.
"""


class ToolException(Exception):
    """
    Base exception for all Tool Framework errors.

    Attributes:
        message:    Human-readable description.
        error_code: Machine-readable code for structured error responses.
        status_code: HTTP status hint (used when surfaced through the API).
    """

    def __init__(
        self,
        message: str,
        error_code: str = "TOOL_ERROR",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class ToolNotFoundError(ToolException):
    """Raised when a requested tool name is not registered in the ToolRegistry."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' is not registered. "
                    f"Register it with ToolRegistry.register() before use.",
            error_code="TOOL_NOT_FOUND",
            status_code=404,
        )
        self.tool_name = tool_name


class ToolValidationError(ToolException):
    """Raised when the arguments passed to a tool fail validation."""

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(
            message=f"Invalid arguments for tool '{tool_name}': {detail}",
            error_code="TOOL_VALIDATION_ERROR",
            status_code=400,
        )
        self.tool_name = tool_name


class ToolExecutionError(ToolException):
    """Raised when a tool fails at runtime during execute()."""

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' encountered an execution error: {detail}",
            error_code="TOOL_EXECUTION_ERROR",
            status_code=500,
        )
        self.tool_name = tool_name


class ToolTimeoutError(ToolException):
    """Raised when a tool exceeds its allowed execution time budget."""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' timed out after {timeout_seconds}s. "
                    f"The service may be temporarily unavailable.",
            error_code="TOOL_TIMEOUT",
            status_code=504,
        )
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds

"""
Tool Calling Framework — Tool Manager

ToolManager is the single, authoritative entry point for all tool execution
in JARVIS. The AI Manager MUST NOT execute tools directly — it delegates
every tool call to ToolManager.

Responsibilities:
  1. Validate tool arguments via the tool's own validate() method.
  2. Execute the tool within a configurable timeout budget.
  3. Emit structured logs: tool selected, execution_time_ms, success/failure.
  4. Convert ALL internal exceptions into user-friendly ToolResult objects
     so that failures never leak raw tracebacks to the caller.
  5. Process batches of tool calls (as returned by AI providers).

Thread/Concurrency model:
  All tool execute() methods are async. ToolManager awaits them within
  asyncio.wait_for() to enforce the timeout. Tools that perform I/O
  (HTTP, filesystem) should use async I/O internally.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from app.config import settings
from app.services.tools.base_tool import ToolResult
from app.services.tools.exceptions import (
    ToolException,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from app.services.tools.registry import ToolRegistry

logger = logging.getLogger("jarvis.services.tools.manager")


class ToolManager:
    """
    Orchestrates tool selection, validation, execution, and error handling.

    Design constraints:
    - AIManager is the ONLY caller of ToolManager.
    - ToolManager never calls AIManager (no circular dependency).
    - Every public method returns ToolResult — callers never handle raw exceptions.
    """

    def __init__(self) -> None:
        self.timeout_seconds: float = float(
            getattr(settings, "TOOL_TIMEOUT_SECONDS", 10)
        )

    # ------------------------------------------------------------------
    # Primary Interface — called by AIManager
    # ------------------------------------------------------------------

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a single named tool with the provided keyword arguments.

        Flow:
          1. Resolve tool from ToolRegistry
          2. Validate arguments
          3. Execute within timeout budget
          4. Return ToolResult (always — never raises)

        Args:
            tool_name: Registered tool identifier (case-insensitive).
            **kwargs:  Tool-specific arguments forwarded to execute().

        Returns:
            ToolResult with success status, data, formatted_output, and timing.
        """
        logger.info(f"[ToolManager] Executing tool: '{tool_name}' | args: {list(kwargs.keys())}")
        start_time = time.perf_counter()

        # ── Step 1: Resolve tool ─────────────────────────────────────
        try:
            tool = ToolRegistry.get_tool(tool_name)
        except ToolNotFoundError as exc:
            elapsed = self._elapsed_ms(start_time)
            logger.error(f"[ToolManager] Tool not found: '{tool_name}'")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=exc.message,
                error_code=exc.error_code,
                execution_time_ms=elapsed,
                formatted_output=(
                    f"I'm sorry, but the tool '{tool_name}' is not available right now. "
                    f"Please try a different approach."
                ),
            )

        # ── Step 2: Validate arguments ───────────────────────────────
        try:
            tool.validate(**kwargs)
        except ToolValidationError as exc:
            elapsed = self._elapsed_ms(start_time)
            logger.warning(
                f"[ToolManager] Validation failed for tool '{tool_name}': {exc.message}"
            )
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=exc.message,
                error_code=exc.error_code,
                execution_time_ms=elapsed,
                formatted_output=(
                    f"I couldn't run the '{tool_name}' tool because the request was invalid: "
                    f"{exc.message}"
                ),
            )

        # ── Step 3: Execute within timeout ───────────────────────────
        try:
            result: ToolResult = await asyncio.wait_for(
                tool.execute(**kwargs),
                timeout=self.timeout_seconds,
            )
            elapsed = self._elapsed_ms(start_time)
            result.execution_time_ms = elapsed

            if result.success:
                logger.info(
                    f"[ToolManager] Tool '{tool_name}' succeeded | "
                    f"time={elapsed:.1f}ms"
                )
            else:
                logger.warning(
                    f"[ToolManager] Tool '{tool_name}' returned failure | "
                    f"code={result.error_code} | time={elapsed:.1f}ms"
                )
            return result

        except asyncio.TimeoutError:
            elapsed = self._elapsed_ms(start_time)
            timeout_exc = ToolTimeoutError(tool_name, self.timeout_seconds)
            logger.error(
                f"[ToolManager] Tool '{tool_name}' timed out after {self.timeout_seconds}s"
            )
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=timeout_exc.message,
                error_code=timeout_exc.error_code,
                execution_time_ms=elapsed,
                formatted_output=(
                    f"The '{tool_name}' tool is taking too long to respond. "
                    f"Please try again in a moment."
                ),
            )

        except ToolException as exc:
            elapsed = self._elapsed_ms(start_time)
            logger.error(
                f"[ToolManager] ToolException in '{tool_name}': {exc.message}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=exc.message,
                error_code=exc.error_code,
                execution_time_ms=elapsed,
                formatted_output=(
                    f"The '{tool_name}' tool encountered an error: {exc.message}"
                ),
            )

        except Exception as exc:
            elapsed = self._elapsed_ms(start_time)
            logger.error(
                f"[ToolManager] Unexpected error in tool '{tool_name}': {str(exc)}",
                exc_info=True,
            )
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=f"An unexpected error occurred: {str(exc)}",
                error_code="TOOL_UNEXPECTED_ERROR",
                execution_time_ms=elapsed,
                formatted_output=(
                    f"I encountered an unexpected problem while using the '{tool_name}' tool. "
                    f"Please try rephrasing your request."
                ),
            )

    async def handle_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of tool call requests as returned by an AI provider.

        The provider returns tool_calls in this shape:
        [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": "{\"query\": \"Python 3.13 features\"}"
                }
            },
            ...
        ]

        This method:
        1. Parses each call's arguments (JSON string → dict).
        2. Executes each tool via execute_tool().
        3. Returns a list of tool result messages ready to be appended
           to the conversation history before the follow-up AI call.

        Args:
            tool_calls: List of tool call dicts from the AI provider response.

        Returns:
            List of message dicts in the format:
            [{"role": "tool", "tool_call_id": "...", "content": "..."}, ...]
        """
        tool_messages: List[Dict[str, Any]] = []

        for call in tool_calls:
            call_id = call.get("id", "unknown")
            function_info = call.get("function", {})
            tool_name = function_info.get("name", "")
            raw_arguments = function_info.get("arguments", "{}")

            logger.info(
                f"[ToolManager] Processing tool call | id={call_id} | tool={tool_name}"
            )

            # Parse JSON arguments safely
            try:
                arguments: Dict[str, Any] = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except json.JSONDecodeError as exc:
                logger.error(
                    f"[ToolManager] Failed to parse arguments for call '{call_id}': {exc}"
                )
                arguments = {}

            # Execute the tool
            result = await self.execute_tool(tool_name, **arguments)

            # Build the tool result message for the conversation
            # Use formatted_output so the AI gets clean, readable context
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.formatted_output or result.error_message or "",
                    "name": tool_name,
                }
            )

        return tool_messages

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Return elapsed wall-clock time in milliseconds since `start`."""
        return (time.perf_counter() - start) * 1000

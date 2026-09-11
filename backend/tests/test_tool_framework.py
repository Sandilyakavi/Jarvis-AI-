"""
Tool Calling Framework — Comprehensive Test Suite

Coverage:
  1. ToolRegistry — registration, retrieval, list, clear
  2. BaseTool — interface enforcement (cannot instantiate abstract class)
  3. SearchTool — metadata, validate (valid + invalid inputs)
  4. ToolManager — execute_tool (success, not-found, validation, timeout, unexpected error)
  5. ToolManager — handle_tool_calls (batch processing, JSON arg parsing)
  6. AIManager integration — tool-call loop with mocked provider + mocked tool

Run with:
    cd backend
    .venv/bin/pytest tests/test_tool_framework.py -v
"""

import asyncio
import json
import pytest
from typing import Any, List, Dict, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# ── Framework imports ──────────────────────────────────────────────────────
from app.services.tools.base_tool import BaseTool, ToolMetadata, ToolResult, ParameterSchema
from app.services.tools.registry import ToolRegistry
from app.services.tools.manager import ToolManager
from app.services.tools.exceptions import (
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
)
from app.services.tools.search.search_tool import SearchTool
from app.services.tools.search.search_provider import BaseSearchProvider, SearchResult
from app.services.tools.search.duckduckgo_provider import DuckDuckGoProvider
from app.services.tools.search.tavily_provider import TavilyProvider
import httpx

# ── Provider / AI Manager imports ──────────────────────────────────────────
from app.providers.base_provider import BaseProvider
from app.services.ai.registry import ProviderRegistry
from app.services.ai.manager import AIManager
from app.services.ai.exceptions import AIServiceException


# ===========================================================================
# Helpers — Minimal concrete implementations for testing
# ===========================================================================


class _ConcreteSuccessTool(BaseTool):
    """A minimal tool that always succeeds. Used to test ToolRegistry."""

    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_success_tool",
            description="A test tool that always succeeds.",
            parameters=[
                ParameterSchema(
                    name="input",
                    type="string",
                    description="Any string input.",
                    required=True,
                )
            ],
        )

    def validate(self, **kwargs: Any) -> bool:
        if not kwargs.get("input"):
            raise ToolValidationError("test_success_tool", "input is required")
        return True

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=True,
            tool_name="test_success_tool",
            data={"echoed": kwargs["input"]},
            formatted_output=f"Echoed: {kwargs['input']}",
        )


class _ConcreteFailureTool(BaseTool):
    """A tool that always returns a failure ToolResult (not raises)."""

    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_failure_tool",
            description="A test tool that always fails.",
        )

    def validate(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(
            success=False,
            tool_name="test_failure_tool",
            error_message="Intentional failure.",
            error_code="TEST_FAILURE",
            formatted_output="This tool intentionally failed.",
        )


class _TimeoutTool(BaseTool):
    """A tool that sleeps longer than the manager timeout."""

    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="test_timeout_tool",
            description="A test tool that simulates timeout.",
        )

    def validate(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(60)  # Will be cancelled by timeout
        return ToolResult(success=True, tool_name="test_timeout_tool")  # Never reached


class _MockSearchProvider(BaseSearchProvider):
    """In-memory search provider for unit tests — no network calls."""

    # Class-level default results (immutable reference)
    _DEFAULT_RESULTS = [
        SearchResult(
            title="Python 3.13 Release Notes",
            url="https://docs.python.org/3.13/",
            snippet="Python 3.13 introduces several new features and improvements.",
        ),
        SearchResult(
            title="What's New in Python 3.13",
            url="https://docs.python.org/3.13/whatsnew/3.13.html",
            snippet="Detailed changes for Python 3.13.",
        ),
    ]

    def __init__(self, results=None, empty: bool = False):
        # `empty=True` means truly no results; `results=None` uses defaults
        if empty:
            self._results = []
        elif results is not None:
            self._results = results
        else:
            self._results = list(self._DEFAULT_RESULTS)

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        return self._results[:max_results]


class _MockAIProvider(BaseProvider):
    """
    Mock AI provider for AIManager integration tests.
    Simulates a single tool-call cycle:
    - First call returns tool_calls (model requests a tool).
    - Second call returns the final natural-language answer.
    """

    def __init__(self, simulate_tool_call: bool = True):
        self._call_count = 0
        self._simulate_tool_call = simulate_tool_call

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._call_count += 1
        if self._simulate_tool_call and self._call_count == 1:
            # First call: model requests the web_search tool
            return {
                "content": "",
                "role": "assistant",
                "model_used": model,
                "tool_calls": [
                    {
                        "id": "call_test_001",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "Python 3.13 features"}),
                        },
                    }
                ],
            }
        # Second call (or non-tool mode): return the final answer
        return {
            "content": "Python 3.13 introduced many improvements including a new JIT compiler.",
            "role": "assistant",
            "model_used": model,
            "tool_calls": None,
        }

    async def stream_chat(self, *args, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"content": "", "role": "assistant", "model_used": "mock"}

    async def health_check(self) -> bool:
        return True


# ===========================================================================
# 1. ToolRegistry Tests
# ===========================================================================


class TestToolRegistry:

    def setup_method(self):
        """Backup existing registry state and clear for isolated tests."""
        self._backup_tools = dict(ToolRegistry._tools)
        self._backup_instances = dict(ToolRegistry._instances)
        ToolRegistry.clear()

    def teardown_method(self):
        """Restore registry to pre-test state."""
        ToolRegistry._tools = self._backup_tools
        ToolRegistry._instances = self._backup_instances

    def test_register_and_retrieve_tool(self):
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        tool = ToolRegistry.get_tool("test_success_tool")
        assert isinstance(tool, _ConcreteSuccessTool)

    def test_singleton_behaviour(self):
        """get_tool() must return the same instance on repeated calls."""
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        instance_a = ToolRegistry.get_tool("test_success_tool")
        instance_b = ToolRegistry.get_tool("test_success_tool")
        assert instance_a is instance_b

    def test_case_insensitive_names(self):
        ToolRegistry.register("MyTool", _ConcreteSuccessTool)
        assert ToolRegistry.is_registered("mytool")
        assert ToolRegistry.is_registered("MYTOOL")
        tool = ToolRegistry.get_tool("MYTOOL")
        assert isinstance(tool, _ConcreteSuccessTool)

    def test_get_unregistered_tool_raises(self):
        with pytest.raises(ToolNotFoundError) as exc_info:
            ToolRegistry.get_tool("nonexistent_tool")
        assert exc_info.value.tool_name == "nonexistent_tool"
        assert exc_info.value.error_code == "TOOL_NOT_FOUND"

    def test_list_tools_returns_metadata(self):
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        ToolRegistry.register("test_failure_tool", _ConcreteFailureTool)
        metadatas = ToolRegistry.list_tools()
        names = [m.name for m in metadatas]
        assert "test_success_tool" in names
        assert "test_failure_tool" in names

    def test_get_function_schemas_format(self):
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        schemas = ToolRegistry.get_function_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        func = schema["function"]
        assert func["name"] == "test_success_tool"
        assert "parameters" in func
        assert "properties" in func["parameters"]
        assert "input" in func["parameters"]["properties"]
        assert "input" in func["parameters"]["required"]

    def test_registered_names_sorted(self):
        ToolRegistry.register("zebra_tool", _ConcreteSuccessTool)
        ToolRegistry.register("alpha_tool", _ConcreteFailureTool)
        names = ToolRegistry.registered_names()
        assert names == sorted(names)

    def test_re_register_invalidates_instance_cache(self):
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        first = ToolRegistry.get_tool("test_success_tool")
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)  # Re-register
        # Instance cache should be cleared
        assert "test_success_tool" not in ToolRegistry._instances


# ===========================================================================
# 2. BaseTool Interface Enforcement Tests
# ===========================================================================


class TestBaseToolInterface:

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_has_name_property(self):
        tool = _ConcreteSuccessTool()
        assert tool.name == "test_success_tool"

    def test_metadata_returns_tool_metadata(self):
        tool = _ConcreteSuccessTool()
        meta = tool.metadata()
        assert isinstance(meta, ToolMetadata)
        assert meta.name == "test_success_tool"
        assert meta.version == "1.0.0"

    def test_function_schema_structure(self):
        tool = _ConcreteSuccessTool()
        schema = tool.metadata().to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_success_tool"
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "input" in params["properties"]


# ===========================================================================
# 3. SearchTool Tests
# ===========================================================================


class TestSearchTool:

    def setup_method(self):
        self._mock_provider = _MockSearchProvider()
        self._tool = SearchTool(provider=self._mock_provider)

    # Metadata
    def test_metadata_name(self):
        assert self._tool.metadata().name == "web_search"

    def test_metadata_has_query_parameter(self):
        params = self._tool.metadata().parameters
        names = [p.name for p in params]
        assert "query" in names

    def test_function_schema_is_valid(self):
        schema = self._tool.metadata().to_function_schema()
        func = schema["function"]
        assert func["name"] == "web_search"
        assert "query" in func["parameters"]["required"]

    # Validate — valid inputs
    def test_validate_with_valid_query(self):
        assert self._tool.validate(query="Python 3.13") is True

    def test_validate_with_valid_max_results(self):
        assert self._tool.validate(query="test", max_results=3) is True

    # Validate — invalid inputs
    def test_validate_missing_query_raises(self):
        with pytest.raises(ToolValidationError) as exc_info:
            self._tool.validate()
        assert "query" in exc_info.value.message.lower()

    def test_validate_empty_query_raises(self):
        with pytest.raises(ToolValidationError):
            self._tool.validate(query="   ")

    def test_validate_non_string_query_raises(self):
        with pytest.raises(ToolValidationError):
            self._tool.validate(query=42)

    def test_validate_invalid_max_results_raises(self):
        with pytest.raises(ToolValidationError):
            self._tool.validate(query="test", max_results=0)

    def test_validate_negative_max_results_raises(self):
        with pytest.raises(ToolValidationError):
            self._tool.validate(query="test", max_results=-1)

    # Execute
    @pytest.mark.anyio
    async def test_execute_returns_tool_result(self):
        result = await self._tool.execute(query="Python 3.13 features")
        assert isinstance(result, ToolResult)
        assert result.tool_name == "web_search"
        assert result.success is True

    @pytest.mark.anyio
    async def test_execute_formats_output(self):
        result = await self._tool.execute(query="Python 3.13")
        assert "Python 3.13" in result.formatted_output
        assert "1." in result.formatted_output  # Numbered list

    @pytest.mark.anyio
    async def test_execute_returns_data_list(self):
        result = await self._tool.execute(query="Python 3.13")
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        assert "title" in result.data[0]
        assert "url" in result.data[0]
        assert "snippet" in result.data[0]

    @pytest.mark.anyio
    async def test_execute_with_no_results(self):
        # Construct a fresh tool with an explicitly empty provider
        empty_provider = _MockSearchProvider(empty=True)
        tool = SearchTool(provider=empty_provider)
        result = await tool.execute(query="extremely_obscure_query_xyz_123")
        assert result.success is True  # "no results" is not a failure
        assert result.data == []
        assert "no web results" in result.formatted_output.lower()

    @pytest.mark.anyio
    async def test_execute_handles_provider_exception(self):
        class _BrokenProvider(BaseSearchProvider):
            async def search(self, query, max_results=5):
                raise ConnectionError("Search service down")

        tool = SearchTool(provider=_BrokenProvider())
        result = await tool.execute(query="anything")
        assert result.success is False
        assert result.error_code == "SEARCH_PROVIDER_ERROR"
        assert "unavailable" in result.formatted_output.lower()

    @pytest.mark.anyio
    async def test_execute_respects_max_results(self):
        many_results = [
            SearchResult(f"Title {i}", f"https://example.com/{i}", f"Snippet {i}")
            for i in range(10)
        ]
        provider = _MockSearchProvider(results=many_results)
        tool = SearchTool(provider=provider)
        result = await tool.execute(query="test", max_results=3)
        assert len(result.data) == 3


# ===========================================================================
# 4. ToolManager Tests
# ===========================================================================


class TestToolManager:

    def setup_method(self):
        self._backup_tools = dict(ToolRegistry._tools)
        self._backup_instances = dict(ToolRegistry._instances)
        ToolRegistry.clear()
        ToolRegistry.register("test_success_tool", _ConcreteSuccessTool)
        ToolRegistry.register("test_failure_tool", _ConcreteFailureTool)
        ToolRegistry.register("test_timeout_tool", _TimeoutTool)

    def teardown_method(self):
        ToolRegistry._tools = self._backup_tools
        ToolRegistry._instances = self._backup_instances

    # execute_tool — success
    @pytest.mark.anyio
    async def test_execute_tool_success(self):
        manager = ToolManager()
        result = await manager.execute_tool("test_success_tool", input="hello")
        assert result.success is True
        assert result.tool_name == "test_success_tool"
        assert result.data == {"echoed": "hello"}
        assert result.execution_time_ms >= 0

    # execute_tool — tool returns failure ToolResult (not exception)
    @pytest.mark.anyio
    async def test_execute_tool_failure_result(self):
        manager = ToolManager()
        result = await manager.execute_tool("test_failure_tool")
        assert result.success is False
        assert result.error_code == "TEST_FAILURE"
        assert result.execution_time_ms >= 0

    # execute_tool — unknown tool
    @pytest.mark.anyio
    async def test_execute_unknown_tool(self):
        manager = ToolManager()
        result = await manager.execute_tool("nonexistent_tool")
        assert result.success is False
        assert result.error_code == "TOOL_NOT_FOUND"
        assert "not available" in result.formatted_output.lower()

    # execute_tool — validation failure
    @pytest.mark.anyio
    async def test_execute_tool_validation_failure(self):
        manager = ToolManager()
        # Missing required 'input' argument
        result = await manager.execute_tool("test_success_tool")
        assert result.success is False
        assert result.error_code == "TOOL_VALIDATION_ERROR"

    # execute_tool — timeout
    @pytest.mark.anyio
    async def test_execute_tool_timeout(self):
        manager = ToolManager()
        manager.timeout_seconds = 0.05  # 50ms timeout for the test
        result = await manager.execute_tool("test_timeout_tool")
        assert result.success is False
        assert result.error_code == "TOOL_TIMEOUT"
        assert "too long" in result.formatted_output.lower()

    # execute_tool — unexpected exception inside tool
    @pytest.mark.anyio
    async def test_execute_tool_unexpected_exception(self):
        class _RaisingTool(BaseTool):
            def metadata(self):
                return ToolMetadata(name="raising_tool", description="Raises unexpectedly.")
            def validate(self, **kwargs):
                return True
            async def execute(self, **kwargs):
                raise RuntimeError("Something exploded!")

        ToolRegistry.register("raising_tool", _RaisingTool)
        manager = ToolManager()
        result = await manager.execute_tool("raising_tool")
        assert result.success is False
        assert result.error_code == "TOOL_UNEXPECTED_ERROR"

    # handle_tool_calls — batch processing
    @pytest.mark.anyio
    async def test_handle_tool_calls_returns_messages(self):
        manager = ToolManager()
        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "test_success_tool",
                    "arguments": json.dumps({"input": "world"}),
                },
            }
        ]
        messages = await manager.handle_tool_calls(tool_calls)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_001"
        assert msg["name"] == "test_success_tool"
        assert isinstance(msg["content"], str)
        assert len(msg["content"]) > 0

    # handle_tool_calls — malformed JSON arguments
    @pytest.mark.anyio
    async def test_handle_tool_calls_bad_json_arguments(self):
        manager = ToolManager()
        tool_calls = [
            {
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "test_success_tool",
                    "arguments": "NOT_VALID_JSON",
                },
            }
        ]
        # Should not raise; gracefully handles bad JSON
        messages = await manager.handle_tool_calls(tool_calls)
        assert len(messages) == 1
        assert messages[0]["tool_call_id"] == "call_002"

    # handle_tool_calls — multiple tool calls
    @pytest.mark.anyio
    async def test_handle_multiple_tool_calls(self):
        manager = ToolManager()
        tool_calls = [
            {
                "id": f"call_{i:03d}",
                "type": "function",
                "function": {
                    "name": "test_success_tool",
                    "arguments": json.dumps({"input": f"item_{i}"}),
                },
            }
            for i in range(3)
        ]
        messages = await manager.handle_tool_calls(tool_calls)
        assert len(messages) == 3
        ids = [m["tool_call_id"] for m in messages]
        assert "call_000" in ids
        assert "call_001" in ids
        assert "call_002" in ids


# ===========================================================================
# 5. AIManager Integration Tests (mocked provider + mocked tool)
# ===========================================================================


class TestAIManagerToolCallIntegration:

    def setup_method(self):
        # Register mock AI provider
        ProviderRegistry.register("mock_tool_provider", _MockAIProvider)

        # Register the real SearchTool with mock search backend
        self._backup_tools = dict(ToolRegistry._tools)
        self._backup_instances = dict(ToolRegistry._instances)
        ToolRegistry.clear()

        mock_search_tool = SearchTool(provider=_MockSearchProvider())
        # Pre-populate the instance cache so ToolRegistry uses our mock
        ToolRegistry._tools["web_search"] = SearchTool
        ToolRegistry._instances["web_search"] = mock_search_tool

    def teardown_method(self):
        ToolRegistry._tools = self._backup_tools
        ToolRegistry._instances = self._backup_instances
        # Remove the mock provider's singleton so it resets call count
        ProviderRegistry._instances.pop("mock_tool_provider", None)

    @pytest.mark.anyio
    async def test_tool_call_loop_produces_final_answer(self):
        """
        AIManager must:
        1. Call the provider → receives tool_calls.
        2. Delegate to ToolManager → execute web_search.
        3. Call the provider again → receive final answer.
        4. Return the final answer (NOT raw tool call JSON).
        """
        # Pre-configure a fresh mock provider instance
        mock_provider = _MockAIProvider(simulate_tool_call=True)
        ProviderRegistry._instances["mock_tool_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "What's new in Python 3.13?"}],
            provider="mock_tool_provider",
            model="mock-model",
        )

        # The return shape must be identical to the pre-tool-calling shape
        assert "message" in result
        assert "provider" in result
        assert "model" in result
        assert result["provider"] == "mock_tool_provider"

        # Final content must be the second (post-tool) response
        assert "JIT" in result["message"]["content"] or "Python 3.13" in result["message"]["content"]

        # Provider must have been called twice (once for tool, once for answer)
        assert mock_provider._call_count == 2

    @pytest.mark.anyio
    async def test_no_tool_calls_skips_loop(self):
        """When the model returns no tool_calls, the loop must be skipped."""
        mock_provider = _MockAIProvider(simulate_tool_call=False)
        ProviderRegistry._instances["mock_tool_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "Hello"}],
            provider="mock_tool_provider",
            model="mock-model",
        )
        assert result["message"]["content"] is not None
        assert mock_provider._call_count == 1  # Only one call needed

    @pytest.mark.anyio
    async def test_return_shape_is_preserved(self):
        """Ensure the return dict shape matches the pre-tool-calling contract."""
        mock_provider = _MockAIProvider(simulate_tool_call=False)
        ProviderRegistry._instances["mock_tool_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "Test"}],
            provider="mock_tool_provider",
            model="mock-model",
        )
        assert set(result.keys()) == {"message", "provider", "model"}
        assert set(result["message"].keys()) == {"role", "content"}


# ===========================================================================
# 7. TavilyProvider Tests
# ===========================================================================


class TestTavilyProvider:
    """Comprehensive test suite for TavilyProvider."""

    _SAMPLE_TAVILY_RESPONSE = {
        "query": "artificial intelligence latest developments",
        "results": [
            {
                "title": "AI Breakthroughs in 2026",
                "url": "https://example.com/ai-2026",
                "content": "Researchers have announced new advances in autonomous reasoning.",
                "score": 0.98,
            },
            {
                "title": "Next-Gen LLM Architectures",
                "url": "https://example.com/next-gen-llm",
                "snippet": "Novel state space and attention hybrids show high efficiency.",
                "score": 0.92,
            },
        ],
        "response_time": 0.42,
    }

    @pytest.mark.anyio
    async def test_tavily_search_success_and_mapping(self):
        """Tavily search returns valid SearchResult list mapped properly."""
        provider = TavilyProvider(api_key="tvly-test-valid-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = self._SAMPLE_TAVILY_RESPONSE

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            results = await provider.search(query="artificial intelligence", max_results=5)

            assert len(results) == 2
            assert isinstance(results[0], SearchResult)
            assert results[0].title == "AI Breakthroughs in 2026"
            assert results[0].url == "https://example.com/ai-2026"
            assert "autonomous reasoning" in results[0].snippet

            # Check fallback to 'snippet' field if 'content' is not primary
            assert results[1].title == "Next-Gen LLM Architectures"
            assert "attention hybrids" in results[1].snippet

            # Verify request payload sent to Tavily
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["api_key"] == "tvly-test-valid-key"
            assert payload["query"] == "artificial intelligence"
            assert payload["max_results"] == 5

    @pytest.mark.anyio
    async def test_tavily_search_empty_results(self):
        """Tavily search returns empty list when API provides zero results."""
        provider = TavilyProvider(api_key="tvly-test-valid-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            results = await provider.search(query="obscure query", max_results=5)
            assert results == []

    @pytest.mark.anyio
    async def test_tavily_missing_api_key_raises(self):
        """Instantiating or calling Tavily search with empty API key raises ValueError."""
        provider = TavilyProvider(api_key="")

        with pytest.raises(ValueError) as exc_info:
            await provider.search(query="test")

        assert "Tavily API key is not configured" in str(exc_info.value)
        # Ensure error does not leak or display secrets
        assert "tvly" not in str(exc_info.value)

    @pytest.mark.anyio
    async def test_tavily_auth_error_401(self):
        """HTTP 401 returns clean authentication failure without leaking key."""
        provider = TavilyProvider(api_key="tvly-invalid-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                await provider.search(query="test")

            assert "authentication failed" in str(exc_info.value).lower()
            assert "tvly-invalid-key" not in str(exc_info.value)

    @pytest.mark.anyio
    async def test_tavily_rate_limit_429(self):
        """HTTP 429 returns rate limit error message."""
        provider = TavilyProvider(api_key="tvly-test-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                await provider.search(query="test")

            assert "rate limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_tavily_server_error_500(self):
        """HTTP 500 returns clean HTTP error message."""
        provider = TavilyProvider(api_key="tvly-test-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError) as exc_info:
                await provider.search(query="test")

            assert "HTTP error status: 500" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_tavily_timeout_handling(self):
        """httpx.TimeoutException is wrapped into TimeoutError."""
        provider = TavilyProvider(api_key="tvly-test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Connection timed out")

            with pytest.raises(TimeoutError) as exc_info:
                await provider.search(query="test")

            assert "timed out" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_tavily_network_error_handling(self):
        """httpx.RequestError is wrapped into RuntimeError."""
        provider = TavilyProvider(api_key="tvly-test-key")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(RuntimeError) as exc_info:
                await provider.search(query="test")

            assert "Network error" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_search_tool_integration_with_tavily(self):
        """SearchTool executes properly when using TavilyProvider."""
        provider = TavilyProvider(api_key="tvly-test-key")
        tool = SearchTool(provider=provider)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = self._SAMPLE_TAVILY_RESPONSE

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await tool.execute(query="artificial intelligence")

            assert result.success is True
            assert result.tool_name == "web_search"
            assert len(result.data) == 2
            assert "AI Breakthroughs in 2026" in result.formatted_output

# ===========================================================================
# 8. Tool-Selection Behavior Tests (System Prompt + AIManager Integration)
# ===========================================================================


class _FreshnessAwareAIProvider(BaseProvider):
    """
    Mock provider that inspects the user message for freshness keywords and
    decides whether to issue a tool call.

    Simulates the expected model behavior after the system-prompt enhancement:
    - Freshness-sensitive queries -> return a web_search tool_call on first call.
    - Stable general-knowledge queries -> return a direct answer (no tool_call).
    """

    def __init__(self):
        self._call_count = 0
        self._system_prompt_seen = ""
        self._tool_result_seen = None

    _FRESHNESS_KEYWORDS = [
        "latest", "current", "recent", "today", "newest",
        "up-to-date", "just launched", "recently released",
        "breaking", "trending", "news", "update",
    ]

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._call_count += 1

        # Capture the system prompt from the first message
        if messages and messages[0].get("role") == "system":
            self._system_prompt_seen = messages[0]["content"]

        # Extract the user message text
        user_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_text = msg["content"].lower()

        # Check if any tool result messages are present (passthrough validation)
        for msg in messages:
            if msg.get("role") == "tool":
                self._tool_result_seen = msg["content"]

        # On the first call, decide whether to issue a tool call
        if self._call_count == 1:
            is_freshness = any(kw in user_text for kw in self._FRESHNESS_KEYWORDS)
            if is_freshness:
                return {
                    "content": "",
                    "role": "assistant",
                    "model_used": model,
                    "tool_calls": [
                        {
                            "id": "call_fresh_001",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": user_text}),
                            },
                        }
                    ],
                }
            else:
                # No tool call for stable questions
                return {
                    "content": "Photosynthesis is the process by which plants convert sunlight into energy.",
                    "role": "assistant",
                    "model_used": model,
                    "tool_calls": None,
                }

        # Second call (post-tool): return the final answer incorporating tool results
        return {
            "content": "Based on the latest web search results, here is the current information.",
            "role": "assistant",
            "model_used": model,
            "tool_calls": None,
        }

    async def stream_chat(self, *args, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"content": "", "role": "assistant", "model_used": "mock"}

    async def health_check(self) -> bool:
        return True


class TestToolSelectionBehavior:
    """
    Validates that the system prompt instructs the AI to call web_search for
    freshness-sensitive queries and to skip it for stable/general-knowledge queries.
    """

    def setup_method(self):
        ProviderRegistry.register("freshness_test_provider", _FreshnessAwareAIProvider)
        self._backup_tools = dict(ToolRegistry._tools)
        self._backup_instances = dict(ToolRegistry._instances)
        ToolRegistry.clear()

        mock_search_tool = SearchTool(provider=_MockSearchProvider())
        ToolRegistry._tools["web_search"] = SearchTool
        ToolRegistry._instances["web_search"] = mock_search_tool

    def teardown_method(self):
        ToolRegistry._tools = self._backup_tools
        ToolRegistry._instances = self._backup_instances
        ProviderRegistry._instances.pop("freshness_test_provider", None)

    @pytest.mark.anyio
    async def test_freshness_query_triggers_web_search(self):
        """
        A query like 'What is the latest model launched by OpenAI?' must result
        in the AI requesting a web_search tool call.
        """
        mock_provider = _FreshnessAwareAIProvider()
        ProviderRegistry._instances["freshness_test_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "What is the latest model launched by OpenAI?"}],
            provider="freshness_test_provider",
            model="mock-model",
        )

        # Provider should have been called twice: tool call + final answer
        assert mock_provider._call_count == 2
        # Final answer should be the post-tool response
        assert "web search results" in result["message"]["content"].lower()

    @pytest.mark.anyio
    async def test_stable_query_skips_web_search(self):
        """
        A stable/general-knowledge query like 'What is photosynthesis?' must NOT
        trigger a web_search tool call.
        """
        mock_provider = _FreshnessAwareAIProvider()
        ProviderRegistry._instances["freshness_test_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "What is photosynthesis?"}],
            provider="freshness_test_provider",
            model="mock-model",
        )

        # Provider should have been called only ONCE — no tool call loop
        assert mock_provider._call_count == 1
        assert "photosynthesis" in result["message"]["content"].lower()

    @pytest.mark.anyio
    async def test_tool_results_are_passed_back_to_provider(self):
        """
        When web_search is invoked, the tool results must be appended to the
        conversation and sent to the provider on the follow-up call.
        """
        mock_provider = _FreshnessAwareAIProvider()
        ProviderRegistry._instances["freshness_test_provider"] = mock_provider

        manager = AIManager()
        await manager.chat(
            messages=[{"role": "user", "content": "What are the latest news in AI?"}],
            provider="freshness_test_provider",
            model="mock-model",
        )

        # The provider's second call should have received tool result messages
        assert mock_provider._tool_result_seen is not None
        assert len(mock_provider._tool_result_seen) > 0

    def test_system_prompt_contains_freshness_tool_instructions(self):
        """
        The SYSTEM_PROMPT must contain explicit freshness-keyword-based
        tool-selection rules (not just generic tool description).
        """
        from app.config import settings

        prompt = settings.SYSTEM_PROMPT.lower()

        # Must reference web_search tool
        assert "web_search" in prompt

        # Must contain key freshness trigger words
        for keyword in ["latest", "current", "recent", "today", "newest"]:
            assert keyword in prompt, f"Freshness keyword '{keyword}' missing from system prompt"

        # Must instruct to call BEFORE answering
        assert "before answering" in prompt

        # Must instruct NOT to search for stable knowledge
        assert "do not call web_search" in prompt.replace("\n", " ")

    @pytest.mark.anyio
    async def test_existing_tool_loop_still_works_with_new_prompt(self):
        """
        The existing tool-call loop must continue to function identically
        with the updated system prompt.
        """
        mock_provider = _MockAIProvider(simulate_tool_call=True)
        ProviderRegistry.register("mock_tool_provider", _MockAIProvider)
        ProviderRegistry._instances["mock_tool_provider"] = mock_provider

        manager = AIManager()
        result = await manager.chat(
            messages=[{"role": "user", "content": "What's new in Python 3.13?"}],
            provider="mock_tool_provider",
            model="mock-model",
        )

        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] is not None
        assert mock_provider._call_count == 2  # tool call + final answer
        assert set(result.keys()) == {"message", "provider", "model"}

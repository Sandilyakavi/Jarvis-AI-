import pytest
from typing import List, Dict, Any, AsyncGenerator
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from app.main import app
from app.providers.base_provider import BaseProvider
from app.services.ai.registry import ProviderRegistry
from app.services.ai.manager import AIManager
from app.services.ai.exceptions import ProviderNotFoundError, AIServiceException

client = TestClient(app)

# 1. Create Mock Provider for testing
class MockAIProvider(BaseProvider):
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        if model == "error-model":
            raise AIServiceException("Mock Provider Error", error_code="MOCK_ERROR", status_code=500)
        return {
            "content": f"Mock response for: {messages[-1]['content']}",
            "role": "assistant",
            "model_used": model
        }

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "content": "Mock stream",
            "role": "assistant",
            "model_used": model
        }

    async def health_check(self) -> bool:
        return True

# Register the Mock Provider
ProviderRegistry.register("mock", MockAIProvider)


def test_registry_registration():
    provider = ProviderRegistry.get_provider("mock")
    assert isinstance(provider, MockAIProvider)

    with pytest.raises(ProviderNotFoundError):
        ProviderRegistry.get_provider("unknown_provider")


@pytest.mark.anyio
async def test_ai_manager_chat_routing():
    manager = AIManager()
    result = await manager.chat(
        messages=[{"role": "user", "content": "Hello Jarvis"}],
        provider="mock",
        model="mock-model"
    )
    assert result["provider"] == "mock"
    assert result["model"] == "mock-model"
    assert result["message"]["content"] == "Mock response for: Hello Jarvis"


def test_api_chat_endpoint_success():
    payload = {
        "messages": [
            {"role": "user", "content": "Ping"}
        ],
        "provider": "mock",
        "model": "mock-model-v1",
        "temperature": 0.5
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["message"] == "Chat response completed successfully"
    assert data["data"]["provider"] == "mock"
    assert data["data"]["model"] == "mock-model-v1"
    assert data["data"]["message"]["content"] == "Mock response for: Ping"


def test_api_chat_validation_failure():
    # Empty messages list should fail validation
    payload = {
        "messages": [],
        "provider": "mock"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"


def test_api_chat_provider_error_propagation():
    payload = {
        "messages": [
            {"role": "user", "content": "Trigger Error"}
        ],
        "provider": "mock",
        "model": "error-model"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "MOCK_ERROR"
    assert "Mock Provider Error" in data["message"]


@pytest.mark.anyio
async def test_groq_provider_chat_success():
    from unittest.mock import AsyncMock, MagicMock
    from app.providers.groq_provider import GroqProvider

    provider = GroqProvider(api_key="test-key")
    
    mock_choice = MagicMock()
    mock_choice.message.content = "Groq hello"
    mock_choice.message.role = "assistant"
    mock_choice.message.tool_calls = None
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "openai/gpt-oss-20b"

    provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="openai/gpt-oss-20b",
        stream=True  # Should be overridden to stream=False internally
    )

    assert result["content"] == "Groq hello"
    assert result["role"] == "assistant"
    # Verify stream=False was passed to create()
    call_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is False


@pytest.mark.anyio
async def test_ai_manager_injects_system_prompt():
    """Verify AIManager automatically prepends the JARVIS system prompt containing creator identity."""
    from unittest.mock import AsyncMock
    from app.config import settings

    captured_messages = []

    class _CaptureProvider(BaseProvider):
        async def chat(self, messages, model, temperature=0.7, **kwargs):
            nonlocal captured_messages
            captured_messages = list(messages)
            return {"content": "Identity confirmed", "role": "assistant", "model_used": model}

        async def stream_chat(self, *args, **kwargs):
            yield {}

        async def health_check(self):
            return True

    ProviderRegistry.register("capture_mock", _CaptureProvider)

    manager = AIManager()
    await manager.chat(
        messages=[{"role": "user", "content": "Who is your creator?"}],
        provider="capture_mock",
        model="test-model"
    )

    assert len(captured_messages) == 2
    assert captured_messages[0]["role"] == "system"
    assert "Sandilya Kavi" in captured_messages[0]["content"]
    assert captured_messages[1]["role"] == "user"
    assert captured_messages[1]["content"] == "Who is your creator?"



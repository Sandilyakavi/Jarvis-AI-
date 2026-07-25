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

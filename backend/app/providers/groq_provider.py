import logging
from typing import List, Dict, Any, AsyncGenerator
# pyrefly: ignore [missing-import]
from groq import AsyncGroq, APIConnectionError, APIStatusError
from app.providers.base_provider import BaseProvider
from app.services.ai.exceptions import ProviderConnectionError, ProviderAPIError

logger = logging.getLogger("jarvis.providers.groq")

class GroqProvider(BaseProvider):
    """
    Groq API provider implementation of BaseProvider.
    Converts Groq specific exceptions into unified AI Service exceptions.
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            logger.warning("GroqProvider initialized without an API key.")
        self.client = AsyncGroq(api_key=api_key)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        try:
            logger.debug(f"Sending chat request to Groq using model: {model}")
            response = await self.client.chat.completions.create(
                messages=messages, # type: ignore
                model=model,
                temperature=temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content or ""
            role = response.choices[0].message.role or "assistant"

            # Capture tool calls if the model requested any
            raw_tool_calls = response.choices[0].message.tool_calls
            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_tool_calls
                ]

            return {
                "content": content,
                "role": role,
                "model_used": response.model,
                "tool_calls": tool_calls,  # None if no tool call; list otherwise
            }
            
        except APIConnectionError as e:
            logger.error(f"Groq API connection error: {str(e)}")
            raise ProviderConnectionError("groq", detail=str(e))
        except APIStatusError as e:
            logger.error(f"Groq API status error: status={e.status_code}, message={e.message}")
            raise ProviderAPIError("groq", detail=e.message, status_code=e.status_code)
        except Exception as e:
            logger.error(f"Unexpected error in Groq provider: {str(e)}")
            raise ProviderAPIError("groq", detail=str(e), status_code=500)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Placeholder for streaming chat responses in future sprints."""
        # Yielding a single block for compatibility during Day 2, or raising not implemented.
        # Let's yield a standard chunk to keep the signature functional.
        try:
            response = await self.client.chat.completions.create(
                messages=messages, # type: ignore
                model=model,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {
                        "content": delta.content,
                        "role": "assistant",
                        "model_used": chunk.model
                    }
        except APIConnectionError as e:
            raise ProviderConnectionError("groq", detail=str(e))
        except APIStatusError as e:
            raise ProviderAPIError("groq", detail=e.message, status_code=e.status_code)
        except Exception as e:
            raise ProviderAPIError("groq", detail=str(e), status_code=500)

    async def health_check(self) -> bool:
        """
        Executes a minimalist request to verify connectivity and API key validity.
        """
        try:
            # Low token request to verify availability
            await self.client.chat.completions.create(
                messages=[{"role": "user", "content": "ping"}],
                model="llama3-8b-8192",
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Groq health check failed: {str(e)}")
            return False

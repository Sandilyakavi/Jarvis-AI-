from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator

class BaseProvider(ABC):
    """
    Abstract Base Class defining the interface that all AI providers must implement.
    Allows the AI Manager to communicate with various services in a unified way.
    """

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Sends a list of messages to the provider and returns a standard response payload.
        
        Expected return dictionary shape:
        {
            "content": str,
            "role": "assistant",
            "model_used": str
        }
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Sends a list of messages to the provider and yields tokens in a streaming fashion.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Checks the connectivity/health of the provider.
        Returns True if the provider is reachable and active, False otherwise.
        """
        pass

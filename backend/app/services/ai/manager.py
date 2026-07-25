import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.ai.registry import ProviderRegistry
from app.services.ai.exceptions import AIServiceException, InvalidRequestError

logger = logging.getLogger("jarvis.services.ai.manager")

class AIManager:
    """
    Core orchestrator for AI Service operations.
    Hides provider implementation details from outer application layers.
    """

    def __init__(self):
        self.default_provider = settings.DEFAULT_PROVIDER
        self.default_model = settings.DEFAULT_MODEL

    async def chat(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Routes the chat request to the designated provider.
        """
        # 1. Input Validation
        if not messages:
            raise InvalidRequestError("Message list cannot be empty.")
            
        selected_provider = (provider or self.default_provider).lower()
        selected_model = model or self.default_model

        logger.info(
            f"Routing chat request - Provider: '{selected_provider}', Model: '{selected_model}', Temp: {temperature}"
        )

        try:
            # 2. Retrieve provider instance from the registry
            provider_instance = ProviderRegistry.get_provider(selected_provider)
            
            # 3. Request Completion from provider
            result = await provider_instance.chat(
                messages=messages,
                model=selected_model,
                temperature=temperature,
                **kwargs
            )
            
            logger.info(f"Chat request completed successfully via provider: '{selected_provider}'")
            return {
                "message": {
                    "role": result["role"],
                    "content": result["content"]
                },
                "provider": selected_provider,
                "model": result["model_used"]
            }

        except AIServiceException as e:
            # Re-raise known/wrapped AI exceptions
            raise e
        except Exception as e:
            logger.error(f"Unhandled exception in AIManager: {str(e)}", exc_info=True)
            raise AIServiceException(
                message=f"An unexpected error occurred during processing: {str(e)}",
                error_code="AI_MANAGER_ERROR",
                status_code=500
            )

    async def check_provider_health(self, provider: str) -> bool:
        """
        Queries health status of a specific provider.
        """
        try:
            provider_instance = ProviderRegistry.get_provider(provider)
            return await provider_instance.health_check()
        except Exception as e:
            logger.error(f"Health check for provider '{provider}' failed with error: {str(e)}")
            return False

import logging
from typing import Dict, Type
from app.providers.base_provider import BaseProvider
from app.providers.groq_provider import GroqProvider
from app.config import settings
from app.services.ai.exceptions import ProviderNotFoundError

logger = logging.getLogger("jarvis.services.ai.registry")

class ProviderRegistry:
    """
    Registry that catalogs and handles the creation of AI Providers.
    Follows open-closed principle to easily register new providers.
    """
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    _instances: Dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Registers a new provider class."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered AI Provider: {name}")

    @classmethod
    def get_provider(cls, name: str) -> BaseProvider:
        """
        Retrieves or instantiates the requested provider.
        Ensures lazy loading / singleton instantiations.
        """
        provider_name = name.lower()
        
        if provider_name in cls._instances:
            return cls._instances[provider_name]
            
        if provider_name not in cls._providers:
            logger.error(f"Attempted to access unregistered provider: {provider_name}")
            raise ProviderNotFoundError(provider_name)
            
        # Instantiate provider based on credentials from settings
        provider_class = cls._providers[provider_name]
        
        if provider_name == "groq":
            instance = provider_class(api_key=settings.GROQ_API_KEY)
        else:
            # Future providers will extract their custom keys from settings here
            instance = provider_class()
            
        cls._instances[provider_name] = instance
        return instance

# Register existing default providers
ProviderRegistry.register("groq", GroqProvider)

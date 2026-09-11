import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.ai.registry import ProviderRegistry
from app.services.ai.exceptions import AIServiceException, InvalidRequestError
from app.services.tools.manager import ToolManager
from app.services.tools.registry import ToolRegistry

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

        # Initialise ToolManager and build function schemas for the provider
        tool_manager = ToolManager()
        tool_schemas = ToolRegistry.get_function_schemas()

        try:
            # 2. Retrieve provider instance from the registry
            provider_instance = ProviderRegistry.get_provider(selected_provider)

            # 3. Build initial message list (with system prompt injected if not present)
            conversation: List[Dict[str, str]] = []
            if not messages or messages[0].get("role") != "system":
                conversation.append({"role": "system", "content": settings.SYSTEM_PROMPT})
            conversation.extend(messages)

            # 4. First provider call — pass tool schemas if any are registered
            call_kwargs: Dict[str, Any] = dict(kwargs)
            if tool_schemas:
                call_kwargs["tools"] = tool_schemas
                call_kwargs["tool_choice"] = "auto"  # let the model decide

            result = await provider_instance.chat(
                messages=conversation,
                model=selected_model,
                temperature=temperature,
                **call_kwargs
            )

            # 5. Tool-Call Loop ─────────────────────────────────────────────────
            # The model may request one or more tool calls before producing its
            # final natural-language answer. We loop until the model produces
            # a plain text response (no more tool_calls in the response).
            #
            # IMPORTANT: The user NEVER sees raw tool-call JSON — only the
            # final natural-language answer is returned.
            max_iterations = 5  # Safety guard against infinite loops
            iteration = 0

            while result.get("tool_calls") and iteration < max_iterations:
                iteration += 1
                tool_calls = result["tool_calls"]

                logger.info(
                    f"[AIManager] Model requested {len(tool_calls)} tool call(s) "
                    f"(iteration {iteration}/{max_iterations})"
                )

                # Append the assistant's tool-call message to conversation history
                conversation.append(
                    {
                        "role": "assistant",
                        "content": result.get("content") or "",
                        "tool_calls": tool_calls,  # type: ignore[typeddict-item]
                    }
                )

                # Delegate ALL tool execution to ToolManager
                tool_result_messages = await tool_manager.handle_tool_calls(tool_calls)

                # Append tool results to conversation
                conversation.extend(tool_result_messages)  # type: ignore[arg-type]

                # Follow-up call to provider for the final answer
                result = await provider_instance.chat(
                    messages=conversation,
                    model=selected_model,
                    temperature=temperature,
                    **call_kwargs
                )

            # ─────────────────────────────────────────────────────────────────

            logger.info(
                f"Chat request completed successfully via provider: '{selected_provider}'"
            )
            return {
                "message": {
                    "role": result["role"],
                    "content": result["content"],
                },
                "provider": selected_provider,
                "model": result["model_used"],
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

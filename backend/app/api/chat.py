import logging
from fastapi import APIRouter, Depends, status
from app.schemas.chat import ChatRequest, ChatResponseData, StandardResponse
from app.services.ai.manager import AIManager

logger = logging.getLogger("jarvis.api.chat")
router = APIRouter(prefix="/chat", tags=["Chat"])

def get_ai_manager() -> AIManager:
    return AIManager()

@router.post(
    "",
    response_model=StandardResponse[ChatResponseData],
    status_code=status.HTTP_200_OK,
    summary="Send chat messages to the AI assistant"
)
async def chat(
    request: ChatRequest,
    ai_manager: AIManager = Depends(get_ai_manager)
):
    """
    Exposes a unified interface to chat with the AI.
    Hides all provider details from the client/frontend.
    """
    logger.debug(f"Received chat request: {request}")
    
    # Map pydantic schema list to dictionary format expected by providers
    messages_payload = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    result = await ai_manager.chat(
        messages=messages_payload,
        provider=request.provider,
        model=request.model,
        temperature=request.temperature
    )
    
    response_data = ChatResponseData(
        message=result["message"],
        provider=result["provider"],
        model=result["model"]
    )
    
    return StandardResponse(
        success=True,
        message="Chat response completed successfully",
        data=response_data
    )

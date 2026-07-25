import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.chat import router as chat_router
from app.services.ai.exceptions import AIServiceException
from app.services.ai.manager import AIManager
from app.schemas.chat import StandardErrorResponse, ErrorDetail

# Configure Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("jarvis.main")

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for Project JARVIS",
    version="1.0.0",
    debug=settings.DEBUG,
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handler for AI Layer specific exceptions
@app.exception_handler(AIServiceException)
async def ai_service_exception_handler(request: Request, exc: AIServiceException):
    logger.error(f"AI Service Exception caught: {exc.message} (Code: {exc.error_code})")
    
    error_response = StandardErrorResponse(
        success=False,
        message=exc.message,
        error=ErrorDetail(code=exc.error_code)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

# Exception handler for Pydantic Validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation failed: {str(exc.errors())}")
    
    error_response = StandardErrorResponse(
        success=False,
        message="Validation failed for the request payload.",
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            details=exc.errors()
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )

# General Catch-all Exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Unhandled system error: {str(exc)}", exc_info=True)
    
    error_response = StandardErrorResponse(
        success=False,
        message="An unexpected server error occurred.",
        error=ErrorDetail(code="INTERNAL_SERVER_ERROR")
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )

# Include API Routers under /api prefix
app.include_router(chat_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "success": True,
        "message": f"Welcome to {settings.APP_NAME}",
        "data": {
            "environment": settings.ENV,
            "status": "online",
            "version": "1.0.0"
        }
    }

@app.get("/health")
async def health_check():
    ai_manager = AIManager()
    
    # Query default provider health status
    default_provider = settings.DEFAULT_PROVIDER
    provider_healthy = await ai_manager.check_provider_health(default_provider)
    
    return {
        "success": True,
        "message": "Health status retrieved successfully",
        "data": {
            "app": settings.APP_NAME,
            "status": "healthy" if provider_healthy else "degraded",
            "environment": settings.ENV,
            "providers": {
                default_provider: "healthy" if provider_healthy else "unreachable"
            }
        }
    }

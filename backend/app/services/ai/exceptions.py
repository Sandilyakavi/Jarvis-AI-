class AIServiceException(Exception):
    """Base exception for all AI Service related errors."""
    def __init__(self, message: str, error_code: str = "AI_SERVICE_ERROR", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class ProviderNotFoundError(AIServiceException):
    """Raised when the requested AI provider is not found or not registered."""
    def __init__(self, provider_name: str):
        super().__init__(
            message=f"AI Provider '{provider_name}' is not registered or supported.",
            error_code="PROVIDER_NOT_FOUND",
            status_code=404
        )


class ProviderConnectionError(AIServiceException):
    """Raised when there is a connection/network issue calling the AI provider."""
    def __init__(self, provider_name: str, detail: str = ""):
        message = f"Connection to AI Provider '{provider_name}' failed."
        if detail:
            message += f" Details: {detail}"
        super().__init__(
            message=message,
            error_code="PROVIDER_CONNECTION_FAILED",
            status_code=503
        )


class ProviderAPIError(AIServiceException):
    """Raised when the AI provider returns an API error response."""
    def __init__(self, provider_name: str, detail: str, status_code: int = 502):
        super().__init__(
            message=f"AI Provider '{provider_name}' returned an error: {detail}",
            error_code="PROVIDER_API_ERROR",
            status_code=status_code
        )


class InvalidRequestError(AIServiceException):
    """Raised when the chat request format or payload is invalid."""
    def __init__(self, detail: str):
        super().__init__(
            message=f"Invalid AI Request: {detail}",
            error_code="INVALID_REQUEST",
            status_code=400
        )

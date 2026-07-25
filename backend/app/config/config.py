import os
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Explicitly load environment variables from .env if present
# This supports environment variables when python-dotenv is imported directly
load_dotenv()

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Jarvis AI API"
    ENV: str = "development"
    DEBUG: bool = True

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Settings
    ALLOWED_ORIGINS: str | List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # External APIs (for future sprints)
    GROQ_API_KEY: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and v:
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return []

# Instantiate settings
settings = Settings()

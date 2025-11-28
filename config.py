"""
Configuration module for FastAPI backend.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VECTOR_STORE_DIR = MODELS_DIR / "vectorstores"


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    app_name: str = "LangChain RAG Service"
    environment: Literal["dev", "prod"] = "dev"
    host: str = "0.0.0.0"
    port: int = 8000

    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    llm_provider: Literal["openai", "groq"] = "openai"
    llm_model_name: str = "gpt-4o-mini"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 3

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance."""
    settings = Settings()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = [
    "get_settings",
    "Settings",
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "VECTOR_STORE_DIR",
]



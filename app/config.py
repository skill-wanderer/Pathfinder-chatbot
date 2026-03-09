from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM Provider: "gemini" or "selfhost"
    LLM_PROVIDER: str = "gemini"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_LLM_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Self-hosted (OpenAI-compatible)
    SELFHOST_BASE_URL: str = "http://localhost:11434/v1"
    SELFHOST_API_KEY: str = "not-needed"
    SELFHOST_LLM_MODEL: str = "llama3"
    SELFHOST_EMBEDDING_MODEL: str = "nomic-embed-text"
    SELFHOST_EMBEDDING_DIMENSIONS: int = 3072

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "website_pages"

    # RAG
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

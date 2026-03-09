"""
Embedding service — abstracts Gemini vs self-hosted embeddings.
Both paths return a LangChain Embeddings instance so the rest of the
app stays provider-agnostic.
"""

from langchain_core.embeddings import Embeddings
from app.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBEDDING_MODEL}",
            google_api_key=settings.GEMINI_API_KEY,
        )

    if provider == "selfhost":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.SELFHOST_EMBEDDING_MODEL,
            openai_api_base=settings.SELFHOST_BASE_URL,
            openai_api_key=settings.SELFHOST_API_KEY,
            dimensions=settings.SELFHOST_EMBEDDING_DIMENSIONS,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'selfhost'.")

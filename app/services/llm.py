"""
LLM service — abstracts Gemini vs self-hosted chat models.
Returns a LangChain BaseChatModel so chains stay provider-agnostic.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from app.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

    if provider == "selfhost":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.SELFHOST_LLM_MODEL,
            base_url=settings.SELFHOST_BASE_URL,
            api_key=settings.SELFHOST_API_KEY,
            temperature=0.3,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'selfhost'.")

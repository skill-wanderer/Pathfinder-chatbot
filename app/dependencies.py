"""
Shared FastAPI dependencies — singletons for Qdrant client, embeddings, and LLM.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from qdrant_client import QdrantClient

from app.config import get_settings
from app.services.embeddings import build_embeddings
from app.services.llm import build_llm
from app.services.retriever import build_qdrant_client


@lru_cache
def get_qdrant() -> QdrantClient:
    return build_qdrant_client(get_settings())


@lru_cache
def get_embeddings() -> Embeddings:
    return build_embeddings(get_settings())


@lru_cache
def get_llm() -> BaseChatModel:
    return build_llm(get_settings())

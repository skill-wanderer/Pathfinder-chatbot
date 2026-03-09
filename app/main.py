"""
Pathfinder — RAG chatbot API powered by FastAPI, LangChain, Qdrant, and Gemini.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_embeddings, get_llm, get_qdrant
from app.models.schemas import HealthResponse
from app.routers.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up singletons on startup so first request isn't slow
    get_qdrant()
    get_embeddings()
    get_llm()
    yield


app = FastAPI(
    title="Pathfinder Chatbot API",
    description="RAG chatbot that answers questions from crawled website content stored in Qdrant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    settings = get_settings()
    qdrant_ok = False
    try:
        info = get_qdrant().get_collection(settings.QDRANT_COLLECTION)
        qdrant_ok = info is not None
    except Exception:
        pass
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        provider=settings.LLM_PROVIDER,
        qdrant_connected=qdrant_ok,
    )

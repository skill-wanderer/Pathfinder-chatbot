"""
Pathfinder — RAG chatbot API powered by FastAPI, LangChain, Qdrant, and Gemini.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import get_embeddings, get_llm, get_qdrant, set_pg_pool
from app.models.schemas import HealthResponse
from app.routers.chat import router as chat_router
from app.services.chat_log import close_pool, init_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up singletons on startup so first request isn't slow
    get_qdrant()
    get_embeddings()
    get_llm()

    # Initialise PostgreSQL connection pool for chat logging
    settings = get_settings()
    pool = await init_pool(settings)
    set_pg_pool(pool)

    yield

    # Shutdown: cancel cleanup task and close the PG pool
    await close_pool(pool)


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

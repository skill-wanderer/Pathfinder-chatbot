"""
PostgreSQL chat logging — stores full interaction context for traceability.

All writes are fire-and-forget (asyncio.create_task) so they never block
the chat API response. A periodic cleanup task deletes rows older than
CHAT_LOG_RETENTION_DAYS.
"""

import asyncio
import json
import logging

import asyncpg

from app.config import Settings

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_logs (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Request fields
    question        TEXT        NOT NULL,
    domains         JSONB,
    url_filter      TEXT,
    personality     TEXT,
    history         JSONB,

    -- RAG retrieval
    retrieved_context TEXT,
    sources         JSONB,

    -- Full prompt sent to LLM
    system_prompt   TEXT,
    messages        JSONB,

    -- LLM response
    answer          TEXT,

    -- Metadata
    llm_provider    TEXT,
    llm_model       TEXT
);
"""

# Index on created_at for fast retention cleanup
CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at ON chat_logs (created_at);
"""

CLEANUP_SQL = """
DELETE FROM chat_logs WHERE created_at < now() - make_interval(days => $1);
"""

# Background cleanup task handle — kept so it can be cancelled on shutdown.
_cleanup_task: asyncio.Task | None = None


async def init_pool(settings: Settings) -> asyncpg.Pool:
    """Create a connection pool, ensure the table exists, and start the cleanup loop."""
    pool = await asyncpg.create_pool(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB,
        min_size=2,
        max_size=10,
    )
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
        await conn.execute(CREATE_INDEX_SQL)
    logger.info("Chat log database ready")

    # Start periodic cleanup
    global _cleanup_task
    _cleanup_task = asyncio.create_task(
        _cleanup_loop(pool, settings.CHAT_LOG_RETENTION_DAYS)
    )

    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Cancel the cleanup task and close the pool."""
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None
    await pool.close()


async def _cleanup_loop(pool: asyncpg.Pool, retention_days: int) -> None:
    """Delete logs older than retention_days. Runs once on startup, then every 6 hours.
    If retention_days is 0, logs are kept forever and this loop exits immediately."""
    if retention_days <= 0:
        logger.info("Chat log retention disabled (CHAT_LOG_RETENTION_DAYS=0) — logs kept forever")
        return
    while True:
        try:
            async with pool.acquire() as conn:
                result = await conn.execute(CLEANUP_SQL, retention_days)
            logger.info("Chat log cleanup done (retention=%d days): %s", retention_days, result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Chat log cleanup failed")
        await asyncio.sleep(6 * 3600)  # 6 hours


async def _insert_log(
    pool: asyncpg.Pool,
    *,
    question: str,
    domains: list[str] | None,
    url_filter: str | None,
    personality: str | None,
    history: list[dict],
    retrieved_context: str | None,
    sources: list[dict],
    system_prompt: str,
    messages: list[dict],
    answer: str,
    llm_provider: str,
    llm_model: str,
) -> None:
    """Actual DB insert — meant to be called inside a fire-and-forget task."""
    try:
        await pool.execute(
            """
            INSERT INTO chat_logs (
                question, domains, url_filter, personality, history,
                retrieved_context, sources, system_prompt, messages,
                answer, llm_provider, llm_model
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            """,
            question,
            json.dumps(domains) if domains else None,
            url_filter,
            personality,
            json.dumps(history),
            retrieved_context,
            json.dumps(sources),
            system_prompt,
            json.dumps(messages),
            answer,
            llm_provider,
            llm_model,
        )
    except Exception:
        logger.exception("Failed to log chat interaction")


def log_chat(
    pool: asyncpg.Pool,
    **kwargs,
) -> None:
    """Fire-and-forget: schedules the DB insert as a background task so the
    chat response is returned immediately without waiting for the write."""
    asyncio.create_task(_insert_log(pool, **kwargs))

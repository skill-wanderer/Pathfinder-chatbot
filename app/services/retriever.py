"""
Qdrant retriever service — searches the vector store with optional
domain / URL filtering and returns ranked chunks with metadata.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue, MatchText
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.models.schemas import Source


def build_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


def retrieve_chunks(
    question: str,
    embeddings: Embeddings,
    qdrant: QdrantClient,
    settings: Settings,
    domains: list[str] | None = None,
    url: str | None = None,
) -> tuple[str, list[Source]]:
    """
    Embed the question, search Qdrant with optional domain/url filter,
    and return (context_string, list[Source]).

    `domains` accepts a list of domains; results matching ANY of them are returned.
    """
    query_vector = embeddings.embed_query(question)

    # Build filter conditions
    must_conditions: list[FieldCondition] = []
    if domains:
        if len(domains) == 1:
            must_conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domains[0]))
            )
        else:
            must_conditions.append(
                FieldCondition(key="domain", match=MatchAny(any=domains))
            )
    if url:
        must_conditions.append(
            FieldCondition(key="url", match=MatchText(text=url))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = qdrant.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=settings.RAG_TOP_K,
        score_threshold=settings.RAG_SCORE_THRESHOLD,
        with_payload=True,
    )

    # Build context and sources
    context_parts: list[str] = []
    sources: list[Source] = []

    for point in results.points:
        p = point.payload
        context_parts.append(
            f"[Source: {p.get('title', '')} — {p.get('url', '')}]\n{p.get('text', '')}"
        )
        sources.append(
            Source(
                title=p.get("title", ""),
                url=p.get("url", ""),
                chunk_index=p.get("chunk_index", 0),
                total_chunks=p.get("total_chunks", 1),
                score=round(point.score, 4),
            )
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def list_domains(qdrant: QdrantClient, settings: Settings) -> list[str]:
    """Scroll through all points and return unique domains."""
    domains: set[str] = set()
    offset = None

    while True:
        result = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=["domain"],
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            domain_val = point.payload.get("domain")
            if domain_val:
                domains.add(domain_val)
        if next_offset is None:
            break
        offset = next_offset

    return sorted(domains)

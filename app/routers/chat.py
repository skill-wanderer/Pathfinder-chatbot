"""
Chat router — /api/chat and /api/domains endpoints.
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.dependencies import get_embeddings, get_llm, get_qdrant
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DomainListResponse,
)
from app.services.retriever import list_domains, retrieve_chunks

router = APIRouter(prefix="/api", tags=["chat"])

SYSTEM_PROMPT = """\
You are **Pathfinder**, a helpful assistant that answers questions \
strictly based on the provided website context.

Rules:
1. Only use the context below to answer. Do NOT use prior knowledge.
2. If the context does not contain enough information, reply: \
   "I don't have enough information from the website to answer that."
3. Cite the source page title and URL when possible.
4. Be concise and accurate.
5. Use the conversation history for follow-up context, but still ground \
   every answer in the retrieved context.

Context:
{context}
"""


def _build_messages(system: str, history: list, question: str) -> list:
    """Build the message list from system prompt, history, and current question."""
    messages = [SystemMessage(content=system)]
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=question))
    return messages


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    settings = get_settings()

    # Merge domain / domains into a single list
    all_domains: list[str] | None = None
    if req.domains:
        all_domains = list(req.domains)
        if req.domain and req.domain not in all_domains:
            all_domains.append(req.domain)
    elif req.domain:
        all_domains = [req.domain]

    # Retrieve relevant chunks from Qdrant
    context, sources = retrieve_chunks(
        question=req.question,
        embeddings=get_embeddings(),
        qdrant=get_qdrant(),
        settings=settings,
        domains=all_domains,
        url=req.url,
    )

    if not sources:
        return ChatResponse(
            answer="I couldn't find any relevant information from the website for your question.",
            sources=[],
        )

    # Build messages with conversation history and invoke LLM
    system_text = SYSTEM_PROMPT.format(context=context)
    messages = _build_messages(system_text, req.history, req.question)
    llm = get_llm()

    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    content = result.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )

    return ChatResponse(answer=content, sources=sources)


@router.get("/domains", response_model=DomainListResponse)
async def domains():
    """Return all domains available in the vector store."""
    settings = get_settings()
    try:
        domain_list = list_domains(get_qdrant(), settings)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}") from exc
    return DomainListResponse(domains=domain_list)

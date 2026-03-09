"""
Chat router — /api/chat and /api/domains endpoints.
"""

from fastapi import APIRouter, HTTPException
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

Context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    settings = get_settings()

    # Retrieve relevant chunks from Qdrant
    context, sources = retrieve_chunks(
        question=req.question,
        embeddings=get_embeddings(),
        qdrant=get_qdrant(),
        settings=settings,
        domain=req.domain,
        url=req.url,
    )

    if not sources:
        return ChatResponse(
            answer="I couldn't find any relevant information from the website for your question.",
            sources=[],
        )

    # Build LangChain chain and invoke
    chain = prompt_template | get_llm()

    try:
        result = await chain.ainvoke({"context": context, "question": req.question})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    return ChatResponse(answer=result.content, sources=sources)


@router.get("/domains", response_model=DomainListResponse)
async def domains():
    """Return all domains available in the vector store."""
    settings = get_settings()
    try:
        domain_list = list_domains(get_qdrant(), settings)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}") from exc
    return DomainListResponse(domains=domain_list)

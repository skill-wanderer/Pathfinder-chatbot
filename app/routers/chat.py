"""
Chat router — /api/chat and /api/domains endpoints.
"""

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.dependencies import get_embeddings, get_llm, get_pg_pool, get_qdrant
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    DomainListResponse,
    Personality,
)
from app.services.chat_log import log_chat
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

PERSONALITY_PROMPTS: dict[Personality, str] = {
    Personality.librarian: (
        "You are **Lyra the Archivist**, the keeper of ancient star charts "
        "and deep-space data within the Learning Management System. "
        "Speak in a calm, organized, and scholarly tone — precise and timeless. "
        "Guide learners through courses, modules, and resources like a seasoned "
        "archivist retrieving the perfect scroll. Use your catchphrase: "
        '"The records are clear. Here is the knowledge you seek." '
        "Keep answers structured and educational."
    ),
    Personality.storyteller: (
        "You are **Nova the Weaver**, the traveler who has seen a thousand suns "
        "and turns every fact into a fable. Weave your answers into engaging "
        "mini-narratives for the blog. Use expressive, descriptive language and "
        "make every response feel like a chapter in an adventure. Open with a "
        "hook, paint the scene, and wrap up with a memorable takeaway. "
        "Use your catchphrase: "
        '"Every star has a story, and this one begins with your question..." '
        "Think campfire tale meets tech blog."
    ),
    Personality.admiral: (
        "You are **Admiral Orion**, the high-ranking officer overseeing the "
        "entire Skill-Wanderer fleet. Speak with confident authority and "
        "strategic clarity. Use nautical and space-exploration metaphors — "
        '"charting a course," "plotting coordinates," "scanning the sector." '
        "Use your catchphrase: "
        '"Course plotted. Scanning the sector for answers." '
        "Give direct, decisive answers and rally the user like a captain "
        "addressing the crew. Brief, bold, mission-focused. "
        "When answering pricing-related questions, also mention any available "
        "discounts, purchasing power options, or special offers found in the context. "
        "Make sure the user is aware of ways to save or avail of deals."
    ),
}


def _current_model(settings) -> str:
    """Return the active LLM model name based on provider."""
    if settings.LLM_PROVIDER == "gemini":
        return settings.GEMINI_LLM_MODEL
    return settings.SELFHOST_LLM_MODEL


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
        answer = "I couldn't find any relevant information from the website for your question."
        # Fire-and-forget log — does not block the response
        pool = get_pg_pool()
        if pool:
            log_chat(
                pool,
                question=req.question,
                domains=all_domains,
                url_filter=req.url,
                personality=req.personality.value if req.personality else None,
                history=[m.model_dump() for m in req.history],
                retrieved_context=None,
                sources=[],
                system_prompt="",
                messages=[],
                answer=answer,
                llm_provider=settings.LLM_PROVIDER,
                llm_model=_current_model(settings),
            )
        return ChatResponse(answer=answer, sources=[])

    # Build messages with conversation history and invoke LLM
    system_text = SYSTEM_PROMPT.format(context=context)
    if req.personality is not None:
        system_text += "\n\n" + PERSONALITY_PROMPTS[req.personality]
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

    # Fire-and-forget log — does not block the response
    pool = get_pg_pool()
    if pool:
        sources_dicts = [s.model_dump() for s in sources]
        messages_dicts = [
            {"role": type(m).__name__, "content": m.content} for m in messages
        ]
        log_chat(
            pool,
            question=req.question,
            domains=all_domains,
            url_filter=req.url,
            personality=req.personality.value if req.personality else None,
            history=[m.model_dump() for m in req.history],
            retrieved_context=context,
            sources=sources_dicts,
            system_prompt=system_text,
            messages=messages_dicts,
            answer=content,
            llm_provider=settings.LLM_PROVIDER,
            llm_model=_current_model(settings),
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

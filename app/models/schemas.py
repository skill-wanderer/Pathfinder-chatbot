from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    domain: str | None = Field(
        None,
        description="Filter results to a specific domain (e.g. 'example.com')",
    )
    url: str | None = Field(
        None,
        description="Filter results to a specific URL prefix",
    )


class Source(BaseModel):
    title: str
    url: str
    chunk_index: int
    total_chunks: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    provider: str
    qdrant_connected: bool


class DomainListResponse(BaseModel):
    domains: list[str]

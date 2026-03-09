from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    domain: str | None = Field(
        None,
        description="Filter results to a single domain (e.g. 'example.com'). Use 'domains' for multiple.",
    )
    domains: list[str] | None = Field(
        None,
        description="Filter results to one or more domains (e.g. ['example.com', 'docs.example.com'])",
    )
    url: str | None = Field(
        None,
        description="Filter results to a specific URL prefix",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=50,
        description="Previous conversation turns (oldest first)",
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

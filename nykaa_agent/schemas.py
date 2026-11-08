"""Pydantic models: crew output schema, review verdict, API request/response."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    record_id: str
    found: bool = True
    status: Optional[str] = None
    order_value_inr: Optional[int] = None
    escalation_score: Optional[float] = Field(None, ge=0, le=1)
    recommend_escalation: bool = False


class CrewAnswer(BaseModel):
    """Every crew response MUST conform to this schema (Task 9)."""
    answer: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
    order: Optional[OrderInfo] = None
    grounded: bool
    refused: bool = False
    escalate: bool = False
    confidence: float = Field(0.0, ge=0, le=1)


class ReviewVerdict(BaseModel):
    """Autogen Final-Editor verdict (Task 14)."""
    approved: bool
    final_answer: str
    reason: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=50_000)
    session_id: str = Field("default", min_length=1, max_length=100)


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    sources: list[str] = []
    grounded: bool
    refused: bool = False
    escalate: bool = False
    order: Optional[OrderInfo] = None
    review: Optional[ReviewVerdict] = None
    guardrails_fired: list[str] = []
    cache_hit: bool = False
    tool_calls: dict[str, int] = {}
    resolved_question: Optional[str] = None
    ticket_id: Optional[str] = None
    estimated_tokens: int = 0
    duration_ms: float = 0.0


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_\-]+$")
    title: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=20, max_length=20_000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_indexed: dict[str, int]

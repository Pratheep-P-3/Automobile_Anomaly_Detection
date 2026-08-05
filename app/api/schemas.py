"""Pydantic request/response schemas for the FastAPI diagnose service."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DiagnoseRequest(BaseModel):
    query: str = Field(..., description="Diagnostic code (e.g. P0300) or free-text symptom description.")
    code: str | None = Field(None, description="Optional explicit diagnostic code to filter retrieval.")
    prompt_version: str | None = Field(None, description="Override the default prompt version (e.g. 'v1').")


class CitationSchema(BaseModel):
    doc_file: str
    heading: str
    code: str
    score: float


class DiagnoseResponse(BaseModel):
    query: str
    matched_code: str | None
    is_fallback: bool
    confidence: float
    explanation: str
    service_recommendation: dict
    citations: list[CitationSchema]


class FeedbackRequest(BaseModel):
    query: str
    matched_code: str | None = None
    rating: str = Field(..., pattern="^(up|down)$")
    comment: str = ""


class FeedbackResponse(BaseModel):
    status: str
    record: dict


class MonitoringStatsResponse(BaseModel):
    total_queries: int
    unresolved_count: int
    unresolved_rate: float
    low_confidence_count: int
    low_confidence_rate: float
    average_confidence: float
    recent_unresolved_queries: list[str]

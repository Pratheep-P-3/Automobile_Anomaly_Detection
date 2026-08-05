"""POST /diagnose - accepts a diagnostic code or symptom and returns a grounded answer."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import CitationSchema, DiagnoseRequest, DiagnoseResponse
from app.workflow.diagnose_workflow import diagnose

router = APIRouter(tags=["diagnose"])


@router.post("/diagnose", response_model=DiagnoseResponse)
def diagnose_endpoint(request: DiagnoseRequest) -> DiagnoseResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = diagnose(request.query, code=request.code, prompt_version=request.prompt_version)

    return DiagnoseResponse(
        query=result.query,
        matched_code=result.matched_code,
        is_fallback=result.is_fallback,
        confidence=result.confidence,
        explanation=result.explanation,
        service_recommendation=result.service_recommendation,
        citations=[CitationSchema(**vars(c)) for c in result.citations],
    )

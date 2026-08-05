"""POST /feedback - capture service engineer feedback on a diagnostic answer."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import FeedbackRequest, FeedbackResponse
from app.services.feedback_store import submit_feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(request: FeedbackRequest) -> FeedbackResponse:
    record = submit_feedback(
        query=request.query,
        matched_code=request.matched_code,
        rating=request.rating,
        comment=request.comment,
    )
    return FeedbackResponse(status="ok", record=record)

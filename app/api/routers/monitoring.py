"""GET /monitoring/stats - unresolved query / low-confidence answer / retrieval monitoring."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import MonitoringStatsResponse
from app.services.feedback_store import get_feedback_summary
from app.services.monitoring import get_monitoring_stats

router = APIRouter(tags=["monitoring"])


@router.get("/monitoring/stats", response_model=MonitoringStatsResponse)
def monitoring_stats_endpoint() -> MonitoringStatsResponse:
    return MonitoringStatsResponse(**get_monitoring_stats())


@router.get("/monitoring/feedback")
def monitoring_feedback_endpoint() -> dict:
    return get_feedback_summary()

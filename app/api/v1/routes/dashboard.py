"""
Dashboard summary endpoint.

GET /api/v1/dashboard/summary

Returns aggregated statistics for the authenticated user from the
analysis_results table. All values default to zero/null when the
user has no stored analyses so the frontend never receives undefined.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_optional_user
from app.models.user import User
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_EMPTY_SUMMARY = {
    "total_analyses": 0,
    "high_risk_count": 0,
    "average_confidence": None,
    "average_duration_seconds": None,
    "weekly_trend": [],
    "risk_distribution": {"low": 0, "moderate": 0, "high": 0, "critical": 0},
    "recent_analyses": [],
}


@router.get("/summary")
async def dashboard_summary(
    current_user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Aggregated dashboard statistics for the current user.

    Returns zeros/empty arrays when the user is unauthenticated or has no
    recorded analyses so the frontend never needs to handle None at top level.
    """
    if current_user is None:
        data = dict(_EMPTY_SUMMARY)
    else:
        svc = AnalysisService(db)
        data = await svc.get_dashboard_summary(current_user.id)

    data["system_status"] = {
        "online":     True,
        "version":    settings.APP_VERSION,
        "model":      "EfficientNet-B0 v6",
        "last_check": datetime.now(timezone.utc).isoformat(),
    }
    return data

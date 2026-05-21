"""
AnalysisService — persists UnifiedAnalysisSession results and provides
query helpers for dashboard statistics.
"""
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis_result import AnalysisResult
from app.repositories.analysis_repository import AnalysisRepository
from app.schemas.medical_reasoning import UnifiedAnalysisSession

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self, db: AsyncSession) -> None:
        self._repo = AnalysisRepository(db)

    async def save_from_session(
        self,
        session: UnifiedAnalysisSession,
        *,
        user_id: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> AnalysisResult:
        """
        Persist a completed analysis session.

        Extracts the fields needed for dashboard statistics from the already-
        assembled UnifiedAnalysisSession response object so we never re-run
        inference just for persistence.
        """
        summary_text: Optional[str] = None
        if session.explainability and session.explainability.summary:
            summary_text = session.explainability.summary[:2000]

        trust_tier: Optional[str] = None
        trust_score: Optional[float] = None
        if session.trust:
            trust_tier = session.trust.trust_tier
            trust_score = session.trust.trust_score

        clinical_used = bool(session.clinical and session.clinical.provided)
        analysis_type = "HYBRID" if clinical_used else "IMAGE"

        logger.info(
            "Persist [%s]: session_id=%s user_id=%s tier=%s score=%.4f",
            analysis_type, session.session_id, user_id,
            session.risk.risk_tier, session.risk.final_score,
        )
        record = await self._repo.create(
            session_id=session.session_id,
            user_id=user_id,
            risk_tier=session.risk.risk_tier,
            final_score=session.risk.final_score,
            imaging_score=session.risk.imaging_score,
            clinical_modifier=session.risk.clinical_modifier,
            predicted_class=session.imaging.predicted_class,
            calibrated_confidence=session.imaging.calibrated_confidence,
            image_used=True,
            clinical_used=clinical_used,
            ood_guard_applied=session.ood_guard_applied,
            trust_tier=trust_tier,
            trust_score=trust_score,
            summary=summary_text,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
            pipeline_version=session.pipeline_version,
            model_version=session.model_version,
            analysis_type=analysis_type,
        )
        logger.info(
            "Persisted [%s] session_id=%s user_id=%s tier=%s score=%.4f row_id=%s",
            analysis_type, session.session_id, user_id,
            session.risk.risk_tier, session.risk.final_score, record.id,
        )
        return record

    async def save_clinical_only(
        self,
        *,
        session_id: str,
        risk_tier: str,
        final_score: float,
        user_id: Optional[int] = None,
        summary: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> AnalysisResult:
        """Persist a clinical-only (no-image) analysis from the frontend scorer."""
        logger.info(
            "Persist [CLINICAL_ONLY]: session_id=%s user_id=%s tier=%s score=%.4f",
            session_id, user_id, risk_tier, final_score,
        )
        record = await self._repo.create(
            session_id=session_id,
            user_id=user_id,
            risk_tier=risk_tier,
            final_score=final_score,
            imaging_score=final_score,
            clinical_modifier=0.0,
            predicted_class="clinical_only",
            calibrated_confidence=None,
            image_used=False,
            clinical_used=True,
            ood_guard_applied=False,
            trust_tier=None,
            trust_score=None,
            summary=summary[:2000] if summary else None,
            duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
            pipeline_version="clinical_v1",
            model_version="clinical_scorer_v1",
            analysis_type="CLINICAL_ONLY",
        )
        logger.info(
            "Persisted [CLINICAL_ONLY] session_id=%s user_id=%s tier=%s score=%.4f row_id=%s",
            session_id, user_id, risk_tier, final_score, record.id,
        )
        return record

    # ── Dashboard query helpers ───────────────────────────────────────────────

    async def get_user_summary(self, user_id: int) -> dict:
        """
        Returns aggregated stats for a user.  Used by the dashboard to populate
        total-count, tier breakdown, and recent-analyses widgets.
        """
        total = await self._repo.count_total_by_user(user_id)
        by_tier = await self._repo.count_by_risk_tier(user_id)
        recent = await self._repo.get_recent_by_user(user_id, limit=5)
        daily = await self._repo.get_daily_counts(user_id, days=30)
        return {
            "total_analyses": total,
            "by_risk_tier": by_tier,
            "daily_counts_30d": daily,
            "recent_analyses": [
                {
                    "session_id": r.session_id,
                    "risk_tier": r.risk_tier,
                    "final_score": r.final_score,
                    "predicted_class": r.predicted_class,
                    "clinical_used": r.clinical_used,
                    "created_at": r.created_at.isoformat(),
                }
                for r in recent
            ],
        }

    async def get_recent_analyses(
        self, user_id: int, *, limit: int = 10
    ) -> list[AnalysisResult]:
        return await self._repo.get_recent_by_user(user_id, limit=limit)

    async def get_dashboard_summary(self, user_id: int) -> dict:
        """Full summary payload for the dashboard /summary endpoint."""
        total     = await self._repo.count_total_by_user(user_id)
        high_risk = await self._repo.count_high_risk_recent(user_id, days=30)
        avg_conf  = await self._repo.get_average_confidence(user_id)
        avg_dur   = await self._repo.get_average_duration_ms(user_id)
        by_tier   = await self._repo.count_by_risk_tier(user_id)
        daily_raw = await self._repo.get_daily_counts(user_id, days=7)
        recent    = await self._repo.get_recent_by_user(user_id, limit=5)

        # Weekly trend — fill zeros for every day in the window
        today = date.today()
        daily_map = {d["date"]: d["count"] for d in daily_raw}
        weekly_trend = [
            {
                "date": str(today - timedelta(days=6 - i)),
                "count": daily_map.get(str(today - timedelta(days=6 - i)), 0),
            }
            for i in range(7)
        ]

        # Risk distribution as absolute counts keyed by lowercase short name
        risk_dist = {
            "low":      by_tier.get("LOW", 0),
            "moderate": by_tier.get("MODERATE", 0),
            "high":     by_tier.get("HIGH_DIFFERENTIAL_RISK", 0),
            "critical": by_tier.get("CRITICAL_PULMONARY_RISK", 0),
        }

        return {
            "total_analyses":          total,
            "high_risk_count":         high_risk,
            "average_confidence":      round(avg_conf, 4) if avg_conf is not None else None,
            "average_duration_seconds": round(avg_dur / 1000, 2) if avg_dur is not None else None,
            "weekly_trend":            weekly_trend,
            "risk_distribution":       risk_dist,
            "recent_analyses": [
                {
                    "id":         r.session_id,
                    "created_at": r.created_at.isoformat(),
                    "risk_score": round(r.final_score, 4),
                    "risk_tier":  r.risk_tier,
                    "confidence": round(r.calibrated_confidence, 4) if r.calibrated_confidence is not None else None,
                }
                for r in recent
            ],
        }

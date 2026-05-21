from datetime import date, timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis_result import AnalysisResult

_HIGH_RISK_TIERS = ("HIGH_DIFFERENTIAL_RISK", "CRITICAL_PULMONARY_RISK")


class AnalysisRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, **kwargs) -> AnalysisResult:
        record = AnalysisResult(**kwargs)
        self.db.add(record)
        await self.db.flush()
        await self.db.refresh(record)
        return record

    async def get_by_session_id(self, session_id: str) -> Optional[AnalysisResult]:
        result = await self.db.execute(
            select(AnalysisResult).where(AnalysisResult.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_recent_by_user(
        self, user_id: int, *, limit: int = 10
    ) -> list[AnalysisResult]:
        result = await self.db.execute(
            select(AnalysisResult)
            .where(AnalysisResult.user_id == user_id)
            .order_by(AnalysisResult.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_total_by_user(self, user_id: int) -> int:
        result = await self.db.execute(
            select(func.count()).select_from(AnalysisResult)
            .where(AnalysisResult.user_id == user_id)
        )
        return result.scalar_one() or 0

    async def count_by_risk_tier(self, user_id: int) -> dict[str, int]:
        result = await self.db.execute(
            select(AnalysisResult.risk_tier, func.count())
            .where(AnalysisResult.user_id == user_id)
            .group_by(AnalysisResult.risk_tier)
        )
        return {row[0]: row[1] for row in result.all()}

    async def get_daily_counts(
        self, user_id: int, *, days: int = 30
    ) -> list[dict]:
        cutoff = date.today() - timedelta(days=days - 1)
        result = await self.db.execute(
            select(
                func.date(AnalysisResult.created_at).label("day"),
                func.count().label("count"),
            )
            .where(
                AnalysisResult.user_id == user_id,
                func.date(AnalysisResult.created_at) >= cutoff,
            )
            .group_by(func.date(AnalysisResult.created_at))
            .order_by(func.date(AnalysisResult.created_at))
        )
        return [{"date": str(row.day), "count": row.count} for row in result.all()]

    async def count_high_risk_recent(self, user_id: int, *, days: int = 30) -> int:
        cutoff = date.today() - timedelta(days=days - 1)
        result = await self.db.execute(
            select(func.count())
            .select_from(AnalysisResult)
            .where(
                AnalysisResult.user_id == user_id,
                AnalysisResult.risk_tier.in_(_HIGH_RISK_TIERS),
                func.date(AnalysisResult.created_at) >= cutoff,
            )
        )
        return result.scalar_one() or 0

    async def get_average_confidence(self, user_id: int) -> Optional[float]:
        result = await self.db.execute(
            select(func.avg(AnalysisResult.calibrated_confidence)).where(
                AnalysisResult.user_id == user_id,
                AnalysisResult.calibrated_confidence.isnot(None),
            )
        )
        val = result.scalar_one_or_none()
        return float(val) if val is not None else None

    async def get_average_duration_ms(self, user_id: int) -> Optional[float]:
        result = await self.db.execute(
            select(func.avg(AnalysisResult.duration_ms)).where(
                AnalysisResult.user_id == user_id,
                AnalysisResult.duration_ms.isnot(None),
            )
        )
        val = result.scalar_one_or_none()
        return float(val) if val is not None else None

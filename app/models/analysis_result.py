from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Unique session identifier produced by the reasoning engine
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)

    # NULL when the request arrived without a valid JWT
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # ── Risk verdict ──────────────────────────────────────────────────────────
    risk_tier: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    final_score: Mapped[float] = mapped_column(nullable=False)
    imaging_score: Mapped[Optional[float]] = mapped_column(nullable=True)
    clinical_modifier: Mapped[Optional[float]] = mapped_column(nullable=True)

    # ── Classification ────────────────────────────────────────────────────────
    predicted_class: Mapped[str] = mapped_column(String(80), nullable=False)
    calibrated_confidence: Mapped[Optional[float]] = mapped_column(nullable=True)

    # ── Inputs used ───────────────────────────────────────────────────────────
    image_used: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    clinical_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ood_guard_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # ── Trust / calibration ───────────────────────────────────────────────────
    trust_tier: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    trust_score: Mapped[Optional[float]] = mapped_column(nullable=True)

    # ── Narrative summary (truncated at 2 000 chars) ──────────────────────────
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Performance ───────────────────────────────────────────────────────────
    duration_ms: Mapped[Optional[float]] = mapped_column(nullable=True)

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    pipeline_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # ── Analysis type: IMAGE | CLINICAL_ONLY | HYBRID ─────────────────────────
    analysis_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

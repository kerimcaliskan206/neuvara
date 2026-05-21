"""create analysis_results table

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-21

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("risk_tier", sa.String(40), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("imaging_score", sa.Float(), nullable=True),
        sa.Column("clinical_modifier", sa.Float(), nullable=True),
        sa.Column("predicted_class", sa.String(80), nullable=False),
        sa.Column("calibrated_confidence", sa.Float(), nullable=True),
        sa.Column(
            "image_used",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "clinical_used",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "ood_guard_applied",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("trust_tier", sa.String(40), nullable=True),
        sa.Column("trust_score", sa.Float(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("pipeline_version", sa.String(32), nullable=True),
        sa.Column("model_version", sa.String(32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_analysis_results_session_id"),
        "analysis_results",
        ["session_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_analysis_results_user_id"),
        "analysis_results",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_analysis_results_risk_tier"),
        "analysis_results",
        ["risk_tier"],
        unique=False,
    )
    op.create_index(
        op.f("ix_analysis_results_created_at"),
        "analysis_results",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_analysis_results_created_at"), table_name="analysis_results")
    op.drop_index(op.f("ix_analysis_results_risk_tier"), table_name="analysis_results")
    op.drop_index(op.f("ix_analysis_results_user_id"), table_name="analysis_results")
    op.drop_index(op.f("ix_analysis_results_session_id"), table_name="analysis_results")
    op.drop_table("analysis_results")

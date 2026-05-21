"""add analysis_type to analysis_results

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-21

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_results",
        sa.Column("analysis_type", sa.String(20), nullable=True),
    )
    op.create_index(
        op.f("ix_analysis_results_analysis_type"),
        "analysis_results",
        ["analysis_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_analysis_results_analysis_type"),
        table_name="analysis_results",
    )
    op.drop_column("analysis_results", "analysis_type")

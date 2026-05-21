"""widen analysis_results.model_version to varchar(64)

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-21

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "analysis_results",
        "model_version",
        type_=sa.String(64),
        existing_type=sa.String(32),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "analysis_results",
        "model_version",
        type_=sa.String(32),
        existing_type=sa.String(64),
        existing_nullable=True,
    )

import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.password_reset_token import PasswordResetToken

_TOKEN_EXPIRY_HOURS = 1


class PasswordResetTokenRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, user_id: int) -> PasswordResetToken:
        now = datetime.now(timezone.utc)
        record = PasswordResetToken(
            user_id=user_id,
            token=secrets.token_urlsafe(32),
            expires_at=now + timedelta(hours=_TOKEN_EXPIRY_HOURS),
            created_at=now,
        )
        self.db.add(record)
        await self.db.flush()
        return record

    async def get_by_token(self, token: str) -> PasswordResetToken | None:
        result = await self.db.execute(
            select(PasswordResetToken).where(PasswordResetToken.token == token)
        )
        return result.scalar_one_or_none()

    async def mark_used(self, record: PasswordResetToken) -> None:
        record.used_at = datetime.now(timezone.utc)
        await self.db.flush()

    async def invalidate_user_tokens(self, user_id: int) -> None:
        now = datetime.now(timezone.utc)
        await self.db.execute(
            update(PasswordResetToken)
            .where(PasswordResetToken.user_id == user_id, PasswordResetToken.used_at.is_(None))
            .values(used_at=now)
        )

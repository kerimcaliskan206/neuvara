"""
AuthService — registration, login, password reset.

Every write path explicitly calls db.flush() to push changes to PostgreSQL
within the current transaction.  The caller (route) must call db.commit()
afterwards.  The get_db dependency commits as a safety net, but routes that
write data commit explicitly so there is no ambiguity.
"""
import logging
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.jwt import create_access_token
from app.core.security import hash_password, verify_password
from app.repositories.password_reset_token_repository import PasswordResetTokenRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)
from app.services.email_service import EmailDeliveryError, send_password_reset_email

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = UserRepository(db)
        self.token_repo = PasswordResetTokenRepository(db)

    # ── Register ──────────────────────────────────────────────────────────────

    async def register(self, data: RegisterRequest) -> UserResponse:
        logger.info("Register: request received (email=%s username=%s)", data.email, data.username)

        # ── Uniqueness checks ──────────────────────────────────────────────
        if await self.repo.get_by_email(data.email):
            logger.info("Register: email already exists (email=%s)", data.email)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Bu e-posta adresi zaten kayıtlı.",
            )
        if await self.repo.get_by_username(data.username):
            logger.info("Register: username already taken (username=%s)", data.username)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Bu kullanıcı adı zaten kullanımda.",
            )

        # ── Hash + create ──────────────────────────────────────────────────
        logger.info("Register: hashing password (email=%s)", data.email)
        hashed_pw = hash_password(data.password)

        logger.info("Register: creating user row (email=%s)", data.email)
        user = await self.repo.create(
            username=data.username,
            email=data.email,
            hashed_password=hashed_pw,
        )
        # repo.create calls flush() + refresh() — user.id is now populated from DB
        logger.info(
            "Register: user flushed (id=%s email=%s is_active=%s)",
            user.id, user.email, user.is_active,
        )

        # ── Post-create verification (within same transaction) ─────────────
        verify = await self.repo.get_by_id(user.id)
        if verify is None:
            logger.error(
                "Register: post-create verification FAILED — user_id=%s not visible in session",
                user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Kayıt oluşturulamadı. Lütfen tekrar deneyin.",
            )
        logger.info(
            "Register: DB verification passed (id=%s email=%s hash_len=%s)",
            verify.id, verify.email, len(verify.hashed_password),
        )

        return UserResponse.model_validate(user)

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, data: LoginRequest) -> TokenResponse:
        logger.info("Login: request received (email=%s)", data.email)

        user = await self.repo.get_by_email(data.email)
        if not user:
            logger.info("Login: no account found for email=%s", data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Geçersiz giriş bilgileri.",
            )

        logger.info(
            "Login: user found (id=%s email=%s is_active=%s)",
            user.id, user.email, user.is_active,
        )

        if not verify_password(data.password, user.hashed_password):
            logger.info("Login: password mismatch (email=%s)", data.email)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Geçersiz giriş bilgileri.",
            )

        if not user.is_active:
            logger.info("Login: account inactive (id=%s email=%s)", user.id, user.email)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Hesap devre dışı.",
            )

        token = create_access_token(subject=user.id)
        logger.info("Login: JWT issued for user_id=%s email=%s", user.id, user.email)
        return TokenResponse(access_token=token)

    # ── Forgot password ───────────────────────────────────────────────────────

    async def forgot_password(self, data: ForgotPasswordRequest) -> None:
        logger.info("ForgotPassword: request received (email=%s)", data.email)

        user = await self.repo.get_by_email(data.email)
        if not user:
            logger.info(
                "ForgotPassword: no account for email=%s — returning 200 silently (security)",
                data.email,
            )
            return

        logger.info("ForgotPassword: invalidating existing tokens for user_id=%s", user.id)
        await self.token_repo.invalidate_user_tokens(user.id)

        logger.info("ForgotPassword: creating reset token for user_id=%s", user.id)
        record = await self.token_repo.create(user.id)

        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={record.token}"

        if settings.DEBUG:
            logger.info(
                "ForgotPassword: token created for user_id=%s\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "  RESET PASSWORD URL:\n"
                "  %s\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                user.id,
                reset_url,
            )

        try:
            await send_password_reset_email(
                to_email=user.email,
                username=user.username,
                reset_url=reset_url,
            )
        except EmailDeliveryError:
            # Email failure must not expose infra details or break the auth flow.
            # Token is already persisted — user can retry via forgot-password.
            logger.error(
                "ForgotPassword: email delivery failed for user_id=%s — token persisted, user may retry",
                user.id,
            )

    # ── Reset password ────────────────────────────────────────────────────────

    async def reset_password(self, data: ResetPasswordRequest) -> None:
        logger.info("ResetPassword: request received (token=%s…)", data.token[:8])

        record = await self.token_repo.get_by_token(data.token)
        if record is None or record.used_at is not None:
            logger.info(
                "ResetPassword: invalid or already-used token (token=%s…)", data.token[:8]
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz veya kullanılmış sıfırlama bağlantısı.",
            )

        if record.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
            logger.info(
                "ResetPassword: expired token (token=%s… expires=%s)",
                data.token[:8], record.expires_at,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sıfırlama bağlantısının süresi dolmuş.",
            )

        user = await self.repo.get_by_id(record.user_id)
        if user is None:
            logger.error("ResetPassword: user_id=%s not found for valid token", record.user_id)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Kullanıcı bulunamadı.",
            )

        logger.info("ResetPassword: updating password for user_id=%s email=%s", user.id, user.email)
        user.hashed_password = hash_password(data.new_password)
        # Explicitly flush the password change before marking token used,
        # so both changes are in the same flush and neither can be missed.
        await self.db.flush()

        await self.token_repo.mark_used(record)
        logger.info("ResetPassword: password updated and token marked used (user_id=%s)", user.id)

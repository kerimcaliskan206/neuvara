import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    logger.info("Route /register: handler entered (email=%s)", data.email)
    try:
        user = await AuthService(db).register(data)
        await db.commit()
        logger.info("Route /register: committed (email=%s)", data.email)
        return user
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        logger.exception("Route /register: unexpected error (email=%s)", data.email)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Kayıt işlemi sırasında beklenmeyen bir hata oluştu.",
        )


@router.post("/login", response_model=TokenResponse)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    logger.info("Route /login: handler entered (email=%s)", data.email)
    return await AuthService(db).login(data)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/forgot-password", status_code=200)
async def forgot_password(data: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    logger.info("Route /forgot-password: handler entered (email=%s)", data.email)
    try:
        await AuthService(db).forgot_password(data)
        await db.commit()
        logger.info("Route /forgot-password: committed (email=%s)", data.email)
        return JSONResponse({"success": True, "message": "Şifre sıfırlama bağlantısı gönderildi."})
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        logger.exception("Route /forgot-password: unexpected error (email=%s)", data.email)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Şifre sıfırlama isteği işlenemedi.",
        )


@router.post("/reset-password", status_code=200)
async def reset_password(data: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    logger.info("Route /reset-password: handler entered (token=%s…)", data.token[:8])
    try:
        await AuthService(db).reset_password(data)
        await db.commit()
        logger.info("Route /reset-password: committed (token=%s…)", data.token[:8])
        return JSONResponse({"success": True, "message": "Şifreniz başarıyla güncellendi."})
    except HTTPException:
        await db.rollback()
        raise
    except Exception:
        logger.exception("Route /reset-password: unexpected error (token=%s…)", data.token[:8])
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Şifre sıfırlama işlemi tamamlanamadı.",
        )


# ── Debug endpoint (development only) ────────────────────────────────────────

@router.get("/debug-users", include_in_schema=False)
async def debug_users(db: AsyncSession = Depends(get_db)):
    """Returns all registered users. Only available when DEBUG=true."""
    if not settings.DEBUG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    users = await UserRepository(db).get_all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]

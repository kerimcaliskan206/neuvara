import time

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db

router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.time()


@router.get("")
async def health_check():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@router.get("/db")
async def health_check_db(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "database": "unreachable", "detail": str(exc)},
        )


@router.get("/full")
async def health_check_full(db: AsyncSession = Depends(get_db)):
    db_status = "connected"
    db_ok = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unreachable"
        db_ok = False

    overall = "ok" if db_ok else "degraded"
    code = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=code,
        content={
            "status": overall,
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "uptime_seconds": round(time.time() - _start_time, 1),
            "checks": {"database": db_status},
        },
    )

from fastapi import APIRouter

from app.api.v1.routes import health
from app.api.v1.routes.dashboard import router as dashboard_router
from app.api.v1.routes.medical_analysis import router as medical_router
from app.modules.ai.routes import router as ai_router
from app.modules.auth.routes import router as auth_router
from app.modules.fusion.routes import router as fusion_router
from app.modules.ml.routes import router as ml_router
from app.modules.vision.routes import router as vision_router

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(dashboard_router)
api_router.include_router(auth_router)
api_router.include_router(ml_router)
api_router.include_router(vision_router)
api_router.include_router(fusion_router)
api_router.include_router(ai_router)
api_router.include_router(medical_router)

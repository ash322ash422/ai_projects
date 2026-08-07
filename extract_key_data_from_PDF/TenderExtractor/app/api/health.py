from fastapi import APIRouter

from app.schemas.response import HealthResponse
from app.settings import settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(app_name=settings.APP_NAME, env=settings.ENV)
